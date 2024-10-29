import torch
from dataclasses import dataclass, field
from .universal_decision_transformer_sb3 import UDT
from .discrete_decision_transformer_sb3 import DiscreteDecisionTransformerSb3
from .models.model_utils import sample_from_logits


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample=None

    def reset(self):
        self.max_seqlen = self.max_seqlen
        self.max_batch_size = self.max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()
    

class DecisionMamba(UDT):

    def __init__(self, policy, env, use_inference_cache=False, **kwargs):
        super().__init__(policy, env, **kwargs)
        self.use_inference_cache = use_inference_cache
        self.inference_params = None
        if self.use_inference_cache:
            self.inference_params = InferenceParams(
                max_seqlen=self.policy.config.max_length, 
                max_batch_size=1,
            )
            
    def get_action_pred(self, policy, states, actions, rewards, returns_to_go, timesteps, attention_mask,
                        deterministic, prompt,  is_eval=False, task_id=None, env_act_dim=None):
        if self.use_inference_cache:
            # only last step
            states, actions, rewards, returns_to_go, timesteps, attention_mask = states[:, -1:], actions[:, -1:],\
                rewards[:, -1:], returns_to_go[:, -1:], timesteps[:, -1:], attention_mask[:, -1:]
        
        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            policy_output = policy(
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask,
                return_dict=True,
                deterministic=deterministic,
                prompt=prompt,
                task_id=task_id,
                ddp_kwargs=self.ddp_kwargs,
                inference_params=self.inference_params
            )
            
        if not is_eval and self.num_timesteps % 10000 == 0 and self.log_attn_maps:
            self._record_attention_maps(policy_output.attentions, step=self.num_timesteps, prefix="rollout")
            if policy_output.cross_attentions is not None:
                self._record_attention_maps(policy_output.cross_attentions, step=self.num_timesteps,
                                            prefix="rollout_cross", lower_triu=False)
        action_preds = policy_output.action_preds
        if env_act_dim is not None: 
            action_preds = action_preds[..., :env_act_dim]
        return action_preds[0, -1], action_preds[0, -1]
    

class DiscreteDecisionMamba(DiscreteDecisionTransformerSb3, DecisionMamba):

    def get_action_pred(self, policy, states, actions, rewards, returns_to_go, timesteps, attention_mask,
                        deterministic, prompt, is_eval=False, task_id=None, env_act_dim=None):
        inputs = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "returns_to_go": returns_to_go,
            "timesteps": timesteps,
            "attention_mask": attention_mask,
            "return_dict": True,
            "deterministic": deterministic,
            "prompt": prompt,
            "task_id": task_id,
            "ddp_kwargs": self.ddp_kwargs,
            "inference_params": self.inference_params
        }
        
        if self.use_inference_cache:
            inputs.update({"states": states[:, -1:], "actions": actions[:, -1:], "rewards": rewards[:, -1:], 
                            "returns_to_go": returns_to_go[:, -1:], "timesteps": timesteps[:, -1:], 
                            "attention_mask": attention_mask[:, -1:]})
        
        # exper-action inference mechanism
        if self.target_return_type == "infer":
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                policy_output = policy(**inputs)
            return_logits = policy_output.return_preds[:, -1]
            return_sample = policy.sample_from_rtg_logits(return_logits, **self.rtg_sample_kwargs)
            inputs["returns_to_go"][0, -1] = return_sample
   
        # autoregressive action prediction
        # e.g., for discretizes continuous action space need to predict each action dim after another
        act_dim = actions.shape[-1] if env_act_dim is None else env_act_dim
        for i in range(act_dim):
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                policy_output = policy(**inputs)
            
            if not is_eval and self.num_timesteps % 10000 == 0 and self.log_attn_maps:
                self._record_attention_maps(policy_output.attentions, step=self.num_timesteps, prefix="rollout")
                if policy_output.cross_attentions is not None:
                    self._record_attention_maps(policy_output.cross_attentions, step=self.num_timesteps + i,
                                                prefix="rollout_cross", lower_triu=False)
            if self.a_sample_kwargs is not None: 
                action_logits = policy_output.action_logits[0, -1, i]
                inputs["actions"][0, -1, i] = sample_from_logits(action_logits, **self.a_sample_kwargs)
            else:     
                inputs["actions"][0, -1, i] = policy_output.action_preds[0, -1, i]
        
        action = inputs["actions"][0, -1]
        if env_act_dim is not None: 
            action = action[:act_dim]
        return action, inputs["returns_to_go"][0, -1] if self.target_return_type == "infer" else action
