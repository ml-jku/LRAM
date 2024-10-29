import torch
import torch.nn as nn
from dataclasses import dataclass
from gym import spaces
from transformers import DecisionTransformerModel
from transformers.models.decision_transformer.modeling_decision_transformer import DecisionTransformerOutput, Conv1D
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, DiagGaussianDistribution
from stable_baselines3.common import preprocessing
from .custom_dt_model import CustomDTGPT2Model
from .image_encoders import make_image_encoder
from .rms_norm import LlamaRMSNorm
from .model_utils import symlog


@dataclass
class OnlineDecisionTransformerOutput(DecisionTransformerOutput):
    action_log_probs: torch.FloatTensor = None
    reward_preds: torch.FloatTensor = None
    action_logits: torch.FloatTensor = None
    entropy: torch.FloatTensor = None
    last_encoder_output: torch.FloatTensor = None
    prompt_infos: dict = None
    cross_attentions: torch.FloatTensor = None
    past_key_values: torch.FloatTensor = None


class OnlineDecisionTransformerModel(DecisionTransformerModel):

    def __init__(
        self,
        config,
        observation_space,
        action_space,
        num_task_heads=1,
        n_layer_head=1,
        use_time_embds=True,
        rtg_condition=True,
        reward_condition=False,
        action_condition=True,
        stochastic_policy=True,
        relative_pos_embds=False,
        global_pos_embds=False,
        both_time_embds=False,
        use_crossattn_mask=True,
        symlog_transform=False,
        separate_ln=False,
        img_is_encoded=False, 
        encoder_kwargs=None,
        orig_act_dim=None,
        max_act_dim=None, 
        inf_dummy_batch_size=None,
        p_mask=0, 
        p_token_drop=0
    ):
        super().__init__(config)
        self.num_task_heads = num_task_heads
        self.n_layer_head = n_layer_head
        self.stochastic_policy = stochastic_policy
        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_condition = reward_condition
        self.rtg_condition = rtg_condition
        self.action_condition = action_condition
        self.use_time_embds = use_time_embds
        self.relative_pos_embds = relative_pos_embds
        self.global_pos_embds = global_pos_embds
        self.both_time_embds = both_time_embds
        self.use_crossattn_mask = use_crossattn_mask
        self.symlog_transform = symlog_transform
        self.separate_ln = separate_ln
        self.orig_act_dim = orig_act_dim
        self.max_act_dim = max_act_dim
        self.p_mask = p_mask
        self.p_token_drop = p_token_drop
        self.img_is_encoded = img_is_encoded
        self.inf_dummy_batch_size = inf_dummy_batch_size
        self.encoder_kwargs = encoder_kwargs if encoder_kwargs is not None else {}
        self.use_fast_attn = config.use_fast_attn if hasattr(config, "use_fast_attn") else False
        # in original implementation predict_action is a linear layer. We set it None and make a function out of it
        del self.predict_action
        self.is_image_space = preprocessing.is_image_space(self.observation_space, check_channels=False)
        self.is_discrete = isinstance(self.action_space, spaces.Discrete)

        if not self.rtg_condition:
            del self.embed_return
            del self.predict_return
        if self.use_fast_attn or self.config.add_cross_attention \
            or self.config.activation_function in ["geglu", "swiglu"] \
            or hasattr(self.config, "rope") or hasattr(self.config, "rms_norm"):  
            del self.encoder 
            self.encoder = CustomDTGPT2Model(config)
        if self.reward_condition:
            self.embed_rewards = torch.nn.Linear(1, config.hidden_size)
            self.predict_reward = torch.nn.Linear(config.hidden_size, 1)
        if self.both_time_embds: 
            self.embed_timestep_rel = nn.Embedding(config.max_length, config.hidden_size)
        elif self.relative_pos_embds:
            del self.embed_timestep
            self.embed_timestep = nn.Embedding(config.max_length, config.hidden_size)
        if self.is_image_space:
            self.embed_image = make_image_encoder(observation_space=observation_space, 
                                                  features_dim=config.hidden_size, encoder_kwargs=self.encoder_kwargs)
            if self.img_is_encoded: 
                # freeze embed_image
                for param in self.embed_image.parameters(): param.requires_grad = False
                self.embed_image.eval()
        if self.is_discrete:
            self.embed_action = nn.Sequential(nn.Embedding(self.action_space.n, config.hidden_size), nn.Tanh())
        if self.separate_ln: 
            self.embed_ln = nn.ModuleDict({"s": nn.LayerNorm(config.hidden_size),
                                           "rtg": nn.LayerNorm(config.hidden_size),
                                           "a": nn.LayerNorm(config.hidden_size),
                                           "r": nn.LayerNorm(config.hidden_size)})
        if hasattr(config, "embed_rms_norm") and config.embed_rms_norm:
            self.embed_ln = LlamaRMSNorm(config.hidden_size)
        self.setup_policy()
        self.post_init()
        self.tok_to_pos = {"s": 0, "rtg": 1,  "a": 2}
        self.tok_to_pred_pos = {"s": 0, "rtg": 2,  "a": 1}

    def get_optim_groups(self, weight_decay):
        """
        From: https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136

        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = self.get_whitelist_mods()
        blacklist_weight_modules = self.get_blacklist_mods()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)

        # add instances of nn.Parameter
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if (isinstance(p, torch.nn.Parameter) or isinstance(m, torch.nn.Parameter)) \
                        and (fpn not in decay and fpn not in no_decay):
                    no_decay.add(fpn)
        
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optim_groups_names = [
            {"params": sorted(list(decay)), "weight_decay": weight_decay},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]
        print("Optim groups:\n", optim_groups_names)
        return optim_groups
    
    def get_blacklist_mods(self): 
        return (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.Parameter)

    def get_whitelist_mods(self): 
        return (torch.nn.Linear, torch.nn.Conv1d, Conv1D, torch.nn.Conv2d)

    def get_action_head_params(self):
        if self.stochastic_policy:
            return list(self.mu.parameters()) + list(self.log_std.parameters())
        return list(self.action_pred.parameters())

    def zero_grads_for_action_heads(self, task_id):
        """
        For some reason zeroing grads for action heads that have been used in prior tasks is required when using
        nn.ModuleList. Otherwise, the backward pass will modify previously used action heads.
        Args:
            task_id: Int.

        """
        for i in range(self.num_task_heads):
            if i != task_id:
                if self.stochastic_policy:
                    self.mu[i].zero_grad(set_to_none=True)
                    self.log_std[i].zero_grad(set_to_none=True)
                else:
                    self.action_pred[i].zero_grad(set_to_none=True)

    def setup_policy(self):
        act_dim = self.config.act_dim if not self.is_discrete else self.action_space.n
        if self.orig_act_dim is not None: 
            act_dim = self.orig_act_dim
        if self.max_act_dim is not None: 
            act_dim = self.max_act_dim
        action_pred_in_dim = self.config.hidden_size
        if self.stochastic_policy:
            if self.num_task_heads > 1:
                self.mu = nn.ModuleList([self.make_head(action_pred_in_dim, act_dim, self.n_layer_head)
                                         for _ in range(self.num_task_heads)])
                self.log_std = nn.ModuleList([self.make_head(action_pred_in_dim, act_dim, self.n_layer_head)
                                              for _ in range(self.num_task_heads)])
            else:
                self.mu = self.make_head(action_pred_in_dim, act_dim, self.n_layer_head)
                self.log_std = self.make_head(action_pred_in_dim, act_dim, self.n_layer_head)
            self.action_dist = SquashedDiagGaussianDistribution(act_dim) if self.config.action_tanh \
                else DiagGaussianDistribution(act_dim)
        else:
            if self.num_task_heads > 1:
                self.action_pred = nn.ModuleList([nn.Sequential(
                    *([self.make_head(action_pred_in_dim, act_dim, self.n_layer_head)] + ([nn.Tanh()] if self.config.action_tanh else []))
                ) for _ in range(self.num_task_heads)])
            else:
                self.action_pred = nn.Sequential(
                    *([self.make_head(action_pred_in_dim, act_dim, self.n_layer_head)] + ([nn.Tanh()] if self.config.action_tanh else []))
                )

    @staticmethod
    def make_head(in_dim, out_dim, n_layer=1):
        layers = []
        for _ in range(n_layer - 1):
            layers.append(nn.Linear(in_dim, in_dim))
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def predict_action(self, x, deterministic=False, task_id=None):
        if not self.stochastic_policy:
            if self.num_task_heads > 1:
                assert task_id is not None
                action = self.action_pred[task_id](x)
            else:
                action = self.action_pred(x)
            return action
        mean_actions, log_std, kwargs = self.get_action_dist_params(x, task_id)
        in_shape = mean_actions.shape

        mean_actions = mean_actions.reshape(-1, self.config.act_dim)
        log_std = log_std.reshape(-1, self.config.act_dim)
        # Note: the action is squashed
        action = self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic)
        return action.reshape(*in_shape)

    def get_action_dist_params(self, x_latent, task_id=None):
        if self.num_task_heads > 1:
            assert task_id is not None
            mean_actions = self.mu[task_id](x_latent)
            log_std = self.log_std[task_id](x_latent)
        else:
            mean_actions = self.mu(x_latent)
            log_std = self.log_std(x_latent)
        # Original Implementation to cap the standard deviation
        log_std = torch.clamp(log_std, -20, 2)
        return mean_actions, log_std, {}

    def action_log_prob(self, x_latent, task_id=None):
        mean_actions, log_std, kwargs = self.get_action_dist_params(x_latent, task_id)
        in_shape = mean_actions.shape
        mean_actions = mean_actions.reshape(-1, self.config.act_dim)
        log_std = log_std.reshape(-1, self.config.act_dim)
        # return action and associated log prob
        action, log_prob = self.action_dist.log_prob_from_params(mean_actions, log_std)
        action = action.reshape(*in_shape)
        log_prob = log_prob.reshape(-1, action.shape[1], 1)
        return action, log_prob

    def compute_log_prob_given_action(self, action):
        return self.action_dist.log_prob(action)

    def get_action_embeddings(self, action, attention_mask=None):
        if self.is_discrete:
            action = action.flatten(start_dim=1)
        emb = self.embed_action(action)
        return emb

    def get_state_embeddings(self, state, mod_vectors=None):
        if self.img_is_encoded and len(state.shape) == 3: 
            return state
        if len(state.shape) > 4:
            # is_image_space
            batch_size, seq_len = state.shape[0], state.shape[1]
            state = state.reshape(-1, *self.observation_space.shape)
            return self.embed_image(state, mod_vectors=mod_vectors).reshape(batch_size, seq_len, self.config.hidden_size)
        return self.embed_state(state)

    def get_return_embeddings(self, returns):
        return_embeddings = None
        if self.rtg_condition:
            if self.symlog_transform: 
                returns = symlog(returns)
            return_embeddings = self.embed_return(returns)
        return return_embeddings
    
    def get_reward_embeddings(self, rewards): 
        reward_embeddings = None
        if self.reward_condition:
            if self.symlog_transform: 
                rewards = symlog(rewards)
            reward_embeddings = self.embed_rewards(rewards)
        return reward_embeddings

    def get_time_embeddings(self, timesteps, attention_mask=None):
        if self.relative_pos_embds or self.both_time_embds: 
            batch_size, context_len = timesteps.shape[0], timesteps.shape[-1]
            pos = torch.arange(context_len, device=timesteps.device)
            pos = pos.repeat(batch_size).reshape(*timesteps.shape)
            if self.both_time_embds:
                return self.embed_timestep(timesteps) + self.embed_timestep_rel(pos) 
            return self.embed_timestep(pos)
        return self.embed_timestep(timesteps)

    def extract_task_id(self, states):
        return states[0, -1, -self.num_task_heads:].argmax()

    def forward(
        self,
        states=None,
        actions=None,
        rewards=None,
        returns_to_go=None,
        timesteps=None,
        attention_mask=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
        deterministic=True,
        with_log_probs=False,
        prompt=None,
        task_id=None,
        ddp_kwargs=None,
        context_trjs=None,
        inference_params=None, 
        past_key_values=None,
        use_inference_cache=False
    ):
        if ddp_kwargs is None:
            ddp_kwargs = {}
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.is_discrete and actions.shape[-1] == 1:
            # assumes discrete action spaces are always one-dimensional
            actions = actions.long()

        x, encoder_outputs, prompt_infos = self.compute_hidden_states(
            states=states, actions=actions, rewards=rewards, returns_to_go=returns_to_go, timesteps=timesteps,
            attention_mask=attention_mask, output_hidden_states=output_hidden_states,
            output_attentions=output_attentions, return_dict=return_dict, prompt=prompt,
            task_id=task_id, context_trjs=context_trjs, inference_params=inference_params,
            past_key_values=past_key_values, use_inference_cache=use_inference_cache
        )
        state_preds, action_preds, action_log_probs, return_preds, reward_preds, action_logits, entropy = self.get_predictions(
            x, with_log_probs=with_log_probs, deterministic=deterministic,
            task_id=task_id if self.num_task_heads != 1 else None,
            is_discrete=not actions.is_floating_point()            
        )

        if not return_dict:
            if with_log_probs:
                return (state_preds, action_preds, action_log_probs, return_preds)
            return (state_preds, action_preds, return_preds)
        
        # when using DDP all output tensors need to contribute to the loss computation
        # this is not the case for: predict_reward, predict_return, predict_state
        # remove from outputs in case of DDP and not using reward prediction, return prediction or state prediction loss
        return OnlineDecisionTransformerOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            state_preds=state_preds if ddp_kwargs.get("predict_state") else None,
            action_preds=action_preds,
            return_preds=return_preds if ddp_kwargs.get("predict_return") else None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            action_log_probs=action_log_probs,
            reward_preds=reward_preds if ddp_kwargs.get("predict_reward") else None,
            action_logits=action_logits,
            entropy=entropy,
            last_encoder_output=x,
            prompt_infos=prompt_infos,
            cross_attentions=encoder_outputs.get("cross_attentions", None),
            past_key_values=encoder_outputs.get("past_key_values", None)
        )

    def compute_hidden_states(
            self,
            states=None,
            actions=None,
            rewards=None,
            returns_to_go=None,
            timesteps=None,
            attention_mask=None,
            output_hidden_states=None,
            output_attentions=None,
            return_dict=None,
            prompt=None,
            task_id=None, 
            context_trjs=None,
            inference_params=None,
            past_key_values=None,
            use_inference_cache=False
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        batch_size, seq_length = actions.shape[0], actions.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        inputs, stacked_inputs, stacked_attention_mask = self.compute_inputs(
            states, actions, returns_to_go, rewards, timesteps, attention_mask, 
            use_inference_cache=use_inference_cache and past_key_values is not None
        )

        # make position ids
        if self.global_pos_embds:
            position_ids = torch.arange(stacked_attention_mask.shape[1],
                                        device=stacked_inputs.device, dtype=torch.long).unsqueeze(0)
        else: 
            position_ids = torch.zeros(stacked_attention_mask.shape, device=stacked_inputs.device, dtype=torch.long)
                
        # we feed in the input embeddings (not word indices as in NLP) to the model
        encoder_inputs = {
            "inputs_embeds": stacked_inputs,
            "attention_mask": stacked_attention_mask,
            "position_ids": position_ids,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
            # exploits default behaviour of DecisionTransformerGPT2Block to add cross attention on (latent) prompts
            "encoder_hidden_states": None,
            "encoder_attention_mask": None,
            "use_cache": use_inference_cache,
        }
        if inference_params is not None: 
            encoder_inputs["inference_params"] = inference_params
        if use_inference_cache: 
            encoder_inputs, seq_length = self.handle_inference_cache(encoder_inputs, seq_length, past_key_values)
        encoder_outputs = self.encoder(**encoder_inputs)
        if use_inference_cache is not None and self.inf_dummy_batch_size is not None: 
            encoder_outputs = self.postprocess_inference_cache(encoder_outputs)
        
        # grab last hidden state
        x = encoder_outputs['last_hidden_state']

        if not self.config.add_cross_attention:
            x = x[:, -seq_length * len(inputs):]
        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, len(inputs), self.hidden_size).permute(0, 2, 1, 3)
        # [batch_size, r_s_a, seq_len, hidden_size]
        return x, encoder_outputs, None
    
    def compute_inputs(self, states, actions, returns_to_go, rewards, timesteps, 
                       attention_mask, use_inference_cache=False): 
        batch_size, seq_length = actions.shape[0], actions.shape[1]
        if use_inference_cache: 
            # only embed last step
            states, actions, returns_to_go, rewards, timesteps = states[:, -1].unsqueeze(1), actions[:, -1].unsqueeze(1),\
                returns_to_go[:, -1].unsqueeze(1), rewards[:, -1].unsqueeze(1), timesteps[:, -1].unsqueeze(1)
            seq_length = 1
            
        # embed each modality with a different head
        state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings = self.embed_inputs(
            states, actions, returns_to_go, rewards, 
            attention_mask[:, -1].unsqueeze(1) if use_inference_cache else attention_mask
        )
        if use_inference_cache and self.use_time_embds: 
            # pad with zeros from front to match seq_length
            pad = seq_length - 1
            state_embeddings = torch.cat([torch.zeros(batch_size, pad, *state_embeddings.shape[2:],
                                                      device=state_embeddings.device), state_embeddings], dim=1)
            if action_embeddings is not None: 
                action_embeddings = torch.cat([torch.zeros(batch_size, pad, *action_embeddings.shape[2:],
                                                        device=action_embeddings.device), action_embeddings], dim=1)
            if returns_embeddings is not None:
                returns_embeddings = torch.cat([torch.zeros(batch_size, pad, *returns_embeddings.shape[2:],
                                                            device=returns_embeddings.device), returns_embeddings], dim=1)
            if rewards_embeddings is not None:
                rewards_embeddings = torch.cat([torch.zeros(batch_size, pad, *rewards_embeddings.shape[2:],
                                                            device=rewards_embeddings.device), rewards_embeddings], dim=1)
                                                       
        if self.use_time_embds:
            if use_inference_cache: 
                timesteps = torch.cat([torch.zeros(batch_size, pad, device=timesteps.device, 
                                                   dtype=timesteps.dtype), timesteps], dim=1)
            time_embeddings = self.get_time_embeddings(timesteps, attention_mask=attention_mask)
            state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings = self.add_pos_embeddings(
                time_embeddings, state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings
            )
        else:
            time_embeddings = None
            
        # prepare inputs + masks
        inputs, masks = self.construct_inputs_and_masks(
            state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings,
            attention_mask, time_embeddings=time_embeddings,  
            reward_condition=self.reward_condition, rtg_condition=self.rtg_condition,
            action_condition=self.action_condition
        )
        
        stacked_inputs, stacked_attention_mask = self.prepare_inputs_and_masks(inputs, masks, batch_size, seq_length)
        
        # mask individual tokens
        if self.training and self.p_mask > 0: 
            mask = torch.bernoulli(torch.full_like(stacked_attention_mask, self.p_mask))
            stacked_attention_mask = stacked_attention_mask * (1 - mask)
        if self.training and self.p_token_drop > 0: 
            mask = torch.bernoulli(torch.full_like(stacked_attention_mask, self.p_token_drop)).bool()
            stacked_inputs = stacked_inputs.masked_fill(mask.unsqueeze(-1), 0)
        return inputs, stacked_inputs, stacked_attention_mask

    def embed_inputs(self, states, actions, returns_to_go, rewards, attention_mask):
        if len(states.shape) > 4:
            # is_image_space
            states = states.float() / 255.0
        state_embeddings = self.get_state_embeddings(states)
        action_embeddings = self.get_action_embeddings(actions, attention_mask=attention_mask)
        return_embeddings = self.get_return_embeddings(returns_to_go)
        reward_embeddings = self.get_reward_embeddings(rewards)
        return state_embeddings, action_embeddings, return_embeddings, reward_embeddings

    def add_pos_embeddings(self, time_embeddings, state_embeddings, action_embeddings,
                           return_embeddings, reward_embeddings):
        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        if action_embeddings is not None: 
            action_embeddings = action_embeddings + time_embeddings
        if return_embeddings is not None:
            return_embeddings = return_embeddings + time_embeddings
        if reward_embeddings is not None:
            reward_embeddings = reward_embeddings + time_embeddings
        return state_embeddings, action_embeddings, return_embeddings, reward_embeddings
    
    def construct_inputs_and_masks(self, state_embeddings, action_embeddings, returns_embeddings, 
                                   rewards_embeddings, attention_mask, time_embeddings=None, 
                                   reward_condition=False, rtg_condition=True, action_condition=True):
        # this can be shortened
        if reward_condition:
            if not rtg_condition:
                if not action_condition:
                    inputs = (state_embeddings, rewards_embeddings)
                    self.tok_to_pred_pos = {"s": len(inputs) - 1, "a": 0, "r": 0}
                    self.tok_to_pos = {"s": 0, "r": 1}
                else: 
                    inputs = (state_embeddings, action_embeddings, rewards_embeddings)
                    self.tok_to_pred_pos = {"s": len(inputs) - 1, "a": 0, "r": 1}
                    self.tok_to_pos = {"s": 0, "a": 1, "r": 2}
            else:
                if not action_condition: 
                    inputs = (state_embeddings, returns_embeddings, rewards_embeddings)
                    self.tok_to_pred_pos = {"s": len(inputs) - 1, "rtg": 0, "a": 1, "r": 1}
                    self.tok_to_pos = {"s": 0, "rtg": 1, "r": 2}
                else: 
                    inputs = (state_embeddings, returns_embeddings, action_embeddings, rewards_embeddings)
                    self.tok_to_pred_pos = {"s": len(inputs) - 1, "rtg": 0, "a": 1, "r": 2}
                    self.tok_to_pos = {"s": 0, "rtg": 1, "a": 2, "r": 3}
        elif not rtg_condition:
            if not action_condition:
                inputs = (state_embeddings)
                self.tok_to_pred_pos = {"s": len(inputs) - 1, "a": 0}
                self.tok_to_pos = {"s": 0}
            else: 
                inputs = (state_embeddings, action_embeddings)
                self.tok_to_pred_pos = {"s": len(inputs) - 1, "a": 0}
                self.tok_to_pos = {"s": 0, "a": 1}
        else:
            if not action_condition: 
                inputs = (returns_embeddings, state_embeddings)
                self.tok_to_pred_pos = {"s": 0, "rtg": len(inputs) - 1, "a": 1}
                self.tok_to_pos = {"s": 1, "rtg": 0}
            else: 
                inputs = (returns_embeddings, state_embeddings, action_embeddings)
                self.tok_to_pred_pos = {"s": 0, "rtg": len(inputs) - 1, "a": 1}
                self.tok_to_pos = {"s": 1, "rtg": 0, "a": 2}
        masks = tuple([attention_mask] * len(inputs))
        return inputs, masks
    
    def prepare_inputs_and_masks(self, inputs, masks, batch_size, seq_length, ln=None):
        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        # shape: [batch_size, len(inputs) * context_len, hidden_size]
        ln = self.embed_ln if ln is None else ln
        if self.separate_ln: 
            # separate layernorms per token type
            inputs = self.apply_lns(inputs, ln)
        stacked_inputs = (
            torch.stack(inputs, dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batch_size, len(inputs) * seq_length, self.hidden_size)
        )
        if not self.separate_ln: 
            stacked_inputs = ln(stacked_inputs)

        # shape: [batch_size, len(inputs) * context_len, hidden_size]
        # to make the attention mask fit the stacked inputs, have to stack it as well
        # shape: [batch_size, len(inputs) * context_len]
        stacked_attention_mask = (
            torch.stack(masks, dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, len(masks) * masks[0].shape[-1])
        )
        return stacked_inputs, stacked_attention_mask

    def apply_lns(self, inputs, ln):
        new_inputs = [0] * len(inputs)
        for tok, pos in self.tok_to_pos.items():
            if not isinstance(pos, (list, range)):
                pos = [pos]
            for p in pos:
                new_inputs[p] = ln[tok](inputs[p])
        return new_inputs

    def get_predictions(self, x, with_log_probs=False, deterministic=False, task_id=None, is_discrete=False):
        action_log_probs, reward_preds, action_logits, entropy, return_preds = None, None, None, None, None
        state_preds = self.predict_state(x[:, self.tok_to_pred_pos["s"]])
        if self.rtg_condition:
            return_preds = self.predict_return(x[:, self.tok_to_pred_pos["rtg"]])
        if self.reward_condition:
            reward_preds = self.predict_reward(x[:, self.tok_to_pred_pos["r"]])
        if with_log_probs:
            action_preds, action_log_probs = self.action_log_prob(x[:,  self.tok_to_pred_pos["a"]], task_id=task_id)
        else:
            action_preds = self.predict_action(x[:,  self.tok_to_pred_pos["a"]],
                                               deterministic=deterministic, task_id=task_id)
        return state_preds, action_preds, action_log_probs, return_preds, reward_preds, action_logits, entropy

    def scale_action(self, action):
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def freeze(self, **kwargs):
        frozen, not_frozen = [], []
        exclude_keys = self._extract_exclude_keys(**kwargs)
        for name, param in self.named_parameters():
            if name in exclude_keys:
                not_frozen.append(name)
                continue
            param.requires_grad = False
            frozen.append(name)
        return frozen, not_frozen
    
    def _extract_exclude_keys(self, exclude_action_head=False, exclude_crossattn=False,
                              exclude_img_encoder=False, exclude_embeds=False,
                              exclude_lns=False, img_encoder_only=False, exclude_layers=None, exclude_mods=None):
        exclude_keys = set()
        if exclude_mods is not None: 
            if not isinstance(exclude_mods, list): 
                exclude_mods = [exclude_mods]
            for mod in exclude_mods:
                exclude_keys.update([name for name, _ in self.named_parameters() if mod in name])
        if exclude_action_head:
            if self.stochastic_policy:
                exclude_keys.update([f"mu.{name}" for name, _ in self.mu.named_parameters()])
                exclude_keys.update([f"log_std.{name}" for name, _ in self.log_std.named_parameters()])
            else:
                if hasattr(self, "action_pred"):
                    exclude_keys.update([f"action_pred.{name}" for name, _ in self.action_pred.named_parameters()])
                if hasattr(self, "action_net"):
                    exclude_keys.update([f"action_net.{name}" for name, _ in self.action_net.named_parameters()])
        if exclude_crossattn:
            exclude_keys.update([name for name, _ in self.named_parameters()
                                 if "crossattention" in name or "ln_cross_attn" in name or "crossattn" in name])
        if exclude_img_encoder: 
            exclude_keys.update([name for name, _ in self.named_parameters() if "embed_image" in name])
        if img_encoder_only: 
            exclude_keys.update([name for name, _ in self.named_parameters() if "embed_image" not in name])
        if exclude_embeds: 
            exclude_keys.update(
                [name for name, _ in self.named_parameters() 
                 if any(sub in name for sub in
                        ["embed_image", "embed_state", "embed_action", "embed_return", "embed_rewards", 
                         "embed_timestep", "encoder.wte", "encoder.wpe"])]
            )
        if exclude_lns: 
            exclude_keys.update(
                [name for name, _ in self.named_parameters() 
                 if any(sub in name for sub in ["embed_ln", ".ln_"])]
            )
        if exclude_layers is not None:
            assert isinstance(exclude_layers, list) and all([l < 0 for l in exclude_layers])
            layer_idx = list(range(self.config.n_layer))
            exclude_layer_prefix = [f"encoder.h.{str(layer_idx[l])}" for l in exclude_layers]
            exclude_keys.update([name for name, _ in self.named_parameters()
                                 if any([prefix in name for prefix in exclude_layer_prefix])])
        return exclude_keys

    def load_action_head_weights(self, model_dict):
        if self.stochastic_policy:
            for i in range(len(self.mu)):
                with torch.no_grad():
                    self.mu[i].weight.copy_(model_dict["mu.weight"])
                    self.mu[i].bias.copy_(model_dict["mu.bias"])
                    self.log_std[i].weight.copy_(model_dict["log_std.weight"])
                    self.log_std[i].bias.copy_(model_dict["log_std.bias"])
        else:
            for i in range(len(self.action_pred)):
                with torch.no_grad():
                    self.action_pred[i].weight.copy_(model_dict["action_pred.weight"])
                    self.action_pred[i].bias.copy_(model_dict["action_pred.bias"])

    def handle_inference_cache(self, encoder_inputs, seq_length, past_key_values=None): 
        if past_key_values is not None: 
            # keep only tokens of last step, as rest is cached in past_key_values
            num_tokens = max([pos for tokpos in self.tok_to_pos.values() 
                              for pos in ([tokpos] if isinstance(tokpos, int) else list(tokpos))]) + 1
            encoder_inputs["inputs_embeds"] = encoder_inputs["inputs_embeds"][:, -num_tokens:]
            encoder_inputs["position_ids"] = encoder_inputs["position_ids"][:, -num_tokens:]
            if isinstance(past_key_values, dict): 
                # recurrent mode
                encoder_inputs["past_key_values"] = past_key_values
            else: 
                # remove very first step from past_key_values
                # contains: contains n_layer tuples, each tuple has 2 tensors of shape (bs, heads, seq_len, head_dim)
                encoder_inputs["past_key_values"] = [tuple([past[0][:, :, num_tokens:], past[1][:, :, num_tokens:]]) 
                                                    for past in past_key_values]
            seq_length = 1
        if self.inf_dummy_batch_size is not None: 
            encoder_inputs = self.preprocess_inference_cache(encoder_inputs)
        return encoder_inputs, seq_length 
    
    def preprocess_inference_cache(self, encoder_inputs):
        # repeat batch size for inf time measurements
        encoder_inputs["inputs_embeds"] = encoder_inputs["inputs_embeds"].repeat(self.inf_dummy_batch_size, 1, 1)
        encoder_inputs["position_ids"] = encoder_inputs["position_ids"].repeat(self.inf_dummy_batch_size, 1)
        encoder_inputs["attention_mask"] = encoder_inputs["attention_mask"].repeat(self.inf_dummy_batch_size, 1)
        return encoder_inputs
    
    def postprocess_inference_cache(self, encoder_outputs):
        # only preserve first batch
        encoder_outputs["last_hidden_state"] = encoder_outputs["last_hidden_state"][:1]
        return encoder_outputs
