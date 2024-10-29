import torch
from .universal_decision_transformer_sb3 import UDT
from .discrete_decision_transformer_sb3 import DiscreteDecisionTransformerSb3


class DecisionXLSTM(UDT):

    def __init__(self, policy, env, **kwargs):
        super().__init__(policy, env, **kwargs)

    def pad_inputs(self, states, actions, returns_to_go, timesteps, context_len=5, rewards=None):
        if self.use_inference_cache:
            # no need to pad inputs 
            context_len = 1
            attention_mask = torch.ones(actions.shape[1], device=self.device, dtype=torch.long).reshape(1, -1) 
            if self.replay_buffer.max_state_dim is not None and len(states.shape) == 3 and not self.s_proj_raw:
                # pad state input to max_state_dim, in case of continous state
                s_pad = self.replay_buffer.max_state_dim - states.shape[-1]
                states = torch.cat([states, torch.zeros((*states.shape[:-1], s_pad), device=self.device)], dim=-1)   
            if self.replay_buffer.max_act_dim is not None and actions.is_floating_point(): 
                # check if observations are images --> discrete action
                if len(states.shape) != 5: 
                    a_pad = self.replay_buffer.max_act_dim - actions.shape[-1] 
                    actions = torch.cat([actions, torch.zeros((*actions.shape[:-1], a_pad), device=self.device)], dim=-1)
            return states.float(), actions, returns_to_go.float(), timesteps, attention_mask, rewards
        else: 
            return super().pad_inputs(states, actions, returns_to_go, timesteps, 
                                      context_len=context_len, rewards=rewards)



class DiscreteDecisionXLSTM(DiscreteDecisionTransformerSb3, DecisionXLSTM):

    def __init__(self, policy, env, **kwargs):
        super().__init__(policy, env, **kwargs)
