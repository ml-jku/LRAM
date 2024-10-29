import torch
from typing import Optional, Tuple, Union
from omegaconf import OmegaConf
from transformers import PreTrainedModel, DecisionTransformerConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from dacite import from_dict
from dacite import Config as DaciteConfig
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.components.ln import MultiHeadLayerNorm, LayerNorm as xLSTMLayerNorm
from xlstm.components.linear_headwise import LinearHeadwiseExpand
from xlstm.blocks.slstm.cell import sLSTMCell_cuda
from xlstm.blocks.mlstm.cell import mLSTMCell
from .rms_norm import LlamaRMSNorm
from .online_decision_transformer_model import OnlineDecisionTransformerModel
from .discrete_decision_transformer_model import DiscreteDTModel
from .multi_domain_discrete_dt_model import MultiDomainDiscreteDTModel


def enable_biases_in_ln(module):
    for name, module in module.named_children():
        if isinstance(module, (xLSTMLayerNorm, MultiHeadLayerNorm)):
            print(f"Enabling biases in {name}")
            module.bias = torch.nn.Parameter(torch.zeros(module.ndim))
            module.reset_parameters()
        else:
            enable_biases_in_ln(module)
            

def setup_slstm_kernels(training, config):
    # add to globals to avoid recompilation
    config_key = str(config)
    if not "slstm_kernels" in globals():
        from xlstm.blocks.slstm.cell import sLSTMCellFuncGenerator 
        global slstm_kernels
        slstm_kernels = {config_key: sLSTMCellFuncGenerator(training, config)}
    elif config_key not in slstm_kernels:
        from xlstm.blocks.slstm.cell import sLSTMCellFuncGenerator
        slstm_kernels[config_key] = sLSTMCellFuncGenerator(training, config)


class CustomLSTMCell_cuda(sLSTMCell_cuda):
    
    def __init__(self, config, skip_backend_init: bool = True, device=None):
        """
        Custom LSTMCell for serialization compatibility.
        LSTMCell_cuda contains cuda kernels, which are not serializable.
        Serialization is required for multiprocessed evaluation, therefore, we manage kernels
        via global namespace. 
        
        Args:
            config: sLSTMCellConfig.
            skip_backend_init: Bool. Defaults to True to skip compilation.
            device: torch.Device. Defaults to None. For some reason, unpickling looses the correct device placement.
                Here we force it to be the same as the original device.
        """
        super().__init__(config, skip_backend_init=skip_backend_init)
        self._kernels_initialized = False
        self.config_key = str(config)
        self.device = device
        if device is not None:
            # necessary for serialization
            self.to(self.device)
            
    def __reduce__(self):
        return (self.__class__, (self.config, True, self.device))
    
    def _init_kernels(self):
        setup_slstm_kernels(self.training, self.config)
        self._kernels_initialized = True
    
    def _impl_step(
        self,
        training: bool,
        input: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if not self._kernels_initialized:
            self._init_kernels()
        return slstm_kernels[self.config_key].apply(
            training,
            input.contiguous(),
            state.contiguous(),
            self._recurrent_kernel.contiguous(),
            self._bias.contiguous(),
        )

    def _impl(
        self,
        training: bool,
        input: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if not self._kernels_initialized:
            self._init_kernels()
        return slstm_kernels[self.config_key].apply(
            training,
            input.contiguous(),
            state.contiguous(),
            self._recurrent_kernel.contiguous(),
            self._bias.contiguous(),
        )


class xLSTMConfig(DecisionTransformerConfig):
    model_type = "xlstm"
    def __init__(
        self,
        xlstm_config=None,
        ln_bias=False,
        chunkwise_step=False,
        **kwargs,
    ):
        super().__init__(resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, **kwargs)
        self.xlstm_config = xlstm_config
        self.ln_bias = ln_bias
        self.chunkwise_step = chunkwise_step


class xLSTMEncoder(PreTrainedModel):
    """
    xLSTMEcoder class for compatibility with Huggingface interface. 
    
    """
    def __init__(
        self,
        config, 
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        xlstm_config = config.xlstm_config if isinstance(config.xlstm_config, dict) \
            else OmegaConf.to_container(config.xlstm_config)
        self.xlstm_config = from_dict(data_class=xLSTMBlockStackConfig, data=xlstm_config, config=DaciteConfig(strict=True))
        self.layers = xLSTMBlockStack(self.xlstm_config)
        if config.ln_bias:
            enable_biases_in_ln(self.layers)
            print("Enabled biases in LayerNorms.")
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # Obtains already embedded inputs, rest of arguments are redundant for xLSTM
        if use_cache: 
            # call step multiple times --> single timestep contains multiple tokens (s,a,r,rtg)
            # concat hidden states, overwrite past_key_values to last one.
            if self.config.chunkwise_step:
                hidden_states, past_key_values = self.layers.step(inputs_embeds, past_key_values)
            else: 
                hidden_states = []
                for i in range(inputs_embeds.shape[1]):
                    hs, past_key_values = self.layers.step(inputs_embeds[:, i].unsqueeze(1), past_key_values)
                    hidden_states.append(hs)
                hidden_states = torch.cat(hidden_states, dim=1)
        else: 
            hidden_states = self.layers(inputs_embeds)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states,
                                                         past_key_values=past_key_values)

    def reset_parameters(self):
        self.layers.reset_parameters()
        
    
class DecisionXLSTMModel(OnlineDecisionTransformerModel):

    def __init__(
        self,
        config,
        observation_space,
        action_space,
        stochastic_policy=False,
        embed_bias_init="normal",
        **kwargs,
    ) -> None:
        super().__init__(config, observation_space, action_space, stochastic_policy=stochastic_policy, **kwargs)
        # set-up xlstm encoder instead
        del self.encoder
        self.encoder = xLSTMEncoder(config=config)
        if hasattr(config, "rms_norm") and config.rms_norm: 
            self.replace_norms(self.encoder, xLSTMLayerNorm, LlamaRMSNorm)
        self.post_init()
        if embed_bias_init == "no": 
            # turn off bias
            self.embed_ln.bias = None
            self.embed_state.bias = None
            if self.reward_condition: 
                self.embed_rewards.bias = None
            if self.rtg_condition:
                self.embed_return.bias = None
        elif embed_bias_init == "normal":
            # # normal bias init
            if hasattr(self.embed_ln, "bias"):
                self.embed_ln.bias.data.normal_(mean=0.0, std=self.config.initializer_range)
            self.embed_state.bias.data.normal_(mean=0.0, std=self.config.initializer_range)
            if self.reward_condition: 
                self.embed_rewards.bias.data.normal_(mean=0.0, std=self.config.initializer_range)
            if self.rtg_condition:
                self.embed_return.bias.data.normal_(mean=0.0, std=self.config.initializer_range)
        
    def post_init(self):
        super().post_init()
        if hasattr(self.encoder, "reset_parameters"): 
            self.encoder.reset_parameters()
            
    def get_blacklist_mods(self): 
        return (*super().get_blacklist_mods(), xLSTMLayerNorm, MultiHeadLayerNorm)

    def get_whitelist_mods(self): 
        return (*super().get_whitelist_mods(), LinearHeadwiseExpand)
    
    def handle_inference_cache(self, encoder_inputs, seq_length, past_key_values=None): 
        # keep only tokens of last step, as rest is cached in past_key_values
        # we can always do this for xlstm
        num_tokens = max([pos for tokpos in self.tok_to_pos.values() 
                          for pos in ([tokpos] if isinstance(tokpos, int) else list(tokpos))]) + 1
        encoder_inputs["inputs_embeds"] = encoder_inputs["inputs_embeds"][:, -num_tokens:]
        encoder_inputs["position_ids"] = encoder_inputs["position_ids"][:, -num_tokens:]
        seq_length = 1
        if past_key_values is not None: 
            encoder_inputs["past_key_values"] = past_key_values
        if self.inf_dummy_batch_size is not None: 
            encoder_inputs = self.preprocess_inference_cache(encoder_inputs)
        return encoder_inputs, seq_length 
        
    def replace_norms(self, model, old, new):
        for name, module in model.named_children():
            if type(module) is old:
                setattr(model, name, new(module.ndim, eps=module.eps))
            else:
                self.replace_norms(module, old, new)
                
    def make_pickleable(self, replace_cell=False):
        for block in self.encoder.layers.blocks:
            if hasattr(block, "xlstm"):
                if hasattr(block.xlstm, "slstm_cell"):
                    if replace_cell: 
                        # replace cuda slstm with vanilla slstm layers - cuda kernels are not serializable
                        # but serialization is required for multiprocessed evaluation
                        slstm_cell = block.xlstm.slstm_cell
                        device = next(slstm_cell.parameters()).device
                        state_dict = slstm_cell.state_dict()
                        new_cell = CustomLSTMCell_cuda(slstm_cell.config, device=device).to(device)
                        # convert to avoid shape mismatch
                        new_cell.load_state_dict(state_dict)
                        block.xlstm.slstm_cell = new_cell
                    else: 
                        # unset cuda kernels to avoid serialization issues
                        block.xlstm.slstm_cell.func = None
    
    def reinit_cuda_kernels(self, replace_cell=False):
        for block in self.encoder.layers.blocks: 
            if hasattr(block, "xlstm") and hasattr(block.xlstm, "slstm_cell"):
                if not replace_cell:
                    from xlstm.blocks.slstm.cell import sLSTMCellFuncGenerator 
                    slstm_cell = block.xlstm.slstm_cell
                    block.xlstm.slstm_cell.func = sLSTMCellFuncGenerator(slstm_cell.training, slstm_cell.config)


class DiscreteDecisionXLSTMModel(DecisionXLSTMModel, DiscreteDTModel):
    def __init__(
        self,
        config,
        observation_space,
        action_space,
        **kwargs
    ):
        super().__init__(config, observation_space, action_space, **kwargs)


class MultiDomainDiscreteDecisionXLSTMModel(DecisionXLSTMModel, MultiDomainDiscreteDTModel):
    def __init__(
        self,
        config,
        observation_space,
        action_space,
        **kwargs
    ):
        super().__init__(config, observation_space, action_space, **kwargs)
