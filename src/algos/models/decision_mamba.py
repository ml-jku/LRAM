import math
import torch
import torch.nn as nn
from functools import partial
from typing import Optional, Tuple, Union
from transformers import PreTrainedModel, DecisionTransformerConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
from mamba_ssm.models.mixer_seq_simple import create_block, _init_weights
from .online_decision_transformer_model import OnlineDecisionTransformerModel
from .discrete_decision_transformer_model import DiscreteDTModel
from .multi_domain_discrete_dt_model import MultiDomainDiscreteDTModel


class MambaConfig(DecisionTransformerConfig):
    model_type = "mamba"
    def __init__(
        self,
        n_layer=64,
        d_state=16,
        d_model=2560,
        d_conv=4,
        expand=2,
        norm_epsilon=1e-5,
        conv_bias=True,
        bias=False,
        rms_norm=True,
        fused_add_norm=True,
        residual_in_fp32=True,
        dt_rank="auto",
        **kwargs,
    ):
        super().__init__(n_layer=n_layer, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, **kwargs)
        self.conv_bias = conv_bias
        self.expand = expand
        self.d_conv = d_conv
        self.d_model = d_model
        self.d_state = d_state
        self.rms_norm = rms_norm
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.norm_epsilon = norm_epsilon
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = dt_rank
        self.bias = bias
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        # set kwargs for compatibility with DTConfig
        self.hidden_size = self.d_model
        

class MambaEncoder(PreTrainedModel):
    def __init__(
        self,
        config, 
        ssm_cfg=None,
        initializer_cfg=None,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        ssm_cfg = {"d_state": config.d_state, "d_conv": config.d_conv, "expand": config.expand,
                   "dt_rank": config.dt_rank, "bias": config.bias, "conv_bias": config.conv_bias} if ssm_cfg is None \
            else ssm_cfg
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        if config.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        dtype = None
        if hasattr(config, "dtype"):
            dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
            dtype = dtypes[config.dtype]
        
        self.layers = nn.ModuleList(
            [
                create_block(
                    config.d_model,
                    d_intermediate=config.d_intermediate,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                    dtype=dtype
                )
                for i in range(config.n_layer)
            ]
        )
        self.norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(config.d_model, eps=config.norm_epsilon)
        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

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
        inference_params=None
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # Obtains already embedded inputs, rest of arguments are redundant for Mamba
        hidden_states = inputs_embeds
        residual = None
        for layer in self.layers:
            if inference_params is not None and inference_params.seqlen_offset > 1: 
                # loop over sequence dimension for recurrent inference
                all_hidden_states, all_residuals = [], []
                # cast inference params to right dtype, they may get initialized in fp32 
                for i in range(hidden_states.shape[1]):
                    # residual is None in the first layer
                    hs, resid = layer(hidden_states[:, i].unsqueeze(1),
                                      residual if residual is None else residual[:, i].unsqueeze(1), 
                                      inference_params=inference_params)
                    all_hidden_states.append(hs)
                    all_residuals.append(resid)
                # concat collected hidden states/residuals for next layer
                hidden_states = torch.cat(all_hidden_states, dim=1)
                residual = torch.cat(all_residuals, dim=1)
            else: 
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
            if inference_params is not None: 
                inference_params.seqlen_offset += hidden_states.shape[1]
        if not self.config.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.config.residual_in_fp32,
            )

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states)


class DecisionMambaModel(OnlineDecisionTransformerModel):

    def __init__(
        self,
        config,
        observation_space,
        action_space,
        stochastic_policy=False,
        ssm_cfg=None,
        initializer_cfg=None,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__(config, observation_space, action_space, stochastic_policy=stochastic_policy, **kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        # set-up mamba encoder instead
        del self.encoder
        self.encoder = MambaEncoder(
            config=config,
            ssm_cfg=ssm_cfg,
            initializer_cfg=initializer_cfg,
            **factory_kwargs,
        )
        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.encoder.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    

class DiscreteDecisionMambaModel(DecisionMambaModel, DiscreteDTModel):
    def __init__(
        self,
        config,
        observation_space,
        action_space,
        **kwargs
    ):
        super().__init__(config, observation_space, action_space, **kwargs)


class MultiDomainDecisionMambaModel(DecisionMambaModel, MultiDomainDiscreteDTModel):
    def __init__(
        self,
        config,
        observation_space,
        action_space,
        **kwargs
    ):
        super().__init__(config, observation_space, action_space, **kwargs)
