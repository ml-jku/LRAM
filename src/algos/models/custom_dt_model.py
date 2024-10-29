import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers.models.decision_transformer.modeling_decision_transformer import (
    DecisionTransformerGPT2Block,
    DecisionTransformerGPT2Model,
    DecisionTransformerGPT2Attention,
    DecisionTransformerGPT2MLP,
    Conv1D,
    BaseModelOutputWithPastAndCrossAttentions
)
from transformers.activations import ACT2FN
from einops import einsum
from .model_utils import SwiGLU, GEGLU
from .rope import LlamaRotaryEmbedding, apply_rotary_pos_emb
from .rms_norm import LlamaRMSNorm

ACT2FN["leaky_relu"] = nn.LeakyReLU
ACT2FN["swiglu"] = SwiGLU
ACT2FN["geglu"] = GEGLU


class CustomDTGPT2Attention(DecisionTransformerGPT2Attention):
    def __init__(self, config, **kwargs):
        """
        Adds functionality for fast self-attention implementations as provided by Pytorch v2.
        Args:
            config: Huggingface config.
        """
        super().__init__(config, **kwargs)
        self.use_fast_attn = config.use_fast_attn if hasattr(config, "use_fast_attn") else False
        self.attn_pdrop = config.attn_pdrop
        self.rope = config.rope if hasattr(config, "rope") else None
        if self.rope is not None:
            # head dim
            self.rope = LlamaRotaryEmbedding(config.hidden_size // config.n_head)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        if self.use_fast_attn:
            # flash attention in Pytorch v2
            # flashattention does not accept attention_mask --> use causal option for now
            # is_causal only supported if not k-v caching. 
            is_causal = not self.is_cross_attention if query.shape[2] == key.shape[2] else False
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    # attn_mask=attention_mask,
                    is_causal=is_causal,
                    dropout_p=self.attn_pdrop,
                )
            # does not provide functionality to return attention weights
            return attn_output, None
        return super()._attn(query, key, value, attention_mask=attention_mask, head_mask=head_mask)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        position_ids=None
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        # copied from modeling_decision_transformer.py to handle rotary positional embeddings
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `DecisionTransformerGPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        
        if self.rope is not None: 
            query, key = self.rotate(query, key, value, position_ids, is_cross_attn=encoder_hidden_states is not None)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
    
    def rotate(self, q, k, v, position_ids, is_cross_attn=False): 
        # apply rope to q and ky
        if position_ids is None:
            position_ids = torch.arange(v.shape[-2], device=v.device).unsqueeze(0)
        cos, sin = self.rope(v, position_ids)
        if is_cross_attn:
            # is cross attention - only add pos to key
            _, k = apply_rotary_pos_emb(k, k, cos, sin)
        else: 
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k


class CustomDecisionTransformerGPT2MLP(DecisionTransformerGPT2MLP):
    def __init__(self, intermediate_size, config):
        super().__init__(intermediate_size, config)
        # add support for leaky relu
        del self.act 
        self.act = ACT2FN[config.activation_function]
        if config.activation_function in ["swiglu", "geglu"]:
            # swiglu splits the intermediates size of c_fc in two parts
            # self.c_fc = Conv1D(intermediate_size * 2, config.hidden_size)
            self.c_proj = Conv1D(config.hidden_size, intermediate_size // 2)
    
    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class CustomDecisionTransformerGPT2Block(DecisionTransformerGPT2Block):
    def __init__(self, config, layer_idx):
        """
        Adds functionality for fast self-attention implementations as provided by Pytorch v2.
        Also adds functionality for adding cross attention layers only for specific layers.

        Args:
            config: Huggingface config.
            layer_idx: Int.
        """
        super().__init__(config, layer_idx=layer_idx)
        self.config = config
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        del self.attn
        self.attn = CustomDTGPT2Attention(config, layer_idx=layer_idx)
        if hasattr(config, "rms_norm") and config.rms_norm:
            self.ln_1 = LlamaRMSNorm(config.hidden_size)
            self.ln_2 = LlamaRMSNorm(config.hidden_size)
        # make cross attention dependent on layer idx
        self.is_cross_attention = False
        if config.add_cross_attention:
            del self.crossattention 
            self.crossattention = CustomDTGPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.is_cross_attention = True
            if hasattr(config, "crossattn_layers") and layer_idx not in config.crossattn_layers:
                del self.crossattention
                del self.ln_cross_attn
                self.is_cross_attention = False
            if self.is_cross_attention and hasattr(config, "rms_norm") and config.rms_norm:
                del self.ln_cross_attn
                self.ln_cross_attn = LlamaRMSNorm(config.hidden_size)   
            
        if hasattr(config, "activation_function") and config.activation_function in ["leaky_relu", "swiglu", "geglu"]:
            del self.mlp 
            self.mlp = CustomDecisionTransformerGPT2MLP(inner_dim, config)
            if self.is_cross_attention:
                self.crossattention = CustomDTGPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        position_ids=None
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            # position ids for rope
            position_ids=position_ids
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual
        
        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if self.is_cross_attention:
                if not hasattr(self, "crossattention"):
                    raise ValueError(
                        f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                        "cross-attention layers by setting `config.add_cross_attention=True`"
                    )
                residual = hidden_states
                hidden_states = self.ln_cross_attn(hidden_states)
                cross_attn_outputs = self.crossattention(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                    # does not get position_ids for rope
                )
                attn_output = cross_attn_outputs[0]
                if hasattr(self.config, "skip_crossattn_resid") and self.config.skip_crossattn_resid:
                    # skip cross attn residual connection
                    hidden_states = attn_output
                else: 
                    # residual connection
                    hidden_states = residual + attn_output
                outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights
            else:
                outputs = outputs + (None,)
        elif self.is_cross_attention or self.config.add_cross_attention:
            outputs = outputs + (None,)

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class CustomDTGPT2Model(DecisionTransformerGPT2Model):
    def __init__(self, config):
        """
        Adds functionality for fast self-attention implementations as provided by Pytorch v2.

        Args:
            config: Huggingface config.
        """
        super().__init__(config)
        del self.h
        self.rope = config.rope if hasattr(config, "rope") else False
        self.h = nn.ModuleList(
            [CustomDecisionTransformerGPT2Block(config=config, layer_idx=i) 
            for i in range(config.num_hidden_layers)]
        )
        if hasattr(config, "rms_norm") and config.rms_norm:
            self.ln_f = LlamaRMSNorm(self.embed_dim)
        self.post_init()

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
        **kwargs,
    ):
        # Copied from modeling_decision_transformer.py to integrate rotary pos embeds
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        if self.rope: 
            # positions are added in Attention
            hidden_states = inputs_embeds
        else: 
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    # add position ids for rope
                    position_ids=position_ids,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
