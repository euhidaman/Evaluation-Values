"""
BitNet model implementation for transformers library compatibility
Based on Llama architecture with BitNet-specific modifications
"""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_bitnet import BitNetConfig

logger = logging.get_logger(__name__)

# Import Llama components and adapt them for BitNet
try:
    from transformers.models.llama.modeling_llama import (
        LlamaRMSNorm,
        LlamaRotaryEmbedding,
        LlamaAttention,
        LlamaMLP,
        LlamaDecoderLayer,
        LlamaPreTrainedModel,
        LlamaModel,
        LlamaForCausalLM,
        rotate_half,
        apply_rotary_pos_emb,
        repeat_kv
    )
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    logger.warning("Llama components not available, using basic implementation")

# Use Llama components with BitNet config if available
if LLAMA_AVAILABLE:
    class BitNetRMSNorm(LlamaRMSNorm):
        pass

    class BitNetRotaryEmbedding(LlamaRotaryEmbedding):
        pass

    class BitNetAttention(LlamaAttention):
        def __init__(self, config: BitNetConfig, layer_idx: Optional[int] = None):
            super().__init__(config, layer_idx)
            # Add BitNet-specific sub-normalization
            self.attn_sub_norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    class BitNetMLP(LlamaMLP):
        def __init__(self, config):
            super().__init__(config)
            # Add BitNet-specific sub-normalization with correct dimensions
            self.ffn_sub_norm = BitNetRMSNorm(config.intermediate_size, eps=config.rms_norm_eps)

    class BitNetDecoderLayer(LlamaDecoderLayer):
        def __init__(self, config: BitNetConfig, layer_idx: int):
            super().__init__(config, layer_idx)
            # Replace with BitNet-specific components
            self.self_attn = BitNetAttention(config=config, layer_idx=layer_idx)
            self.mlp = BitNetMLP(config)

    class BitNetPreTrainedModel(LlamaPreTrainedModel):
        config_class = BitNetConfig
        base_model_prefix = "model"
        supports_gradient_checkpointing = True
        _no_split_modules = ["BitNetDecoderLayer"]
        _skip_keys_device_placement = ["past_key_values"]
        _supports_flash_attn_2 = True
        _supports_sdpa = True
        _supports_cache_class = True

    class BitNetModel(LlamaModel):
        def __init__(self, config: BitNetConfig):
            super().__init__(config)
            # Override with BitNet-specific layers
            self.layers = nn.ModuleList(
                [BitNetDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )
            self.norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_init()

    class BitNetForCausalLM(LlamaForCausalLM):
        _tied_weights_keys = ["lm_head.weight"]

        def __init__(self, config):
            super().__init__(config)
            self.model = BitNetModel(config)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()

else:
    # Fallback implementation if Llama is not available
    class BitNetRMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)

    class BitNetLinear(nn.Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)

        def forward(self, input):
            return F.linear(input, self.weight, self.bias)

    class BitNetRotaryEmbedding(nn.Module):
        def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
            super().__init__()
            self.dim = dim
            self.max_position_embeddings = max_position_embeddings
            self.base = base
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        @torch.no_grad()
        def forward(self, x, position_ids):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            device_type = x.device.type
            device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    # Basic fallback implementation
    class BitNetPreTrainedModel(PreTrainedModel):
        config_class = BitNetConfig
        base_model_prefix = "model"
        supports_gradient_checkpointing = True
        _no_split_modules = ["BitNetDecoderLayer"]
        _skip_keys_device_placement = ["past_key_values"]

    class BitNetForCausalLM(BitNetPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            # This is a minimal implementation that should work with the actual model weights
            print("Warning: Using minimal BitNet implementation. Some features may not work.")
            self.config = config
