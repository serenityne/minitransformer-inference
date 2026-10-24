from typing import Optional, Tuple, List
import torch
import torch.nn as nn

from .config import ModelConfig
from .rope import RotaryEmbedding
from .layers import RMSNorm, GatedMLP
from .attention import CausalAttention


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CausalAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GatedMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        attn_out, new_kv_cache = self.self_attn(
            self.input_layernorm(x), cos, sin, kv_cache=kv_cache, use_cache=use_cache
        )
        x = residual + attn_out

        residual = x
        x = residual + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv_cache


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        batch_size, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(seq_len, device=x.device, offset=start_pos)

        new_past_key_values = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            layer_cache = past_key_values[idx] if past_key_values is not None else None
            x, updated_cache = layer(x, cos=cos, sin=sin, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache:
                new_past_key_values.append(updated_cache)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_past_key_values
