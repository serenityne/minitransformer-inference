from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("vocab_size, hidden_size, and intermediate_size must be positive")
        if self.num_hidden_layers <= 0 or self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
            raise ValueError("layer and attention head counts must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0 or self.max_position_embeddings <= 0:
            raise ValueError("normalization epsilon, RoPE theta, and max positions must be positive")
