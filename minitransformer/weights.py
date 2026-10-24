from .model import TransformerLM


def load_from_hf_llama(custom_model: TransformerLM, hf_model) -> None:
    """Copy compatible Llama-style weights; architecture settings must match."""
    import torch
    state_dict = hf_model.state_dict()
    custom_dict = custom_model.state_dict()

    mapping = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    for i in range(custom_model.config.num_hidden_layers):
        mapping[f"model.layers.{i}.input_layernorm.weight"] = f"layers.{i}.input_layernorm.weight"
        mapping[f"model.layers.{i}.self_attn.q_proj.weight"] = f"layers.{i}.self_attn.q_proj.weight"
        mapping[f"model.layers.{i}.self_attn.k_proj.weight"] = f"layers.{i}.self_attn.k_proj.weight"
        mapping[f"model.layers.{i}.self_attn.v_proj.weight"] = f"layers.{i}.self_attn.v_proj.weight"
        mapping[f"model.layers.{i}.self_attn.o_proj.weight"] = f"layers.{i}.self_attn.o_proj.weight"
        mapping[f"model.layers.{i}.post_attention_layernorm.weight"] = f"layers.{i}.post_attention_layernorm.weight"
        mapping[f"model.layers.{i}.mlp.gate_proj.weight"] = f"layers.{i}.mlp.gate_proj.weight"
        mapping[f"model.layers.{i}.mlp.up_proj.weight"] = f"layers.{i}.mlp.up_proj.weight"
        mapping[f"model.layers.{i}.mlp.down_proj.weight"] = f"layers.{i}.mlp.down_proj.weight"

    with torch.no_grad():
        for hf_name, scratch_name in mapping.items():
            if hf_name in state_dict and scratch_name in custom_dict:
                custom_dict[scratch_name].copy_(state_dict[hf_name])
            elif scratch_name == "lm_head.weight" and "lm_head.weight" not in state_dict:
                custom_dict["lm_head.weight"].copy_(state_dict["model.embed_tokens.weight"])
