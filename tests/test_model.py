import torch
from minitransformer import ModelConfig, TransformerLM, generate_cached, load_from_hf_llama


def test_causal_masking_and_shapes():
    config = ModelConfig(
        vocab_size=500, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    )
    model = TransformerLM(config).eval()

    dummy_input = torch.randint(0, 500, (2, 8))
    logits, _ = model(dummy_input)
    assert logits.shape == (2, 8, 500)

    seq_a = torch.randint(0, 500, (1, 10))
    seq_b = seq_a.clone()
    seq_b[0, 5:] = torch.randint(0, 500, (5,))

    logits_a, _ = model(seq_a)
    logits_b, _ = model(seq_b)
    diff = (logits_a[:, :5, :] - logits_b[:, :5, :]).abs().max().item()
    assert diff < 1e-6


def test_cached_vs_uncached_equivalence():
    config = ModelConfig(
        vocab_size=500, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    )
    model = TransformerLM(config).eval()
    prompt = torch.randint(0, 500, (1, 6))

    uncached_out, _ = model(prompt)
    cached_out, kv_cache = model(prompt, use_cache=True, start_pos=0)
    assert torch.allclose(uncached_out[:, -1, :], cached_out[:, -1, :], atol=1e-5)

    next_token = torch.argmax(cached_out[:, -1, :], dim=-1, keepdim=True)
    full_seq = torch.cat([prompt, next_token], dim=-1)

    target_logits, _ = model(full_seq)
    step_logits, _ = model(next_token, past_key_values=kv_cache, use_cache=True, start_pos=6)
    assert torch.allclose(target_logits[:, -1, :], step_logits[:, -1, :], atol=1e-4)


import pytest


@pytest.mark.network
def test_hf_equivalence():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return

    model_id = "HuggingFaceTB/SmolLM-135M"
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    hf_cfg = hf_model.config
    config = ModelConfig(
        vocab_size=hf_cfg.vocab_size,
        hidden_size=hf_cfg.hidden_size,
        intermediate_size=hf_cfg.intermediate_size,
        num_hidden_layers=hf_cfg.num_hidden_layers,
        num_attention_heads=hf_cfg.num_attention_heads,
        num_key_value_heads=hf_cfg.num_key_value_heads,
        rms_norm_eps=hf_cfg.rms_norm_eps,
        rope_theta=hf_cfg.rope_theta,
        max_position_embeddings=hf_cfg.max_position_embeddings,
    )
    scratch_model = TransformerLM(config).eval()
    load_from_hf_llama(scratch_model, hf_model)

    inputs = tokenizer("Testing transformer parity", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        hf_logits = hf_model(inputs).logits
        scratch_logits, _ = scratch_model(inputs)

    diff = (hf_logits - scratch_logits).abs().max().item()
    assert diff < 1e-3


if __name__ == "__main__":
    test_causal_masking_and_shapes()
    test_cached_vs_uncached_equivalence()
    test_hf_equivalence()


def test_generation_zero_tokens_and_shapes():
    config = ModelConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
    )
    model = TransformerLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (2, 5))
    assert torch.equal(generate_cached(model, prompt, max_new_tokens=0), prompt)
    assert torch.equal(__import__("minitransformer").generate_uncached(model, prompt, max_new_tokens=0), prompt)
    assert generate_cached(model, prompt, max_new_tokens=2).shape == (2, 7)

def test_config_rejects_invalid_head_layout():
    with pytest.raises(ValueError):
        ModelConfig(hidden_size=63, num_attention_heads=4)
