# MiniTransformer

A compact, educational decoder-only Transformer inference engine implemented in PyTorch, with grouped-query attention, RoPE, RMSNorm, a gated MLP, KV-cached decoding, optional experimental Triton kernels, tests, and benchmarks.

> **Status:** research/learning project, not a production inference server. The Triton kernels are standalone experiments and are **not wired into** the default model forward pass. Benchmark numbers must be measured on your own hardware; none are claimed here.

## Features

- Decoder-only language model with pre-norm residual blocks
- Rotary positional embeddings (RoPE)
- Grouped-query attention (GQA)
- RMSNorm and SwiGLU-style gated MLP
- Greedy or nucleus-sampling generation, with and without a KV cache
- Weight-copy helper for compatible Hugging Face Llama-style models
- Optional Triton RMSNorm and causal attention kernels
- Correctness tests and GPU benchmark scripts

## Install

Python 3.10+ is recommended. Install a PyTorch build appropriate for your operating system and CUDA setup first; see [PyTorch's installation selector](https://pytorch.org/get-started/locally/). Then, from the repository root:

```bash
python -m pip install -e .
```

For GPU kernels and kernel benchmarks:

```bash
python -m pip install -e '.[kernels,benchmarks]'
```

For Hugging Face weight-parity tests (which download a model):

```bash
python -m pip install -e '.[hf]'
```

## Quick start

```python
import torch
from minitransformer import ModelConfig, TransformerLM, generate_cached

config = ModelConfig(
    vocab_size=512,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
)
model = TransformerLM(config).eval()
prompt = torch.randint(0, config.vocab_size, (1, 8))
with torch.inference_mode():
    tokens = generate_cached(model, prompt, max_new_tokens=16)
print(tokens)
```

The model is randomly initialized in this example, so generated tokens are not meaningful language. To use a pretrained model, ensure the source model's architecture and configuration match this implementation before calling `load_from_hf_llama`. Tokenizer integration is intentionally left to the caller.

## Run tests

CPU-compatible tests:

```bash
python -m pytest -m 'not cuda and not network'
```

GPU kernel tests (requires CUDA and Triton):

```bash
python -m pytest -m cuda tests/test_kernels.py
```

The optional Hugging Face parity test is marked `network` because it downloads a checkpoint. Run it explicitly with `python -m pytest -m network tests/test_model.py`.

## Benchmarks

- `python benchmarks/benchmark_suite.py` compares uncached and cached decoding and a `torch.compile` variant. It requires CUDA and can take a while.
- `python benchmarks/benchmark_kernels.py` compares the Triton attention experiment with PyTorch SDPA and a manual baseline. It requires CUDA + Triton.

Benchmarks use randomly initialized models and synthetic tokens. Treat results as microbenchmarks, not a comparison of language-model quality or production serving systems.

## Important implementation notes

- The default config is intentionally modest in structure but still large enough to require substantial memory. Use a smaller config for CPU or limited VRAM.
- KV caching avoids recomputing prior key/value projections during autoregressive decoding, but this implementation concatenates cache tensors each step; a preallocated cache can be faster.
- `load_from_hf_llama` only copies matching named tensors. It does not automatically validate every architectural detail or convert arbitrary Hugging Face model variants.
- Triton kernel compatibility and performance depend on GPU, CUDA, Triton, and dtype. Validate numerical behavior on the target hardware.

## Layout

```text
minitransformer/
├── minitransformer/       # model implementation
│   └── kernels/           # optional Triton experiments
├── tests/                 # correctness tests
└── benchmarks/            # GPU benchmark scripts
```

## License

MIT; see `LICENSE`.
