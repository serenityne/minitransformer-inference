# MiniTransformer Inference Engine

A lightweight, educational Transformer inference engine built with PyTorch and Triton, exploring autoregressive generation, KV caching, and GPU kernel optimization.

**Status:** Work in Progress

## Overview

MiniTransformer is a small-scale implementation of a decoder-only Transformer designed to explore the engineering behind large language model inference.

The project focuses on understanding how Transformer components fit together, how autoregressive generation works, and how memory management and GPU kernels affect inference performance.

Rather than building a production serving system, this project prioritizes implementation clarity, correctness, experimentation, and performance analysis.

## Features

- **Decoder-only Transformer:** Modular implementation of Transformer blocks and model components.
- **Grouped-Query Attention (GQA):** Supports multiple query heads sharing key-value heads.
- **Rotary Positional Embeddings (RoPE):** Implements rotary positional encoding for attention.
- **RMSNorm:** Implements root mean square layer normalization.
- **Gated MLP:** Implements the feed-forward component of the Transformer.
- **Autoregressive Generation:** Supports token-by-token text generation.
- **KV Caching:** Reuses previously computed keys and values during generation.
- **Hugging Face Weight Loading:** Includes a loader for compatible Llama-style model weights.
- **Experimental Triton Kernels:** Includes custom implementations of attention and RMSNorm kernels.
- **Benchmarking:** Includes scripts for investigating inference latency, throughput, and memory usage.

## Architecture

A simplified view of the inference pipeline:

```text
Input Tokens
     |
     v
Token Embeddings
     |
     v
Transformer Blocks
     |
     +-- RMSNorm
     +-- Grouped-Query Attention
     |      +-- Rotary Positional Embeddings
     |      +-- KV Cache
     |
     +-- Residual Connection
     +-- Gated MLP
     +-- Residual Connection
     |
     v
Final Normalization
     |
     v
Language Model Head
     |
     v
Next-Token Logits
     |
     v
Autoregressive Generation
```

## Technology Stack

- Python
- PyTorch
- Triton
- pytest
- Hugging Face Transformers for compatible pretrained-model workflows

## Getting Started

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/minitransformer-inference.git
cd minitransformer-inference
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install the project and its declared dependencies:

```bash
pip install -e .
```

For development and testing, install any optional dependencies required by the test suite as documented in the project configuration.

## Running Tests

Run the test suite from the repository root:

```bash
pytest
```

GPU-specific tests require compatible CUDA, PyTorch, and Triton installations. Some pretrained-model tests may also require additional dependencies and model access.

## Benchmarking

The repository includes a benchmark suite for investigating inference performance.

Run the benchmark using the command documented by the benchmark script and inspect the available configuration options before selecting batch sizes and sequence lengths.

Relevant measurements include:

- Prompt prefill latency
- Time to first token
- Decode latency
- Tokens generated per second
- Peak GPU memory usage

Benchmark results depend on hardware, software versions, model configuration, sequence length, batch size, and numerical precision.

Performance comparisons are still being developed. No general performance advantage over established inference implementations is claimed at this stage.

## Current Limitations

This is an experimental project, not a production-ready LLM inference runtime.

- Pretrained-weight compatibility and numerical parity require further validation.
- The custom Triton kernels are experimental and are not yet integrated into the standard model execution path.
- KV-cache allocation and memory management can be optimized further.
- Benchmark methodology and performance comparisons are being refined.
- GPU support depends on the installed hardware and compatible software versions.

## Roadmap

- [ ] Strengthen correctness tests and pretrained-model parity checks.
- [ ] Improve inference benchmarking and reproducibility.
- [ ] Implement preallocated KV-cache storage.
- [ ] Integrate custom Triton kernels into model execution.
- [ ] Compare manual attention against PyTorch scaled dot-product attention.
- [ ] Investigate fused operations and quantized inference.
- [ ] Profile GPU execution and document performance bottlenecks.
- [ ] Add continuous integration and automated testing.

## Motivation

This project is an exploration of the intersection between deep learning and systems engineering.

The goal is to understand not only how Transformer models work, but also how inference performance is shaped by tensor operations, memory bandwidth, caching strategies, and GPU kernel execution.

## License

See the repository's `LICENSE` file for licensing information.
