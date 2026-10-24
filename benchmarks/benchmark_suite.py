import gc
import time
from dataclasses import dataclass
from typing import List

import torch
import matplotlib.pyplot as plt

from minitransformer import ModelConfig, TransformerLM, generate_cached, generate_uncached


@dataclass
class BenchmarkResult:
    engine: str
    batch_size: int
    prompt_len: int
    gen_len: int
    prefill_latency_ms: float
    ttft_ms: float
    decode_latency_ms: float
    throughput_tps: float
    peak_vram_mb: float


def flush_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def benchmark_run(
    engine_name: str,
    model: TransformerLM,
    batch_size: int,
    prompt_len: int,
    gen_len: int,
    warmup: int = 3,
    trials: int = 5,
    device: str = "cuda",
) -> BenchmarkResult:
    prompt_tokens = torch.randint(0, model.config.vocab_size, (batch_size, prompt_len), device=device)

    for _ in range(warmup):
        if engine_name == "Scratch-Uncached":
            _ = generate_uncached(model, prompt_tokens, max_new_tokens=4)
        else:
            _ = generate_cached(model, prompt_tokens, max_new_tokens=4)
    torch.cuda.synchronize()

    prefill_latencies = []
    ttft_latencies = []
    decode_latencies = []
    total_latencies = []

    for _ in range(trials):
        flush_gpu()
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        if engine_name == "Scratch-Uncached":
            curr_tokens = prompt_tokens.clone()
            t_prefill_end = None

            for step in range(gen_len):
                t_step_start = time.perf_counter()
                logits, _ = model(curr_tokens, use_cache=False)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr_tokens = torch.cat([curr_tokens, next_token], dim=-1)
                torch.cuda.synchronize()
                t_step_end = time.perf_counter()

                if step == 0:
                    t_prefill_end = t_step_end
                    ttft_latencies.append((t_prefill_end - t_start) * 1000)
                    prefill_latencies.append((t_prefill_end - t_start) * 1000)
                else:
                    decode_latencies.append((t_step_end - t_step_start) * 1000)

            t_end = time.perf_counter()

        else:
            curr_tokens = prompt_tokens.clone()
            logits, kv_cache = model(prompt_tokens, use_cache=True, start_pos=0)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr_tokens = torch.cat([curr_tokens, next_token], dim=-1)
            torch.cuda.synchronize()
            t_first_token = time.perf_counter()

            ttft_latencies.append((t_first_token - t_start) * 1000)
            prefill_latencies.append((t_first_token - t_start) * 1000)

            step_times = []
            current_pos = prompt_len
            for _ in range(gen_len - 1):
                t_dec_start = time.perf_counter()
                logits, kv_cache = model(
                    next_token, past_key_values=kv_cache, use_cache=True, start_pos=current_pos
                )
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                torch.cuda.synchronize()
                t_dec_end = time.perf_counter()
                step_times.append((t_dec_end - t_dec_start) * 1000)
                current_pos += 1

            decode_latencies.append(sum(step_times) / len(step_times) if step_times else 0.0)
            t_end = time.perf_counter()

        total_latencies.append(t_end - t_start)

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    mean_prefill = sum(prefill_latencies) / len(prefill_latencies)
    mean_ttft = sum(ttft_latencies) / len(ttft_latencies)
    mean_decode = sum(decode_latencies) / len(decode_latencies)
    mean_total_time = sum(total_latencies) / len(total_latencies)
    throughput = (batch_size * gen_len) / mean_total_time

    return BenchmarkResult(
        engine=engine_name,
        batch_size=batch_size,
        prompt_len=prompt_len,
        gen_len=gen_len,
        prefill_latency_ms=mean_prefill,
        ttft_ms=mean_ttft,
        decode_latency_ms=mean_decode,
        throughput_tps=throughput,
        peak_vram_mb=peak_vram,
    )


def plot_results(results: List[BenchmarkResult]):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    for engine in set(r.engine for r in results):
        subset = [r for r in results if r.engine == engine and r.batch_size == 1]
        if subset:
            ax1.plot([r.prompt_len for r in subset], [r.decode_latency_ms for r in subset], marker='o', label=engine)
    ax1.set_title("Decode Latency per Token (Batch Size = 1)")
    ax1.set_xlabel("Prompt Length")
    ax1.set_ylabel("Latency (ms / token)")
    ax1.legend()
    ax1.grid(True)

    ax2 = axes[0, 1]
    for engine in set(r.engine for r in results):
        subset = [r for r in results if r.engine == engine and r.prompt_len == 128]
        if subset:
            ax2.plot([r.batch_size for r in subset], [r.throughput_tps for r in subset], marker='s', label=engine)
    ax2.set_title("Throughput (Prompt Length = 128)")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Tokens / Second")
    ax2.legend()
    ax2.grid(True)

    ax3 = axes[1, 0]
    for engine in set(r.engine for r in results):
        subset = [r for r in results if r.engine == engine and r.batch_size == 4]
        if subset:
            ax3.plot([r.prompt_len for r in subset], [r.peak_vram_mb for r in subset], marker='^', label=engine)
    ax3.set_title("Peak VRAM Usage (Batch Size = 4)")
    ax3.set_xlabel("Prompt Length")
    ax3.set_ylabel("Memory (MB)")
    ax3.legend()
    ax3.grid(True)

    ax4 = axes[1, 1]
    for engine in set(r.engine for r in results):
        subset = [r for r in results if r.engine == engine and r.batch_size == 1]
        if subset:
            ax4.plot([r.prompt_len for r in subset], [r.ttft_ms for r in subset], marker='d', label=engine)
    ax4.set_title("Time To First Token (Batch Size = 1)")
    ax4.set_xlabel("Prompt Length")
    ax4.set_ylabel("TTFT (ms)")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=300)


def run_benchmark_matrix():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA required")

    config = ModelConfig(
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
    )

    raw_model = TransformerLM(config).to(device).eval()
    compiled_model = torch.compile(raw_model, mode="reduce-overhead")

    engines = {
        "Scratch-Uncached": raw_model,
        "Scratch-Cached": raw_model,
        "PyTorch-Compiled": compiled_model,
    }

    batch_sizes = [1, 4, 8]
    prompt_lengths = [128, 512, 1024]
    gen_len = 32

    results: List[BenchmarkResult] = []

    for prompt_len in prompt_lengths:
        for bs in batch_sizes:
            for engine_name, model_inst in engines.items():
                if engine_name == "Scratch-Uncached" and prompt_len > 512:
                    continue

                res = benchmark_run(
                    engine_name=engine_name,
                    model=model_inst,
                    batch_size=bs,
                    prompt_len=prompt_len,
                    gen_len=gen_len,
                    device=device,
                )
                results.append(res)

    plot_results(results)


if __name__ == "__main__":
    run_benchmark_matrix()
