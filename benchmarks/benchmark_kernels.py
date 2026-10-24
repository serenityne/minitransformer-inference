import math
import torch
import torch.nn.functional as F
import triton
import triton.testing
from minitransformer.kernels import triton_attention


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['SEQ_LEN'],
        x_vals=[512, 1024, 2048, 4096],
        line_arg='provider',
        line_vals=['triton', 'torch_sdpa', 'torch_manual'],
        line_names=['Triton FlashAttention', 'PyTorch Native SDPA', 'Manual PyTorch'],
        styles=[('blue', '-'), ('green', '--'), ('red', ':')],
        ylabel='Execution Time (ms)',
        plot_name='attention-performance-comparison',
        args={'BATCH': 4, 'HEADS': 16, 'HEAD_DIM': 64},
    )
)
def benchmark_attention(BATCH, HEADS, SEQ_LEN, HEAD_DIM, provider):
    device = "cuda"
    dtype = torch.float16
    q = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch_sdpa':
        fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        return ms, min_ms, max_ms

    if provider == 'triton':
        fn = lambda: triton_attention(q, k, v)
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        return ms, min_ms, max_ms

    if provider == 'torch_manual':
        def manual_attn():
            scale = 1.0 / math.sqrt(HEAD_DIM)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask = torch.triu(torch.full((SEQ_LEN, SEQ_LEN), float('-inf'), device=device), diagonal=1)
            scores = scores + mask
            probs = F.softmax(scores, dim=-1)
            return torch.matmul(probs, v)

        if SEQ_LEN > 2048:
            return float('nan'), float('nan'), float('nan')
        ms, min_ms, max_ms = triton.testing.do_bench(manual_attn, quantiles=quantiles)
        return ms, min_ms, max_ms


if __name__ == "__main__":
    benchmark_attention.run(print_data=True, show_plots=False)
