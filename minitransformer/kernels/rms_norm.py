import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    N_COLS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS

    x_ptrs = X_ptr + row_idx * stride_x_row + cols
    y_ptrs = Y_ptr + row_idx * stride_y_row + cols
    w_ptrs = W_ptr + cols

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptrs, mask=mask, other=0.0)

    variance = tl.sum(x * x, axis=0) / N_COLS
    rsqrt = tl.rsqrt(variance + EPS)
    out = (x * rsqrt).to(w.dtype) * w
    tl.store(y_ptrs, out, mask=mask)


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)

    rms_norm_kernel[(M,)](
        x,
        out,
        weight,
        x.stride(0),
        out.stride(0),
        N_COLS=N,
        EPS=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
