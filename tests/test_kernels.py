import pytest
import torch
import torch.nn.functional as F
from minitransformer.kernels import triton_rmsnorm, triton_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.cuda
def test_triton_rmsnorm():
    device = "cuda"
    dtype = torch.float16
    M, N = 512, 1024
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.ones(N, device=device, dtype=dtype)

    ref_norm = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5).type_as(x) * weight
    tri_norm = triton_rmsnorm(x, weight, eps=1e-5)

    assert torch.allclose(ref_norm, tri_norm, atol=1e-3, rtol=1e-3)


@pytest.mark.cuda
def test_triton_attention():
    device = "cuda"
    dtype = torch.float16
    BATCH, HEADS, SEQ_LEN, HEAD_DIM = 2, 8, 1024, 64
    q = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype)

    ref_attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    tri_attn = triton_attention(q, k, v)

    assert torch.allclose(ref_attn, tri_attn, atol=2e-2, rtol=1e-2)


if __name__ == "__main__":
    test_triton_rmsnorm()
    test_triton_attention()
