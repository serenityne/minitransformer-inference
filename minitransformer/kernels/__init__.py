"""Optional Triton kernels. Install the `kernels` extra to use these."""

try:
    from .rms_norm import triton_rmsnorm
    from .attention import triton_attention
except ImportError as exc:  # pragma: no cover - depends on optional GPU stack
    if exc.name not in {"triton", "triton.language"}:
        raise
    triton_rmsnorm = None
    triton_attention = None

__all__ = ["triton_rmsnorm", "triton_attention"]
