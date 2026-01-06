from __future__ import annotations


def configure_tf32(enable: bool = True) -> None:
    """
    Configure TF32 behavior using PyTorch 2.9+ "fp32_precision" APIs to avoid deprecation warnings.

    Falls back to older allow_tf32 flags when running on older PyTorch versions.
    """

    import torch

    mode = "tf32" if enable else "ieee"

    try:
        torch.backends.cuda.matmul.fp32_precision = mode  # type: ignore[attr-defined]
    except Exception:
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = enable

    try:
        torch.backends.cudnn.conv.fp32_precision = mode  # type: ignore[attr-defined]
    except Exception:
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = enable

