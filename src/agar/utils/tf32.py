from __future__ import annotations


def configure_tf32(enable: bool = True) -> None:
    """
    Configure TF32 behavior without mixing legacy/new PyTorch APIs.

    Prefer the "new" matmul precision API (`torch.set_float32_matmul_precision`) when available,
    and only fall back to legacy `allow_tf32` flags on older PyTorch builds.
    """

    import torch
    import os

    # Matmul: use only one API to avoid "mix of legacy and new APIs" RuntimeError.
    try:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if enable else "highest")
        elif hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = enable
    except Exception:
        pass

    # cuDNN conv TF32 is controlled separately; keep it opt-in to avoid mixing API styles.
    if os.environ.get("AGAR_TF32_CUDNN", "0").lower() in {"1", "true", "yes", "on"}:
        try:
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = enable
        except Exception:
            pass
