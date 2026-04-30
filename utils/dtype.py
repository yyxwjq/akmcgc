"""Helpers for consistent floating-point dtype configuration."""
from __future__ import annotations

from typing import Optional

import torch


_FLOAT_DTYPE_ALIASES = {
    "float64": torch.float64,
    "double": torch.float64,
    "f64": torch.float64,
    "torch.float64": torch.float64,
    "torch.double": torch.float64,
    "float32": torch.float32,
    "float": torch.float32,
    "f32": torch.float32,
    "torch.float32": torch.float32,
    "torch.float": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "f16": torch.float16,
    "torch.float16": torch.float16,
    "torch.half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "torch.bfloat16": torch.bfloat16,
}


def resolve_float_dtype(
    dtype: Optional[torch.dtype | str],
    default: torch.dtype = torch.float64,
) -> torch.dtype:
    """Resolve a dtype config value to a floating-point ``torch.dtype``."""
    if dtype is None:
        return default
    if isinstance(dtype, torch.dtype):
        resolved = dtype
    elif isinstance(dtype, str):
        key = dtype.strip().lower()
        if key not in _FLOAT_DTYPE_ALIASES:
            raise ValueError(f"Unsupported floating dtype: {dtype!r}")
        resolved = _FLOAT_DTYPE_ALIASES[key]
    else:
        raise TypeError(f"dtype must be a torch.dtype, string, or None, got {type(dtype)!r}")

    if not torch.empty((), dtype=resolved).is_floating_point():
        raise ValueError(f"dtype must be floating point, got {resolved}")
    return resolved
