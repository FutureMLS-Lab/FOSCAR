# Copyright 2026 Donglin Zhuang
# SPDX-License-Identifier: Apache-2.0
"""Device detection and the capability descriptor.

This module imports nothing from the CuTe DSL, so importing ``fosca``
works on a machine that has no CUDA toolkit installed.
"""
import functools
from typing import Optional

import torch


def get_num_sm(device: torch.device) -> int:
    """Number of streaming multiprocessors on ``device``."""
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.cache
def _cc(index: Optional[int]) -> tuple:
    return torch.cuda.get_device_capability(index)


def device_cc(device: Optional[torch.device] = None) -> tuple:
    """Compute capability of ``device`` as ``(major, minor)``."""
    index = None if device is None else torch.device(device).index
    return _cc(index)


def is_sm90(device: Optional[torch.device] = None) -> bool:
    """Whether ``device`` is a Hopper GPU."""
    return device_cc(device)[0] == 9


def is_sm100(device: Optional[torch.device] = None) -> bool:
    """Whether ``device`` is a Blackwell GPU this package builds for."""
    return device_cc(device) == (10, 0)


def is_supported(device: Optional[torch.device] = None) -> bool:
    """Whether this package can run on ``device``."""
    if not torch.cuda.is_available():
        return False
    return is_sm90(device) or is_sm100(device)


def get_capabilities(device: Optional[torch.device] = None) -> dict:
    """Static capability descriptor for the running GPU.

    Integrators use this to route a request without attempting a plan that would
    be refused. Every key is always present, with empty or ``False`` values when
    the device is not supported, so a caller can read any key without a
    ``KeyError``.

    Every value describes the quantized path, which is the only decode path.
    ``dtypes`` is the Q dtype; K and V are always packed ``uint8`` pools plus a
    separate scale pool. ``head_dims`` is the accepted set for the running
    architecture, comma-joined.
    """
    if torch.cuda.is_available():
        sm90 = is_sm90(device)
        sm100 = is_sm100(device)
    else:
        sm90 = sm100 = False
    supported = sm90 or sm100
    arch = "sm90" if sm90 else "sm100" if sm100 else "unsupported"
    # Imported here rather than at module scope: ``wrapper`` imports this module.
    from .wrapper import _ACCEPTED_HEAD_DIMS
    return {
        "arch": arch,
        "paged": supported,
        "dtypes": (["float8_e4m3fn"] if sm100 else ["bfloat16"]) if supported else [],
        "head_dims": (",".join(str(d) for d in _ACCEPTED_HEAD_DIMS[arch])
                      if supported else ""),
        # One scale block per (page, kv-head), so 64 is the only page size.
        "page_sizes": [64] if supported else [],
        "reductions": ["none", "kernel"] if supported else [],
        "speculative": False,
        "atomic_reduction": False,
        "skip_softmax": False,
        "kv_layouts": ["NHD"] if supported else [],
        "quantized_bits": [2, 4] if supported else [],
        "quantized_validated": supported,
        "quantized_page_sizes": [64] if supported else [],
    }


def api(func=None, **_kwargs):
    """Passthrough decorator kept for signature compatibility with FlashInfer's.

    It accepts and ignores keyword arguments such as ``trace=``.
    """
    if func is None:
        return lambda f: f
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
