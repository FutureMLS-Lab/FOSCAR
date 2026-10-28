# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""Local utilities: dtype mapping, arch detection, capability query, and a no-op
API decorator — replacing the FlashInfer-specific helpers the wrapper used."""
import functools
from typing import Optional

import torch
import cutlass

_TORCH_TO_CUTLASS = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float8_e5m2: cutlass.Float8E5M2,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
}


def torch_to_cutlass(dtype: torch.dtype):
    if dtype not in _TORCH_TO_CUTLASS:
        raise TypeError(
            f"cutedsl-decode does not support {dtype}; supported: {list(_TORCH_TO_CUTLASS)}"
        )
    return _TORCH_TO_CUTLASS[dtype]


def get_num_sm(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.cache
def _cc(index: Optional[int]) -> tuple:
    return torch.cuda.get_device_capability(index)


def device_cc(device: Optional[torch.device] = None) -> tuple:
    index = None if device is None else torch.device(device).index
    return _cc(index)


def is_sm90(device: Optional[torch.device] = None) -> bool:
    return device_cc(device)[0] == 9


def is_sm100(device: Optional[torch.device] = None) -> bool:
    return device_cc(device) == (10, 0)


def is_supported(device: Optional[torch.device] = None) -> bool:
    if not torch.cuda.is_available():
        return False
    return is_sm90(device) or is_sm100(device)


# Per-arch supported KV page sizes. The floor (8) is the WGMMA 128B-swizzle row atom;
# the ceiling differs by arch. SM90's KV tile is mma_n = 128, so page_size up to 128
# works (the degenerate one-page-per-tile case). SM100's PV-gemm contraction tile is
# mma_tile_k = 128*8 // dtype.width = 64 for 16-bit, which caps page_size at 64.
SM90_PAGE_SIZES = (8, 16, 32, 64, 128)
SM100_PAGE_SIZES = (8, 16, 32, 64)


def supported_page_sizes(device: Optional[torch.device] = None) -> tuple:
    if is_sm90(device):
        return SM90_PAGE_SIZES
    if is_sm100(device):
        return SM100_PAGE_SIZES
    return ()


def get_capabilities(device: Optional[torch.device] = None) -> dict:
    """Static capability descriptor for the running GPU. Integrators use this to
    avoid requesting features the current arch doesn't support. The full key set is
    always returned (with empty/False values when unsupported) so callers can read
    e.g. ``caps["speculative"]`` without a KeyError on non-SM90/SM100 hardware."""
    if torch.cuda.is_available():
        sm90 = is_sm90(device)
        sm100 = is_sm100(device)
    else:
        sm90 = sm100 = False
    supported = sm90 or sm100
    arch = "sm90" if sm90 else "sm100" if sm100 else "unsupported"
    # Imported here, not at module level: `wrapper` imports this module.
    from .wrapper import _ACCEPTED_HEAD_DIMS
    return {
        "arch": arch,
        "paged": supported,
        # Every value below describes the QUANTIZED path, which is the only decode
        # path left -- the dense (bf16/fp16) kernels and their wrapper dispatch were
        # removed. `dtypes` is the Q dtype: SM100 requires fp8 Q, SM90 takes bf16.
        # K/V are always packed uint8 code pools. `page_sizes` collapses to the
        # quant-pool contract (one scale block per (page, kv-head)), so
        # `supported_page_sizes` / SM90_PAGE_SIZES / SM100_PAGE_SIZES are no longer
        # reachable from here and describe hardware, not this library.
        "dtypes": (["float8_e4m3fn"] if sm100 else ["bfloat16"]) if supported else [],
        # The running architecture's accepted set from `_ACCEPTED_HEAD_DIMS`, comma-joined
        # ("128,256" on SM90, "128" on SM100): `PagedDecodeWrapper` refuses everything
        # else, and the kernels raise below 128. This read "multiples of 64" until
        # 2026-09-04, which advertised a configuration that wrote out of range.
        "head_dims": (",".join(str(d) for d in _ACCEPTED_HEAD_DIMS[arch])
                      if supported else ""),
        "page_sizes": [64] if supported else [],
        # "atomic" was dense-only; both quantized planners reject it.
        "reductions": ["none", "kernel"] if supported else [],
        # Speculative decode (q_len_per_req > 1), BLASST skip-softmax and the
        # atomic/cluster reduction were all properties of the deleted dense kernels.
        "speculative": False,
        "atomic_reduction": False,
        "skip_softmax": False,
        "kv_layouts": ["NHD"] if supported else [],
        # Quantized-KV decode: PagedDecodeWrapper.plan(bits=2|4). Requires
        # page_size == 64 and packed code pools (see the wrapper docstring).
        # BOTH arches are hardware-validated against a shared oracle: SM100 by
        # scripts/test_wrapper_quant_sm100.py, SM90 by baselines/validate.py
        # --arch sm90 plus baselines/check_sm90_split_parity.py, which covers all
        # three split-K merge mechanisms at every emittable split count, uniform
        # and ragged.
        "quantized_bits": [2, 4] if supported else [],
        "quantized_validated": supported,
        "quantized_page_sizes": [64] if supported else [],
    }


def api(func=None, **_kwargs):
    """No-op passthrough replacing FlashInfer's @flashinfer_api (logging) decorator.
    Accepts and ignores keyword args (e.g. trace=...) for signature compatibility."""
    if func is None:
        return lambda f: f
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
