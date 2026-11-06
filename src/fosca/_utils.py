# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: MIT
"""Local utilities: dtype mapping, arch detection, capability query, and a no-op
API decorator."""
import contextlib
import functools
from typing import Optional

import torch


def torch_to_cutlass(dtype: torch.dtype):
    """Map a torch dtype to the CuTe DSL dtype of the same width.

    The DSL is imported here rather than at module scope so that importing the
    package does not require a CUDA toolkit.
    """
    import cutlass

    table = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float8_e5m2: cutlass.Float8E5M2,
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    }
    if dtype not in table:
        raise TypeError(f"fosca does not support {dtype}; supported: {list(table)}")
    return table[dtype]


def get_num_sm(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.cache
def _cc(index: int) -> tuple:
    return torch.cuda.get_device_capability(index)


def cuda_index(device=None) -> int:
    """The concrete CUDA ordinal of `device`; None, or a CUDA device without an index, means the
    current device AT THIS CALL.

    The capability cache is keyed by this ordinal."""
    if device is None:
        return torch.cuda.current_device()
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"{device} is not a CUDA device")
    return torch.cuda.current_device() if device.index is None else device.index


def device_cc(device=None) -> tuple:
    return _cc(cuda_index(device))


#: The compute capabilities a decode kernel is built for, by the architecture name every API
#: reports. A WHITELIST: a device of any other compute capability has no decode kernel.
ARCH_OF_CC = {(9, 0): "sm90", (10, 0): "sm100"}


def device_arch(device=None) -> Optional[str]:
    """"sm90", "sm100", or None for a device no decode kernel is built for."""
    return ARCH_OF_CC.get(device_cc(device))


def require_arch(device=None) -> str:
    """`device_arch`, refusing a device no decode kernel is built for."""
    arch = device_arch(device)
    if arch is None:
        major, minor = device_cc(device)
        raise RuntimeError(
            f"fosca has no kernel for compute capability {major}.{minor}; it runs on "
            f"sm90 (compute capability 9.0) and sm100 (10.0)")
    return arch


@functools.cache
def _dsl_target() -> tuple:
    """(target, architecture name) of the CuTe DSL's compile target, e.g. ("sm_90a", "sm90").

    Read once per process: the DSL fixes the target the first time it is read and never
    changes it, so every later plan pays a cached lookup rather than a DSL call."""
    from cutlass.cutlass_dsl import CuTeDSL  # lazy: CPU tests import this module

    target = str(CuTeDSL._get_dsl().envar.arch)
    digits = "".join(ch for ch in target.split("_", 1)[-1] if ch.isdigit())
    return target, f"sm{digits}"


def require_dsl_target(arch: str) -> None:
    """Refuse to compile for `arch` when the CuTe DSL's compile target is another architecture.

    The DSL compiles every kernel of a process for ONE target, `envar.arch`, fixed the first
    time it is read: `CUTE_DSL_ARCH` if set, else the architecture of CUDA device 0 as the
    process sees it (after `CUDA_VISIBLE_DEVICES`), whatever device is current. A wrapper or
    cache on a device of another architecture would otherwise compile a kernel for the wrong
    GPU, and the kernel cache would file it under the DSL's target
    (`kernel_cache.dsl_identity`). Set `CUTE_DSL_ARCH` (sm_90a or sm_100a) before the DSL is
    imported to choose the target."""
    target, name = _dsl_target()
    if name != arch:
        raise RuntimeError(
            f"the CuTe DSL in this process compiles for {target}, so it cannot build the {arch} "
            f"kernel; the target is fixed once per process (CUTE_DSL_ARCH, else CUDA device 0), "
            f"so decode this device from a process whose CUTE_DSL_ARCH names its architecture")


def on_device(device):
    """A context that makes `device` current, or a no-op when it already is.

    A launch binds torch's current stream (`tvm_ffi.use_torch_stream`), which belongs to the
    current device, so a launch for tensors on another device must switch first."""
    index = cuda_index(device)
    if torch.cuda.current_device() == index:
        return contextlib.nullcontext()
    return torch.cuda.device(index)


def is_sm90(device=None) -> bool:
    return device_arch(device) == "sm90"


def is_sm100(device=None) -> bool:
    return device_arch(device) == "sm100"


def is_supported(device: Optional[torch.device] = None) -> bool:
    if not torch.cuda.is_available():
        return False
    return is_sm90(device) or is_sm100(device)


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
        # Every value below describes the QUANTIZED decode path. `dtypes` is the Q
        # dtype: SM100 requires fp8 Q, SM90 takes bf16. K/V are always packed uint8
        # code pools. `page_sizes` is the quant-pool contract: one scale block per
        # (page, kv-head) of 64 tokens.
        "dtypes": (["float8_e4m3fn"] if sm100 else ["bfloat16"]) if supported else [],
        # The running architecture's accepted set from `_ACCEPTED_HEAD_DIMS`, comma-joined
        # ("128,256" on SM90, "128" on SM100): `PagedDecodeWrapper` refuses everything
        # else, and the kernels raise below 128.
        "head_dims": (",".join(str(d) for d in _ACCEPTED_HEAD_DIMS[arch])
                      if supported else ""),
        "page_sizes": [64] if supported else [],
        # "atomic" is rejected by both quantized planners.
        "reductions": ["none", "kernel"] if supported else [],
        # Speculative decode (q_len_per_req > 1), BLASST skip-softmax and the
        # atomic/cluster reduction are not available on the quantized path.
        "speculative": False,
        "atomic_reduction": False,
        "skip_softmax": False,
        # The cache is the library's own quantized layout, `QuantKVLayout`: per (page, KV head)
        # blocks of packed codes plus a scale block, built by `QuantKVLayout.allocate` or
        # `.bind`. It is neither FlashInfer's NHD nor its HND.
        "kv_layouts": ["QuantKVLayout"] if supported else [],
        # Quantized-KV decode: `PagedDecodeWrapper.plan(..., layout)` over a
        # `QuantKVCache` of 64-token pages (see the wrapper docstring).
        # Both arches are validated against a shared oracle, and the SM90 checks
        # cover all three split-K merge mechanisms at every emittable split count,
        # uniform and ragged.
        "quantized_bits": [2, 4] if supported else [],
        "quantized_validated": supported,
        "quantized_page_sizes": [64] if supported else [],
    }


def api(func=None, **_kwargs):
    """No-op passthrough with FlashInfer's @flashinfer_api (logging) signature.
    Accepts and ignores keyword args (e.g. trace=...) for signature compatibility."""
    if func is None:
        return lambda f: f
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
