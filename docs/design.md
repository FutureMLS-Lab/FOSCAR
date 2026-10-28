# Design and Internals

> **Scope.** This document covers the wrapper, the module layout, arch dispatch, workspace
> slicing and stream binding. The kernel internals themselves are documented where they
> live: the `int_fold_decode.py` and `int_decode_pingpong.py` module docstrings for the
> dataflow, the README's quantized-KV sections for the envelope, and the per-kernel
> engineering records under [`docs/kernel/`](kernel/README.md) for the development
> history. A per-kernel walkthrough in this file is outstanding work.

This document describes how `cutedsl-decode` works internally. It is intended for contributors
and integrators who want to understand the kernel architecture, arch dispatch, and feature
gating. Users of the public API do not need to read this.

## Package provenance

The kernels are adapted from two upstream sources (see NOTICE):

- **NVIDIA CUTLASS** — the Hopper FMHA CuTeDSL example
  (`examples/python/CuTeDSL/hopper/fmha.py`), adapted into the package's original dense
  SM90 paged-decode kernel. That kernel was removed when the package became
  quantized-only.
- **FlashInfer** — the CuTe DSL GQA decode kernels (`flashinfer/cute_dsl/attention/`) and
  the paged-decode wrapper.

The package is a **clean fork**: the upstream-derived modules (`wrapper.py` and the
`_sm100` kernel files that carry NVIDIA copyright headers) no longer match upstream,
because every decode path is now quantized, and they do not auto-track upstream changes.
Future improvements from FlashInfer must be manually merged. Two FlashInfer-specific
touchpoints were replaced with local code:

- `@flashinfer_api` (an API-logging decorator) → local no-op decorator in `_utils.py`.
- `get_num_sm(device)` → `torch.cuda.get_device_properties(device).multi_processor_count`.

## Module layout

```
src/cutedsl_decode/
├── __init__.py                    # public API: PagedDecodeWrapper, get_capabilities, is_supported
├── wrapper.py                     # unified wrapper: plan/run + arch dispatch + feature gating
├── _utils.py                      # arch detection, get_capabilities, no-op api decorator
├── quant_layout.py                # (axis, group) scale-block layout and quant envelope, both kernels
├── kernel_contract.py             # constructor contract both decode kernels share
├── kernel_cache.py                # persistent on-disk cache of compiled kernels
├── _sm90/
│   ├── int_decode_pingpong.py     # Hopper int2/int4 decode — the SHIPPED SM90 path
│   ├── serving_cache.py           # packed-code + interleaved-scale POOL CONTRACT
│   ├── splitk_combine.py, scheduler.py, pingpong_runtime.py, int_append.py
│   └── warpmma_decode.py          # earlier-gen configurable-granularity int2/int4
└── _sm100/
    ├── int_fold_decode.py         # Blackwell fp8-fold quantized decode — the SHIPPED SM100 path
    ├── int_fold_asm.py            # single-instruction inline-asm primitives for the fold kernel
    ├── gqa_decode.py              # SHARED split-K reduction library (see below)
    ├── fold_runtime.py            # compile/launch glue, BITS_TUNING, resolve_fold_tiling
    └── scheduler.py               # split-K cost model / planner
```

`_sm100/gqa_decode.py` holds the split-K reduction library the SM100 kernel uses,
despite a filename that suggests a kernel. `int_fold_decode.py` imports 10 module-level
names from it, and also imports the
`GroupedQueryAttentionDecode` class, which serves as the namespace for `reduction_none` /
`reduction_epilogue` / `reduction_cluster` / `launch_reduction` / `reduction_kernel` /
`gqa_pack` / `gemm_view*` / `can_implement`.

**Do not rename the module or the class, and do not prune anything that looks unused here.**
Two of its dependencies are invisible to a grep of those import lists: `launch_reduction`
reaches the kernel through a class-qualified `GroupedQueryAttentionDecode.reduction_kernel(...)`
call, and `reduction_epilogue` resolves `gmem_fmax` as a module global rather than taking it as
an argument. Breaking either one fails at TRACE time, not at import time — so the failure
surfaces as a broken `reduction="kernel"` launch rather than an `ImportError`. `can_implement`
is likewise called unbound with a foreign `self` (the fold kernel passes its own instance),
which is why it is a plain method on a class nothing instantiates.

The `_sm90` and `_sm100` subpackages are private (leading underscore). Nothing arch-specific
is exported from the top level.

## Arch dispatch

`PagedDecodeWrapper` detects the compute capability of the current CUDA device at `plan()`
time via `_utils.is_sm90()` and stores `self._is_sm90`:

- SM major == 9 → SM90 path (`IntPingPongDecodeSm90`)
- SM (10, 0) → SM100 path (`IntDecodeSm100`)
- Anything else → `is_supported()` returns False; constructing a wrapper on an unsupported
  device will raise at kernel dispatch time.

Feature gating happens inside `plan()` and `run()`. The same features are rejected on both
arches:

```
plan():  q_len_per_req > 1                       →  NotImplementedError (checked before arch detection)
         reduction="atomic"                      →  NotImplementedError
run():   skip_softmax_threshold_scale_factor set →  NotImplementedError
         lse passed                              →  NotImplementedError
```

This is enforced with explicit guard branches, not by the kernel itself — so the error message
is actionable and printed before any compilation is attempted.

## Workspace slicing

The `float_workspace_buffer` carve is the SM100 kernel-reduction path. When
`reduction="kernel"` on SM100, `plan()` carves the buffer into four views:

```
o_partial : [kv_splits, batch, q_len, num_qo_heads, head_dim]  (float32)
l_partial : [kv_splits, batch, q_len, num_qo_heads]             (float32)
m_partial : [kv_splits, batch, q_len, num_qo_heads]             (float32)
m         : [batch, q_len, num_qo_heads]                         (float32, initialized to -inf)
```

The `m` buffer is filled with `-inf` at `run()` time before each kernel launch. With
`reduction="none"` no workspace is needed; output is written directly to `out_4d`.

SM90 does not use the caller's buffer: `plan()` allocates the ping-pong kernel's own partial
buffers through `pingpong_runtime.alloc_pingpong_workspace`.

## Stream binding

The CuTe DSL kernels are compiled with `use_tvm_ffi_env_stream=True`, which means they launch
on whatever CUDA stream is currently bound to the TVM-FFI environment. Both launch sites,
`_sm100/fold_runtime.launch_fold` and `_sm90/pingpong_runtime.launch_pingpong`, open a
`tvm_ffi.use_torch_stream()` scope around the launch, so `run()` always lands on torch's
current stream, including under CUDA graph capture.

Code that calls a compiled kernel directly, rather than through those two functions, must open
that scope itself:

```python
import tvm_ffi
with tvm_ffi.use_torch_stream():
    compiled(...)
```

Without this binding the kernel may launch on a different stream from the surrounding torch
operations, causing incorrect sequencing.
