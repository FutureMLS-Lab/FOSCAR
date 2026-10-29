# Design and Internals

This page describes how `fosca` is built: the package layout, how the wrapper
picks a kernel for the device it runs on, the lifecycle of that wrapper, the workspace
the split-K reduction needs, and how a launch is bound to a CUDA stream. It is written
for contributors and for integrators wiring the package into a serving engine, who want
to know what happens behind `plan()` and `run()`.

If you only want to call the package, [README](../README.md) and the
[SGLang integration guide](integration.md) cover the API. The kernel dataflow is
documented in the module docstrings of `_sm100/int_fold_decode.py` and
`_sm90/int_decode_pingpong.py`, and the supported envelope is the
[support matrix](../README.md#support-matrix).

[NOTICE](../NOTICE) records what this package derives from and redistributes, and the
license each part carries. The derived files are modified and do not track upstream, so an
upstream change has to be merged by hand.

## Package layout

```
src/fosca/
├── __init__.py                 public API: PagedDecodeWrapper, QuantKVCache, QuantKVLayout
├── wrapper.py                  plan/run, arch dispatch, feature gating, graph mode
├── kv_cache.py                 the cache layout and the cache object a decode reads
├── graph_plan.py               graph-mode buckets and their host work list
├── _append_core.py             the quantize-and-write kernel both architectures share
├── _dequant_core.py            the paged read-back kernel both architectures share
├── _utils.py                   device detection and the capability descriptor
├── quant_layout.py             per-(page, kv-head) scale-block layout and the granularity envelope
├── kernel_contract.py          the constructor contract both decode kernels share
├── kernel_cache.py             persistent on-disk cache of compiled kernels
├── _vendor/                    QuACK and flash-attention subsets
├── _sm90/
│   ├── int_decode_pingpong.py  the Hopper decode kernel
│   ├── pingpong_runtime.py     compile and launch glue, workspace, NVVM options
│   ├── quant_append.py         the Hopper cache writer
│   ├── quant_dequant.py        the Hopper read-back
│   ├── scheduler.py            split-K planner
│   ├── serving_cache.py        the pool geometry the layout is built from
│   └── splitk_combine.py       external split-K combine kernel
└── _sm100/
    ├── int_fold_decode.py      the Blackwell decode kernel
    ├── int_fold_asm.py         single-instruction inline-asm primitives
    ├── fold_runtime.py         compile and launch glue, per-bits tuning, tiling
    ├── code_layout.py          pool byte layout for the fold kernel
    ├── item_loop.py            the persistent item loop
    ├── gqa_decode.py           the shared split-K reduction library
    ├── quant_append.py         the Blackwell cache writer
    ├── quant_dequant.py        the Blackwell read-back
    └── scheduler.py            split-K planner
```

`_sm90` and `_sm100` are private, and nothing architecture-specific is exported at the
top level. `_utils`, `quant_layout` and `kernel_contract` import nothing from the CuTe
DSL, so `import fosca` works on a machine with no CUDA toolkit installed; the
DSL is imported at the point a kernel is compiled.

`_sm100/gqa_decode.py` is a split-K reduction library rather than a kernel, despite the
name. `int_fold_decode.py` imports the `GroupedQueryAttentionDecode` class from it and
uses it as the namespace for the reduction entry points. Two of those dependencies are
invisible to a search of the import list: `launch_reduction` reaches the kernel through a
class-qualified call, and `reduction_epilogue` reads `gmem_fmax` as a module global. Keep
the module name, the class name and `gmem_fmax` as they are, because breaking either
dependency fails at trace time rather than at import time, as a broken reduction launch.

## Architecture dispatch

`_utils` reads the compute capability of the current device. `is_sm90` is true when the
major version is 9, `is_sm100` is true only for exactly `(10, 0)`, and `is_supported` is
true when CUDA is available and one of those holds. `get_capabilities()` reports the same
decision as a dictionary whose keys are always present, with empty or false values on an
unsupported device, so a router can read any key without a `KeyError`.

```python
from fosca import get_capabilities, is_supported

print(is_supported())                 # True on an SM90 or SM100 device
print(get_capabilities()["arch"])     # "sm90", "sm100" or "unsupported"
```

`plan()` refuses anything else before it compiles. It raises `RuntimeError` when torch
reports no CUDA device, and `ValueError` naming the compute capability when the device is
neither Hopper nor Blackwell. The decision is stored on the wrapper and selects the
planner, the compile entry point and the launch site:

| Device | Kernel | Compile | Launch |
|---|---|---|---|
| SM90 (major 9) | `IntPingPongDecodeSm90` | `_sm90/pingpong_runtime.compile_pingpong` | `launch_pingpong` |
| SM100 (10, 0) | `IntDecodeSm100` | `_sm100/fold_runtime.compile_fold` | `launch_fold` |

Feature gating happens in the wrapper rather than in the kernel, so a refused request
raises an actionable message before any trace is attempted:

| Call | Argument | Result |
|---|---|---|
| `plan()` | `q_len_per_req` other than 1 | `NotImplementedError` |
| `plan()` | `reduction="atomic"` | `NotImplementedError` |
| `run()` | `lse=` | `NotImplementedError` |
| `run()` | `skip_softmax_threshold_scale_factor=` | `NotImplementedError` |

On a supported device, `get_capabilities()` reports the same set: `speculative`,
`skip_softmax` and `atomic_reduction` are false, and `kv_layouts` is `["NHD"]`.

Each architecture narrows the shared envelope further, and `plan()` raises for the rest.
SM100 takes fp8 e4m3 Q, accepts head dimension 128, and packs at most 8 Q head rows per
threadblock. SM90 takes bf16 Q and accepts head dimension 128 or 256, where 256 builds a
single layout. The [support matrix](../README.md#support-matrix) lists the full set.

## Wrapper lifecycle

`PagedDecodeWrapper` holds the plan-time state and the compiled kernels. The constructor
records the workspace tensor and the device taken from it and does nothing else.

`plan()` establishes everything `run()` reads. It validates the problem, converts
`indptr`, `indices` and `seq_lens` to int32 on the wrapper's device, runs the
architecture's split-K planner, slices the workspace, and compiles the kernel. The page
table and sequence lengths are stored on the wrapper, which is why `run()` takes only the
query and the pools; calling `plan()` again replaces them, and SM90 also builds its own
2-D block table from the same inputs.

Compilation is keyed on the static configuration: bits, head counts, head dimension, page
size, quantization granularity, scale-pool precision, reduction mode, output dtype and any
`kernel_kwargs`. Batch size, physical page count and split count are traced as symbols, so
one compiled module serves every batch within that configuration. The result is memoized
per process by `functools.cache` and cached on disk by `kernel_cache`, which serves a
repeat of the same trace to a later process. That pair is what keeps the first decode step
from paying for a trace. `plan()` also compiles the SM90 external combine when the plan
uses one.

A plan is transactional. Every state field is invalidated on entry to `plan()` and again
if the call raises, so a replan that fails leaves the wrapper unplanned and `run()` raises
`RuntimeError` rather than launching a half-built plan.

`plan()` synchronizes when `max_kv_len` is omitted, because it reads `seq_lens.max()` for
the planner's tile count.

`run()` requires a plan and checks the cache against it before launching: the Q dtype, the
cache's layout against the planned one, the scale pool's shape against the planned granularity, the scale pool's
dtype against the planned precision, and, on SM100, the declared K pool order. Each of
those mismatches would otherwise decode to finite but wrong values.

Four read-only properties report what the plan decided: `kv_splits`, `reduction`, `bits`
and `scale_pool_dtype`.

The reduction mode is resolved differently on the two architectures, and `plan()` enforces
the difference. On SM100 the mode selects a compile-time epilogue, so the planner chooses
it before the trace. On SM90 the kernel always writes partials and a merge follows, so one
split is `"none"` and any larger split count is `"kernel"`; asking for the other pairing
raises.

## Workspace

On SM100 with `reduction="kernel"`, `plan()` carves the caller's `float_workspace_buffer`
into four float32 views, in this order:

| View | Shape |
|---|---|
| `o_partial` | `[kv_splits, batch, 1, num_qo_heads, head_dim]` |
| `l_partial` | `[kv_splits, batch, 1, num_qo_heads]` |
| `m_partial` | `[kv_splits, batch, 1, num_qo_heads]` |
| `m` | `[batch, 1, num_qo_heads]` |

`plan()` raises `RuntimeError` naming the required byte count when the buffer is too
small. `run()` fills `m` with `-inf` before every launch, because the kernel reduces into
it with an atomic max and needs the identity value there. With `reduction="none"` the
kernel writes straight to the output and no workspace is used.

SM90 does not use the caller's buffer. `plan()` allocates the kernel's own `o_partial`,
`lse` and `counters` through `alloc_pingpong_workspace`. `counters` is allocated zeroed
and must stay zeroed across launches, because the kernel rearms its own cell.

## Stream binding

Both kernels are compiled with `use_tvm_ffi_env_stream=True`, so a launch goes to whatever
CUDA stream is bound to the TVM-FFI environment at call time.
`_sm100/fold_runtime.launch_fold` and `_sm90/pingpong_runtime.launch_pingpong` open a
`tvm_ffi.use_torch_stream()` scope around the launch, which binds it to torch's current
stream. `run()` therefore lands in the caller's stream, including under CUDA graph
capture, with no stream setup around the call. The SM90 external combine launches inside
the same scope.

Code that calls a compiled kernel directly, rather than through those two functions, must
open that scope itself. Without it the kernel may launch on a different stream from the
torch work around it, and the sequencing is wrong.
