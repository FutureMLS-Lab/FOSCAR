# SGLang Integration Guide

This guide shows how to wire `cutedsl-decode` into an SGLang-style attention backend.
The wrapper follows the same plan/run lifecycle as FlashInfer's
`BatchDecodeWithPagedKVCacheWrapper`, so adoption is nearly mechanical.

Every decode path in this package is **quantized**: K and V live in the cache as packed
int2/int4 codes plus a separate affine `(scale, zero)` block, and `plan()` requires `bits=2`
or `bits=4`. There is no dense bf16/fp16/fp8 KV mode.

## Overview

`PagedDecodeWrapper` has two phases:

1. **`plan()`** — called once per decode step when the page table or sequence lengths change.
   Compiles the kernel for the static configuration on first use. Repeat calls with the same
   configuration hit the in-process memo, and `cutedsl_decode.kernel_cache` serves a repeat
   of the same trace from disk across processes.
2. **`run()`** — called every decode step. Graph-capturable (see below).

## Constructor

```python
import torch
from cutedsl_decode import PagedDecodeWrapper

workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
wrapper = PagedDecodeWrapper(workspace)
```

Allocate one `PagedDecodeWrapper` per layer (or per worker). On SM100 the workspace is a flat
`uint8` CUDA tensor used for the split-K partial-result buffers (`reduction="kernel"`); 256 MiB
is sufficient for typical decode batches, and the wrapper raises `RuntimeError` with the
required byte count if it is too small. On SM90 the wrapper allocates the kernel's own partial
buffers and does not use the caller's workspace.

The constructor also accepts `use_cuda_graph=`, which is ignored and kept only so existing
callers do not break.

## plan()

```python
wrapper.plan(
    indptr,          # int32 [batch_size + 1]: prefix-sum offsets into indices
    indices,         # int32 [num_pages_total]: virtual page indices
    seq_lens,        # int32 [batch_size]: per-sequence KV token count
    num_qo_heads=32,
    num_kv_heads=8,        # GQA group num_qo_heads / num_kv_heads must be <= 8
    head_dim=128,          # 128; also 256 on SM90 (token-wise K and V, f32 scale pool only)
    page_size=64,          # 64 only (one code block and one scale block per (page, kv-head))
    q_data_type=torch.float8_e4m3fn,   # SM100: fp8 e4m3 only. SM90: torch.bfloat16 only
    bits=2,                # REQUIRED: 2 or 4
    # o_data_type=None,    # defaults to torch.bfloat16
    sm_scale=None,         # defaults to 1 / sqrt(head_dim)
    kv_splits=None,        # None: the arch cost model picks
    reduction="auto",      # "auto"|"none"|"kernel"
    q_len_per_req=1,       # must be 1
    max_kv_len=None,       # pass explicitly to avoid a GPU→CPU sync
    non_blocking=True,
    # k_quant=("token", 128), v_quant=("token", 128),   # granularity, see below
    # kernel_kwargs=None,                                # ADVANCED, see below
)
```

`q_data_type` defaults to `torch.bfloat16`, which is what SM90 takes; on SM100 pass
`torch.float8_e4m3fn` explicitly. The SM100 kernel takes no `q_scale`: fold the per-tensor Q
descale into the softmax scale yourself (`sm_scale=sm_scale / q_scale`).

Call `plan()` whenever the batch's page table or sequence lengths change — typically once per
decode forward pass when the scheduler has updated the running batch. For a static batch
(same seqlens and page table across steps), calling `plan()` once and caching the wrapper
state across steps is valid.

With `kv_splits=None`, the split count and the reduction mode come from the per-arch quantized
cost model (`_sm100/scheduler.plan_splits_sm100`, `_sm90/scheduler.plan_splits`); passing
`kv_splits` explicitly overrides it. On SM90 the reduction mode is derived from the split
count rather than chosen (one split is `"none"`, more than one is `"kernel"`), so asking for
the other pairing raises. `max_kv_len=` matters here: without it `plan()` syncs to read
`seq_lens.max()`, because the cost model needs a real tile count.

**JIT warm-up:** on the first call with a new static configuration (bits, head counts,
granularity, scale-pool precision, reduction or combine mode, output dtype and any
`kernel_kwargs`), `plan()` traces and compiles the kernel.
A trace takes seconds (`kernel_cache.py` records the measured figures), and the on-disk
kernel cache removes that cost for a configuration any earlier process has already compiled.
To avoid it on the first real decode step, call `plan()` with a small dummy batch during
model loading:

```python
# During model initialization (not on the critical path)
wrapper.plan(dummy_indptr, dummy_indices, dummy_seq_lens,
             num_qo_heads=32, num_kv_heads=8, head_dim=128, page_size=64,
             q_data_type=torch.float8_e4m3fn,   # torch.bfloat16 on SM90
             bits=2)
```

The cache lives under `$XDG_CACHE_HOME/cutedsl-decode/kernel-cache` (or
`~/.cache/cutedsl-decode/kernel-cache`); `CUTEDSL_DECODE_KERNEL_CACHE_DIR` moves it and
`CUTEDSL_DECODE_KERNEL_CACHE=0` disables it.

## run()

```python
out = wrapper.run(
    q,             # [batch_size, num_qo_heads, head_dim] or [batch_size, 1, num_qo_heads, head_dim]
    k_cache,       # packed uint8 K code pool
    v_cache,       # packed uint8 V code pool
    out=None,      # pre-allocated output buffer; pass one in a serving loop
    sm_scale=None,        # per-call override of the plan-time softmax scale
    o_scale=None,         # output scale folded into the epilogue; default 1.0
    enable_pdl=True,
    scale_pool=scales,    # REQUIRED: the affine (scale, zero) block
    pscale_lp=None,       # SM100: derived from scale_pool with a device sync unless passed
    k_pool_order=None,    # SM100: "permuted" or "logical", required where the K pool is permuted
)
```

The output shape mirrors the input `q` shape: a 3-D `q` gives a 3-D
`[batch_size, num_qo_heads, head_dim]` output, and a 4-D `q` gives a 4-D one. When `out=` is
passed, `run()` writes into it and returns it.

`k_cache` and `v_cache` are the packed code pools (`uint8`, NHD-contiguous, `page_size=64`),
and `scale_pool` is the per-(page, kv-head) affine block. With the default
`("token", head_dim)` granularity for both tensors it is the interleaved f32 pool
`[num_pages, num_kv_heads, page_size, 4] = (s_k, z_k, s_v, z_v)`; with any other granularity
it is the flat sectioned block `[num_pages, num_kv_heads, block_tot]` laid out by
`cutedsl_decode.quant_layout.quant_block_offsets`. `run()` checks the pool's shape against
the planned granularity and its dtype against the planned scale precision. Build the pools
with `cutedsl_decode._sm90.serving_cache` (the SM90 pool contract) or see
`scripts/probes/repack_sm100.py` for the SM100 reference repack.

**On SM90 both code pools are chunk-major.** Each (page, kv-head) block is `[per/16, 128, 16]`
bytes, where `per` is one warpgroup thread's row of quantized values, 32 bytes at INT4 and 16
at INT2 at head_dim 128 (at head_dim 256, which `PagedDecodeWrapper` accepts on SM90 only, a row
is twice as long): chunk c of thread t holds bytes [16c, 16c+16)
of that thread's row. The kernel names
this order in `IntPingPongDecodeSm90.code_pool_order`, and `run()` refuses a row-major
`[num_pages, num_kv_heads, 128, per]` pool by its shape. Nothing in the bytes says which order
a pool is in, so a row-major pool that is reshaped or viewed to the chunk-major shape passes
that check and decodes to wrong values at INT4. Write the pool with
`repack_rs.repack_K_rs` / `repack_V_rs(order=IntPingPongDecodeSm90.code_pool_order)`, or
convert thread rows with `serving_cache.code_rows_to_order(rows,
IntPingPongDecodeSm90.code_pool_order)`. At INT2 at head_dim 128 a row is a single chunk, so
the two orders are the same bytes.

**Hoist both `out=` and `pscale_lp=` out of the hot path.** Omit `out=` and every call
allocates the output; omit `pscale_lp=` on SM100 and every call does a `.max().item()` device
sync to derive it. `wrapper.pscale_lp(scale_pool)` computes the value once; recompute it only
when the pool's maximum V scale changes.

**`k_pool_order` is mandatory on SM100 wherever the kernel reads a permuted K pool**, which is
int4 at every configuration and int2 with the default E2M1 feed.
Nothing in the pool's bytes reveals the order, so a pool in the wrong order decodes to finite,
plausible, wrong values; the wrapper therefore refuses an undeclared or contradictory order
rather than guessing. Write the pool by asking `IntDecodeSm100.k_pool_is_permuted`, and see
[`docs/K_STORE_PERM.md`](K_STORE_PERM.md) for the store order an external writer has to meet.
SM90 ignores the argument; its code-pool order is the chunk-major one described above, and
`run()` checks it by shape.

## Routing with get_capabilities()

Use `get_capabilities()` to decide whether a batch can be routed here, avoiding
`NotImplementedError` or `ValueError` at runtime:

```python
from cutedsl_decode import get_capabilities, is_supported

caps = get_capabilities()

# Speculative/MTP decode (q_len_per_req > 1) is not supported on either arch.
if not is_supported() or (q_len_per_req > 1 and not caps["speculative"]):
    return fallback_attention_backend(q, k_cache, v_cache, ...)
```

`speculative`, `skip_softmax` and `atomic_reduction` are `False` on both arches, and
`kv_layouts` is `["NHD"]`. Requesting `q_len_per_req > 1` or `reduction="atomic"` from
`plan()`, or `lse=` or `skip_softmax_threshold_scale_factor=` from `run()`, raises
`NotImplementedError`.

## Quantization granularity

`k_quant` / `v_quant` select the per-tensor `(axis, group)` granularity independently for K
and V: `("token", head_dim)` (the default and the shipped fast path), `("token", head_dim // 2)`
for K, or `("channel", G)` with `G` in `{64, 128, 256}`. V token-64 is not supported. The pool
layout must match the granularity (see `quant_layout.quant_block_offsets`). At head_dim 256,
which only SM90 accepts, the kernel builds the default pair with a float32 scale pool and nothing
else, and `plan()` refuses every other pair or pool precision with a `ValueError`.

K token-64 needs a **bfloat16** scale pool. On SM90 build the pool with
`make_serving_cache_flex(..., sdtype=torch.bfloat16)` and pass
`kernel_kwargs={"sdtype": torch.bfloat16}` to `plan()`; a mismatch is an error naming both
arguments. On SM100 K token-64 selects the bfloat16 pool by default, and
`wrapper.scale_pool_dtype` reports the dtype the plan compiled for.

`kernel_kwargs=` is ADVANCED. On SM100 it forwards constructor knobs verbatim to the kernel,
and anything set there overrides the promoted per-bits tuning. On SM90 it accepts only
`fused_combine` (the in-kernel split-K merge against the external combine, for A/B work) and
`sdtype`.

## CUDA graph capture

`run()` is graph-capturable. Both launch sites bind the kernel to torch's current stream
internally, so no stream setup is needed around the call:

```python
static_out = torch.empty_like(...)          # [batch_size, num_qo_heads, head_dim], bfloat16
lp = wrapper.pscale_lp(scale_pool)          # SM100: must be precomputed before capture

# warm up
wrapper.run(q, k_cache, v_cache, out=static_out, scale_pool=scale_pool,
            pscale_lp=lp, k_pool_order="permuted")
torch.cuda.synchronize()

# capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    wrapper.run(q, k_cache, v_cache, out=static_out, scale_pool=scale_pool,
                pscale_lp=lp, k_pool_order="permuted")

# replay
g.replay()
```

Under capture on SM100 you must pass `pscale_lp=` explicitly: deriving it reads
`scale_pool.max().item()`, which is a device→host sync and is illegal during capture. Pass a
static `out=` so the graph writes into a buffer you own. The `k_pool_order` shown is for a
configuration that reads a permuted K pool; SM90 ignores both `pscale_lp` and `k_pool_order`.

## Feature reference

| plan() parameter | Default | Notes |
|---|---|---|
| `bits=` | — | REQUIRED: `2` or `4`; selects the stored bit width |
| `page_size=` | — | must be `64` |
| `head_dim=` | — | `128`, or `256` on SM90 (default `k_quant`/`v_quant` and a float32 scale pool only) |
| `q_data_type=` | `torch.bfloat16` | SM90: bfloat16 only. SM100: `torch.float8_e4m3fn` only |
| `o_data_type=` | `torch.bfloat16` | |
| `reduction="auto"` | the arch cost model picks `none` or `kernel` | |
| `reduction="none"` | no split-K; requires kv_splits==1 | |
| `reduction="kernel"` | split-K with an LSE merge | SM100 uses the workspace buffer |
| `q_len_per_req=1` | plain decode | anything else raises |
| `k_quant=`, `v_quant=` | `("token", head_dim)` | per-tensor `(axis, group)` granularity |
| `kernel_kwargs=` | None | ADVANCED: see "Quantization granularity" |

| run() parameter | Default | Notes |
|---|---|---|
| `scale_pool=` | None | REQUIRED: the affine (scale, zero) block for the packed code pools |
| `out=` | None (allocate per call) | pass a pre-allocated buffer in a serving loop |
| `o_scale=` | 1.0 | folded into the epilogue; no extra kernel pass |
| `pscale_lp=` | None (derive from `scale_pool`) | SM100: deriving it costs a device sync per call, so cache it |
| `k_pool_order=` | None | SM100: required wherever the planned kernel reads a permuted K pool |
| `lse=`, `skip_softmax_threshold_scale_factor=` | None | not supported; passing either raises |
