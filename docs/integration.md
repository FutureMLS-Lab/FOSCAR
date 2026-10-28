# SGLang Integration Guide

This page shows how to wire `cutedsl-decode` into an SGLang-style attention backend.
It covers the plan/run lifecycle, the page-table inputs `plan()` consumes, the arguments that belong outside the decode loop, CUDA graph capture, and the feature set the wrapper accepts.
Read it if you are adding this package to an engine's decode path; install steps, the support matrix, and pool construction are in the [README](../README.md).
The wrapper keeps the plan/run split of FlashInfer's paged-decode wrapper.

Every decode path in the package is quantized.
K and V live in the cache as packed 2-bit or 4-bit integer values plus a separate scale-and-offset block, and `plan()` requires `bits=2` or `bits=4`.
There is no dense bf16, fp16, or fp8 KV mode.

## Lifecycle

`PagedDecodeWrapper` has two phases.

1. `plan()` compiles the kernel for one static configuration and records the page table.
   Call it whenever the page table or the sequence lengths change.
2. `run()` launches one decode step.
   It is graph-capturable.

Allocate one wrapper per layer or per worker.
`plan()` is transactional: it discards the previous plan before it starts, so a wrapper whose `plan()` raised must be planned again before `run()`.

## Constructor

```python
import torch
from cutedsl_decode import PagedDecodeWrapper

workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
wrapper = PagedDecodeWrapper(workspace)
```

On SM100 the workspace holds the split-K partial results for `reduction="kernel"`.
It is a flat `uint8` CUDA tensor, and the wrapper raises `RuntimeError` with the required byte count when it is too small.
On SM90 the wrapper allocates its own partial buffers and the workspace stays unused.

Graph capture needs no constructor flag; see [CUDA graph capture](#cuda-graph-capture).

## Page-table construction

`plan()` takes the page table as three int32 tensors:

- `indptr` is `[batch_size + 1]` and holds the prefix sum of each sequence's page count.
- `indices` is `[num_pages_total]` and holds the physical page ids of every sequence, concatenated.
- `seq_lens` is `[batch_size]` and holds each sequence's KV length in tokens.

A block table of shape `[batch_size, pages_per_seq]` converts as follows.

```python
def plan_inputs(block_table, seq_lens, page_size=64):
    """block_table: int32 [batch_size, pages_per_seq] physical page ids."""
    device = block_table.device
    counts = ((seq_lens + page_size - 1) // page_size).to(torch.int32)
    indptr = torch.zeros(seq_lens.numel() + 1, dtype=torch.int32, device=device)
    indptr[1:] = torch.cumsum(counts, dim=0)
    indices = torch.cat(
        [block_table[b, :n] for b, n in enumerate(counts.tolist())]
    ).to(torch.int32)
    return indptr, indices, seq_lens.to(torch.int32)
```

A scheduler that tracks a last-page length instead of a token count obtains `seq_lens` from `(indptr.diff() - 1) * page_size + last_page_len`.

The page size is 64.
Each (page, kv-head) carries one code block and one scale block, so no other page size exists.

## plan()

```python
indptr, indices, seq_lens = plan_inputs(block_table, seq_lens)

wrapper.plan(
    indptr,                  # int32 [batch_size + 1]
    indices,                 # int32 [num_pages_total]
    seq_lens,                # int32 [batch_size]
    num_qo_heads=32,
    num_kv_heads=8,
    head_dim=128,
    page_size=64,
    q_data_type=torch.float8_e4m3fn,   # torch.bfloat16 on SM90
    bits=2,
)
```

`num_qo_heads` must be a multiple of `num_kv_heads`, and the GQA ratio must be at most 8.
`head_dim` is 128 on both architectures.
SM90 also accepts 256, where the kernel builds the default token-wise K and V layout with an f32 scale pool and refuses every other layout and scale-pool precision.

`q_data_type` is `torch.bfloat16` on SM90 and `torch.float8_e4m3fn` on SM100.
The SM100 kernel takes no Q scale, so fold the per-tensor Q descale into the softmax scale by passing `sm_scale=sm_scale / q_scale`.
`sm_scale` defaults to `1 / sqrt(head_dim)`, and `o_data_type` defaults to `torch.bfloat16`.

`kv_splits=None` lets the per-architecture cost model pick the split count; an explicit value overrides it.
`reduction="auto"` follows the cost model, `"none"` requires `kv_splits == 1`, `"kernel"` is split-K with an LSE merge, and `"atomic"` raises `NotImplementedError`.
On SM90 the reduction mode follows from the split count instead of being chosen: one split is `"none"` and more than one is `"kernel"`, and asking for the other pairing raises.

`max_kv_len` is the batch's maximum KV length.
Pass it explicitly; without it `plan()` reads `seq_lens.max()` and synchronizes the device.
`q_len_per_req` must be 1.

For a static batch, with the same page table and sequence lengths across steps, one planned wrapper serves every step.

### JIT warm-up

The first `plan()` for a static configuration traces and compiles the kernel, and a trace costs seconds.
Repeat calls with the same configuration reuse the in-process compile.
The static configuration covers `bits`, the head counts, the quantization granularity, the scale-pool precision, the reduction and combine mode, the output dtype, and any `kernel_kwargs`.
The on-disk kernel cache removes the trace cost for a configuration that any earlier process has compiled.
Call `plan()` during model loading, with a small dummy batch, to keep it off the first real decode step:

```python
# During model initialization, not on the critical path.
wrapper.plan(dummy_indptr, dummy_indices, dummy_seq_lens,
             num_qo_heads=32, num_kv_heads=8, head_dim=128, page_size=64,
             q_data_type=torch.float8_e4m3fn,   # torch.bfloat16 on SM90
             bits=2)
```

The cache lives under `$XDG_CACHE_HOME/cutedsl-decode/kernel-cache`, or `~/.cache/cutedsl-decode/kernel-cache` when `XDG_CACHE_HOME` is unset.
`CUTEDSL_DECODE_KERNEL_CACHE_DIR` moves it, and `CUTEDSL_DECODE_KERNEL_CACHE=0` disables it.

`plan()` exposes what it decided as read-only properties: `kv_splits`, `reduction`, `bits`, and `scale_pool_dtype`.

## run()

```python
out = wrapper.run(
    q,                        # [batch_size, num_qo_heads, head_dim]
    k_pool,                   # packed uint8 K code pool
    v_pool,                   # packed uint8 V code pool
    out=out_buffer,           # optional pre-allocated output
    sm_scale=None,            # per-call override of the plan-time scale
    o_scale=None,             # default 1.0
    enable_pdl=True,
    scale_pool=scale_pool,    # required
    pscale_lp=None,           # SM100: precompute with wrapper.pscale_lp()
    k_pool_order=None,        # SM100: "permuted" or "logical"
)
```

The output shape mirrors the input `q`: a 3-D `q` of shape `[batch_size, num_qo_heads, head_dim]` gives a 3-D output, and a 4-D `q` of shape `[batch_size, 1, num_qo_heads, head_dim]` gives a 4-D one.
When `out=` is passed, `run()` writes into it and returns it.

`k_pool` and `v_pool` are the packed code pools: `uint8`, with one contiguous block per (page, kv-head).
`scale_pool` is the per-(page, kv-head) affine block and is required.
With the default `("token", head_dim)` granularity for both tensors it is the interleaved f32 pool `[num_pages, num_kv_heads, page_size, 4] = (s_k, z_k, s_v, z_v)`.
With any other granularity it is the flat sectioned block `[num_pages, num_kv_heads, block_tot]` that `cutedsl_decode.quant_layout.quant_block_offsets` describes.
`run()` checks the pool's shape against the planned granularity and its dtype against the planned scale precision.

On SM90 both code pools are chunk-major.
Each (page, kv-head) block is `[per/16, 128, 16]` bytes, where `per` is one warpgroup thread's row of quantized values: 32 bytes at 4 bits and 16 at 2 bits at head_dim 128, and twice that at head_dim 256.
Chunk c of thread t holds bytes `[16c, 16c + 16)` of that thread's row.
The kernel names this order, and `run()` refuses a row-major `[num_pages, num_kv_heads, 128, per]` pool by its shape.
Nothing in the bytes says which order a pool is in, so a row-major pool reshaped to the chunk-major shape passes that check and decodes to wrong values at 4 bits.
At 2 bits at head_dim 128 a row is a single chunk, and the two orders are the same bytes.
SM90 pool construction is not packaged; the [README](../README.md) names the builder the kernel reads from.

## Arguments to keep out of the decode loop

**`out=`** avoids one allocation per call.
Pass a buffer of the planned output shape and dtype, and `run()` writes into it.

**`pscale_lp=`** is the SM100 launch-global P-scale exponent, `floor(log2(448 / max s_v))`.
When omitted, `run()` derives it from `scale_pool` with a `.max().item()` device synchronization on every call.
`wrapper.pscale_lp(scale_pool)` computes it once; recompute it only when the pool's maximum V scale changes.

**`k_pool_order=`** is mandatory on SM100 wherever the plan reads a permuted K pool, which is 4 bits at every configuration and 2 bits with the default E2M1 feed.
Declare `"permuted"` for a pool built with `pool.k_store_perm_sm100_f4(bits)`, and `"logical"` for a pool in plain `pack_rows_lut` order, which the plan accepts only where it reads logical order.
Nothing in the pool's bytes reveals the order, so an undeclared or contradictory declaration raises instead of being guessed.
The store order an external writer has to meet is in [`K_STORE_PERM.md`](K_STORE_PERM.md).
SM90 ignores this argument; its code-pool order is the chunk-major one described above, checked by shape.

**`sm_scale=` and `o_scale=`** are per-call constants.
Set them at `plan()` and pass them only when a request overrides the planned value.
`o_scale` is folded into the reduction epilogue, so it costs no extra kernel pass.

**`enable_pdl=`** enables programmatic dependent launch on SM100 and defaults to `True`.
SM90 ignores it.

## Routing with get_capabilities()

`get_capabilities()` describes the running device without planning anything.
Every key is present on an unsupported device, with empty or `False` values, so a caller can read any key without a `KeyError`.

```python
from cutedsl_decode import get_capabilities

caps = get_capabilities()
q_len_per_req = 1                       # this request's query length
routable = caps["paged"] and q_len_per_req == 1
```

`routable` is false on an unsupported device and for any query length other than one; send those requests to another backend.
`is_supported()` is the same check as `caps["paged"]`.

The keys that matter for routing are `arch` (`"sm90"`, `"sm100"`, or `"unsupported"`), `paged`, `dtypes` (the Q dtype), `head_dims` (a comma-joined string), `page_sizes` (`[64]`), `reductions` (`["none", "kernel"]`), `quantized_bits` (`[2, 4]`), and `kv_layouts` (`["NHD"]`).
`speculative`, `atomic_reduction`, and `skip_softmax` are `False` on both architectures.
`plan()` raises `NotImplementedError` for `q_len_per_req > 1` or `reduction="atomic"`, and `run()` raises `NotImplementedError` for `lse=` or `skip_softmax_threshold_scale_factor=`.

## Quantization granularity

`k_quant` and `v_quant` select an `(axis, group)` pair for K and V independently.
The default for both is `("token", head_dim)`, which is the shipped fast path.
K also accepts `("token", head_dim // 2)` and `("channel", G)`, and V accepts `("token", head_dim)` and `("channel", G)`.
`G` is one of 64, 128, or 256 and must be a multiple of the page size.
The scale pool layout follows the pair and is given by `cutedsl_decode.quant_layout.quant_block_offsets`.
Pass the granularity through `k_quant` and `v_quant`; naming it in `kernel_kwargs` raises.

At head_dim 256, which only SM90 accepts, the kernel builds the default pair with an f32 scale pool and nothing else.
`plan()` raises `ValueError` for every other pair or pool precision.

K token-64 needs a bfloat16 scale pool.
On SM90, build the pool with `cutedsl_decode._sm90.serving_cache.make_serving_cache_flex(..., D=head_dim, sdtype=torch.bfloat16)` and pass `kernel_kwargs={"sdtype": torch.bfloat16}` to `plan()`; a mismatch raises an error naming both arguments.
On SM100, K token-64 selects the bfloat16 pool by default, and `wrapper.scale_pool_dtype` reports the dtype the plan compiled for.

`kernel_kwargs=` is an advanced escape hatch.
On SM100 it forwards constructor knobs to the kernel, and any value set there overrides the per-bits tuning the plan promotes.
On SM90 it accepts only `fused_combine`, which selects the in-kernel split-K merge over the external combine, and `sdtype`.

## CUDA graph capture

`run()` is graph-capturable.
Both launch paths bind the kernel to torch's current stream, so no stream setup is needed around the call.

```python
static_out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
lp = wrapper.pscale_lp(scale_pool)          # SM100: compute before capture

wrapper.run(q, k_pool, v_pool, out=static_out, scale_pool=scale_pool,
            pscale_lp=lp, k_pool_order="permuted")
torch.cuda.synchronize()

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    wrapper.run(q, k_pool, v_pool, out=static_out, scale_pool=scale_pool,
                pscale_lp=lp, k_pool_order="permuted")

graph.replay()
```

Under capture on SM100 you must pass `pscale_lp=` explicitly, because deriving it reads `scale_pool.max().item()`, a device-to-host synchronization that is illegal during capture.
Pass a static `out=` so the graph writes into a buffer you own.
The `k_pool_order` value shown matches a plan that reads a permuted K pool, and SM90 ignores both `pscale_lp` and `k_pool_order`.

## Feature reference

| `plan()` parameter | Default | Notes |
|---|---|---|
| `bits=` | none | required: `2` or `4`, the stored bit width |
| `page_size=` | none | must be `64` |
| `head_dim=` | none | `128`; `256` on SM90 with token-wise K and V and an f32 scale pool |
| `q_data_type=` | `torch.bfloat16` | SM90: bfloat16 only. SM100: `torch.float8_e4m3fn` only |
| `o_data_type=` | `torch.bfloat16` | output dtype |
| `sm_scale=` | `1 / sqrt(head_dim)` | softmax scale |
| `kv_splits=` | `None` | `None`: the cost model picks; an integer overrides |
| `reduction=` | `"auto"` | `"none"` requires `kv_splits == 1`; `"kernel"` is split-K with an LSE merge; `"atomic"` raises |
| `q_len_per_req=` | `1` | must be `1` |
| `max_kv_len=` | `None` | `None` reads `seq_lens.max()` and synchronizes the device |
| `non_blocking=` | `True` | asynchronous copies of the plan-time integer buffers |
| `k_quant=`, `v_quant=` | `("token", head_dim)` | per-tensor `(axis, group)` granularity |
| `kernel_kwargs=` | `None` | advanced; see [Quantization granularity](#quantization-granularity) |

| `run()` parameter | Default | Notes |
|---|---|---|
| `q` | none | required; `[batch_size, num_qo_heads, head_dim]` or `[batch_size, 1, num_qo_heads, head_dim]` |
| `k_cache`, `v_cache` | none | required; packed `uint8` code pools |
| `scale_pool=` | `None` | required: the affine scale-and-offset block |
| `out=` | `None` | allocates per call when omitted |
| `sm_scale=` | the planned value | per-call override |
| `o_scale=` | `1.0` | folded into the reduction epilogue |
| `enable_pdl=` | `True` | SM100 only; SM90 ignores it |
| `pscale_lp=` | `None` | SM100: derived from `scale_pool` with a device synchronization unless passed |
| `k_pool_order=` | `None` | SM100: required wherever the plan reads a permuted K pool |
| `lse=`, `skip_softmax_threshold_scale_factor=` | `None` | not supported; passing either raises |
