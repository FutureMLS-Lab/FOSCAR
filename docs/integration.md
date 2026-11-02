# Integrating FOSCA into a decode loop

This page covers the plan/run lifecycle, the page-table inputs, writing the
cache, what to keep out of the per-step path, and CUDA graph capture.
Installation, the support matrix and the cache layout are in the
[README](../README.md).

## Lifecycle

`plan(indptr, indices, seq_lens, num_qo_heads=, layout=)` fixes one batch: the
page table, the lengths, the split count and the compiled kernel.
`run(q, cache, out=)` decodes one layer's cache for that batch.

One plan serves every layer of the batch, because the layers share the page
table and differ only in the cache `run()` reads. Plan when the page table or
the lengths change, or once for a static batch, and build one wrapper per device
or worker.

`plan()` is transactional: it discards the previous plan before it starts, so a
wrapper whose `plan()` raised must be planned again before `run()`.

```python
import torch
from fosca import PagedDecodeWrapper, QuantKVLayout

device = "cuda"
batch, kv_heads, q_heads, head_dim, bits = 8, 8, 32, 128, 2
pages_per_request = 16                     # 16 pages of 64 tokens

layout = QuantKVLayout.create(bits=bits, head_dim=head_dim,
                              num_kv_heads=kv_heads, device=device)
cache = layout.allocate(batch * pages_per_request)

workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
wrapper = PagedDecodeWrapper(workspace)
```

The workspace holds the split-K partial results on Blackwell, and the wrapper
decodes on the workspace's device. Hopper allocates its own partial buffers and
leaves the workspace unused. `plan()` raises `RuntimeError` with the byte count
it needs when the workspace is too small.

The tensor form `run(q, k_cache, v_cache, scale_pool=, pscale_lp=,
k_pool_order=)` and the per-argument form of `plan(bits=, head_dim=,
num_kv_heads=, page_size=, k_quant=, v_quant=, kernel_kwargs=)` are retired and
raise `TypeError`; the layout carries those settings.

## Page-table inputs

Three int32 tensors describe the batch:

- `indptr` is `[batch_size + 1]`, the prefix sum of each request's page count.
- `indices` is `[num_pages_total]`, the physical page ids of every request,
  concatenated in logical order.
- `seq_lens` is `[batch_size]`, each request's KV length in tokens. `plan()`
  copies it to the device as int32, so int32 or int64 input works.

The page size is 64 tokens and is fixed: one code block and one scale block per
(page, KV head).

A block table of shape `[batch_size, pages_per_row]` converts as follows.

```python
def plan_inputs(block_table, seq_lens, page_size=64):
    """block_table: int32 [batch_size, pages_per_row] physical page ids."""
    device = block_table.device
    counts = ((seq_lens + page_size - 1) // page_size).to(torch.int32)
    indptr = torch.zeros(seq_lens.numel() + 1, dtype=torch.int32, device=device)
    indptr[1:] = torch.cumsum(counts, dim=0)
    indices = torch.cat([block_table[b, :n] for b, n in enumerate(counts.tolist())])
    return indptr, indices.to(torch.int32), seq_lens.to(torch.int32)
```

A scheduler that tracks a last-page length instead of a token count gets
`seq_lens` from `(indptr.diff() - 1) * 64 + last_page_len`.

```python
# One page per request, one token written at position 0 of each.
block_table = (torch.arange(batch, device=device, dtype=torch.int32)
               * pages_per_request).view(batch, 1)
seq_lens = torch.ones(batch, device=device, dtype=torch.int32)

indptr, indices, seq_lens = plan_inputs(block_table, seq_lens)

wrapper.plan(
    indptr,                  # int32 [batch_size + 1]
    indices,                 # int32 [num_pages_total]
    seq_lens,                # int32 [batch_size]
    num_qo_heads=q_heads,
    layout=layout,
    max_kv_len=1,            # the batch's longest context
)
```

`num_qo_heads` must be a positive multiple of `layout.num_kv_heads`; on Hopper a
ratio above eight must be a multiple of eight.

`max_kv_len` is the batch's longest context. Pass it: without it `plan()` reads
`seq_lens.max()` and waits for the device. Hopper's plan also waits while it
builds its block table.

`sm_scale` defaults to `1 / sqrt(head_dim)` and `o_scale` to 1.0. Blackwell's
query is fp8 and the kernel applies no query descale, so fold a per-tensor Q
descale into `sm_scale` or pass it to `run(q_descale=)`. Leave `kv_splits=None`
and `reduction="auto"` at their defaults; Hopper derives the reduction mode from
the split count, where one split is `"none"` and more than one is `"kernel"`.
`q_len_per_req` must be 1.

Then run, once per layer:

```python
q_dtype = torch.float8_e4m3fn if layout.arch == "sm100" else torch.bfloat16
q = torch.randn(batch, q_heads, head_dim, dtype=torch.bfloat16,
                device=device).to(q_dtype)
out = torch.empty(batch, q_heads, head_dim, dtype=torch.bfloat16, device=device)

wrapper.run(q, cache, out=out)
```

`q` is `[batch_size, num_qo_heads, head_dim]` or
`[batch_size, 1, num_qo_heads, head_dim]`, C-contiguous, 16-byte aligned and in
the planned dtype. A query sliced out of a fused projection is not contiguous,
so pass `q.contiguous()`. The output matches the shape of `q`. `run()` also
takes `lse=` and `q_descale=`, both fp32 `[batch_size, num_qo_heads]`: the
natural-log log-sum-exp, and the query scale on Blackwell. Each compiles a
variant on first use.

## Writing the cache

`QuantKVLayout.create()` resolves the stored bit width, head dimension, KV-head
count, quantization granularity, scale dtype and pool byte orders against the
device's decode kernel. Nothing else declares them, and `run()` refuses a cache
of another layout, even one whose pools have the same shapes.

`layout.allocate(num_pages)` allocates a cache. `layout.bind(buffer, num_pages)`
wraps a `uint8` buffer the engine owns: C-contiguous, 16-byte aligned at its
start, at least `num_pages * layout.bytes_per_page()` bytes. An engine's
`[num_blocks, bytes_per_page]` tensor works. The pools inside it are
region-major, so row `i` is not page `i`; go through the cache rather than
copying blocks by index.

`cache.append(k, v, slot_mapping)` quantizes bf16 tokens on the device, one
launch for K and V. `k` and `v` are `[tokens, num_kv_heads, head_dim]` bf16 with
the head dimension contiguous and every row 16-byte aligned. `slot_mapping` is
int64 `[tokens]`: token `t` goes to page `slot // 64`, position `slot % 64`, and
a negative slot is skipped.

```python
k = torch.randn(batch, kv_heads, head_dim, dtype=torch.bfloat16, device=device)
v = torch.randn_like(k)
slot_mapping = (torch.arange(batch, device=device, dtype=torch.int64)
                * (pages_per_request * 64))
cache.append(k, v, slot_mapping)
```

`cache.append_quantized(k_codes, k_scale, k_zero, v_codes, v_scale, v_zero,
slot_mapping)` writes tokens the engine already quantized: uint8 integer values
in `[tokens, num_kv_heads, head_dim]` with a float32 scale and zero point per
token, or per half-head group when K is token-wise over `head_dim // 2`. A
stored value is `s * q + z` per channel.

`cache.dequantize(k_out, v_out, page_indptr, page_indices, seq_lens, out_indptr)`
reads rows back as bf16, one launch for both tensors, from the page table in the
form `plan()` takes and an `out_indptr` naming each request's first output row.

Three rules the engine keeps:

- Every use of a page begins with a write at position 0. `append()` then clears
  that page's other per-token scale records, so no stale scale is read past the
  request's length.
- No decode of a layer runs concurrently with an append to that layer, on any
  stream. Stream order on one stream gives this; across streams the engine
  orders them.
- Bytes copied into the cache outside `append()` (swap-in, offload, transfer
  between engines) move the Blackwell page maxima behind the count table the
  decode reads. After such a copy, call `cache.recount_v_max()` before the next
  decode of that layer.

Call `bind()` once per buffer, before any append, and never inside a capture.
Duplicate valid slots in one append leave the whole page unspecified on Hopper
and the duplicated slot unspecified on Blackwell, so route padding writes to a
page no request uses. `append()`, `append_quantized()` and `dequantize()` handle
token-wise layouts only and raise `NotImplementedError` for a channel-wise K or
V.

## What belongs outside the decode loop

- **`out=`** avoids one allocation per call; `run()` writes into it and returns
  it.
- **`plan()`** runs once per page-table or length change, not once per layer.
- **Kernel compilation.** The first `plan()` per configuration compiles the
  decode and takes seconds; the on-disk kernel cache removes that for a
  configuration an earlier process compiled. The first `run()` with `lse=` or
  `q_descale=` compiles a variant. Warm both at model load, before serving and
  before any capture.
- **`sm_scale` and `o_scale`.** Set them at `plan()` and pass them per call only
  when a request overrides the planned value.
- **The cache writes.** `append()` is its own launch, ordered before the decode
  on the same stream.

`run()` allocates nothing when `out=` is passed and reads no device value. On
Blackwell with token-wise V, the P-scale exponent comes from the cache's own
page-maximum count table, so `run()` needs no host-derived value and no per-call
synchronization.

## CUDA graph capture

Graph mode is a separate implementation, selected at construction:
`PagedDecodeWrapper(workspace, use_cuda_graph=True)`. One such wrapper serves one
captured batch size, a bucket, for its whole life and plans the step's split
count on the host, because a captured graph freezes every launch argument.

`graph_workspace_bytes(max_batch, num_qo_heads, layout, max_kv_len=)` returns the
bytes that the buckets of every batch size up to `max_batch` share. Buckets share
it because no two of their graphs run at the same time; an engine that overlaps
two batches gives each stream its own workspace.

```python
from fosca import graph_workspace_bytes

max_kv_len = 1024
page_table = torch.zeros(batch, pages_per_request, dtype=torch.int32, device=device)
page_table[:, 0] = (torch.arange(batch, device=device, dtype=torch.int32)
                    * pages_per_request)
engine_seq_lens = torch.ones(batch, dtype=torch.int32, device=device)

workspace = torch.empty(
    graph_workspace_bytes(batch, q_heads, layout, max_kv_len=max_kv_len,
                          device=device),
    dtype=torch.uint8, device=device)
gw = PagedDecodeWrapper(workspace, use_cuda_graph=True)
gw.plan(page_table=page_table, seq_lens=engine_seq_lens, num_qo_heads=q_heads,
        layout=layout, max_kv_len=max_kv_len)
```

`page_table` is int32 `[batch_size, max_pages]`, row-major, on the wrapper's
device, and `seq_lens` is int32 `[batch_size]`. Both are the engine's own
buffers, kept in step with the lengths passed to `update()`. The plan records
their addresses and reads no value from them. `max_kv_len` is the bucket's
longest context, and every `update()` length stays within it.

Capture once per layer:

```python
out = torch.empty(batch, q_heads, head_dim, dtype=torch.bfloat16, device=device)
gw.run(q, cache, out=out)                 # warm-up, outside the graph
torch.cuda.synchronize()

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    gw.run(q, cache, out=out)
```

Then once per step, on the host and outside the graph:

```python
lengths = [1] * batch
engine_seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
gw.update(lengths)
graph.replay()
```

`update(lengths)` takes one integer per request of the bucket, as a list, a
numpy array or a CPU tensor, each in `[1, max_kv_len]`, and copies the step's
work list to the device. It runs on the stream the graph replays on, the same
stream at every call, and never waits for the device as a whole. One update
serves every layer of the step, so update once and replay every layer's graph.

Graph mode requires `out=` on every `run()`, and the call must pass exactly the
`lse` and `q_descale` variants the plan declared, because those are the builds it
compiled. A second `plan()` on the same wrapper raises.

Blackwell graphs take head_dim 128 and GQA ratios 3 to 8; Hopper takes ratios 1
to 8. A ratio outside that range runs in eager mode only.
