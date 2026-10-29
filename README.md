# fosca

Quantized-KV paged GQA decode kernels for Hopper (SM90) and Blackwell (SM100),
written in CuTe DSL and exposed through one architecture-agnostic wrapper.

K and V live in the cache as 2-bit or 4-bit packed integer values plus a separate
scale-and-offset pool. The kernels dequantize on chip and never materialize a
16-bit KV tile, so a decode step reads a fraction of the bytes it would read from
a bf16 cache. At 2 bits the cache costs about 2.5 bits per value including the
scale and offset, which is 6.4 times smaller than bf16 and 1.8 times smaller than
nvfp4.

## Install

From a source checkout:

```sh
git clone https://github.com/donglinz/fosca.git
cd fosca
pip install -e ".[cu13]"    # CUDA 13: Blackwell, and Hopper
pip install -e ".[cu12]"    # CUDA 12: Hopper only
```

The same extras work with `pip install "fosca[cu13]"` once the package
is published to PyPI.

The package contains no compiled extensions. Everything is JIT-compiled by
`nvidia-cutlass-dsl` on first use, and the result is memoized in process and
cached on disk across processes. Bring your own `torch>=2.4`.

Blackwell requires the `cu13` extra. The DSL's CUDA 12 backend miscompiles the
SM100 kernel, and the package refuses to compile it under that backend rather
than producing a wrong result.

## Quickstart

```python
import torch
from fosca import PagedDecodeWrapper, QuantKVLayout, get_capabilities

device = "cuda"
arch = get_capabilities()["arch"]                 # "sm90" or "sm100"
q_dtype = torch.float8_e4m3fn if arch == "sm100" else torch.bfloat16

batch, kv_heads, q_heads, head_dim, page_size = 8, 8, 32, 128, 64
pages_per_seq, bits = 64, 2

# One layer's cache layout, resolved against the reader that will decode it.
layout = QuantKVLayout.create(bits=bits, head_dim=head_dim,
                              num_kv_heads=kv_heads, device=device)
cache = layout.allocate(batch * pages_per_seq)

# Write a token per request into the cache. k/v are bf16 [tokens, kv_heads, head_dim]
# and slot_mapping is int64: token t goes to page slot // 64, position slot % 64.
k = torch.randn(batch, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
v = torch.randn_like(k)
slot_mapping = (torch.arange(batch, device=device, dtype=torch.int64)
                * (pages_per_seq * page_size))
cache.append(k, v, slot_mapping)

# Page-table inputs.
indptr = torch.arange(batch + 1, device=device, dtype=torch.int32) * pages_per_seq
indices = torch.arange(batch * pages_per_seq, device=device, dtype=torch.int32)
seq_lens = torch.full((batch,), pages_per_seq * page_size, device=device,
                      dtype=torch.int32)

q = torch.randn(batch, q_heads, head_dim, device=device,
                dtype=torch.bfloat16).to(q_dtype)

w = PagedDecodeWrapper(torch.empty(256 * 1024 * 1024, dtype=torch.uint8,
                                   device=device))
w.plan(indptr, indices, seq_lens, num_qo_heads=q_heads, layout=layout)

out = torch.empty(batch, q_heads, head_dim, device=device, dtype=torch.bfloat16)
w.run(q, cache, out=out)                    # [batch, q_heads, head_dim]
```

`plan()` compiles the kernel for one static configuration and is called when the
page table or the sequence lengths change. `run()` is called every decode step
and is graph-capturable. Call `plan()` once at model load to pay the JIT cost
before the first request.

The cache object carries the pool orders, the scale dtype and the granularity, so
`run()` takes no declaration of them: a cache built for a different layout is
refused.

### Quantization granularity

The quickstart asks for token-wise K and V: one scale and offset per token, over
the whole head dimension, which is the default. Granularity is chosen per tensor
by an `(axis, group)` pair on the layout, and the cache is written by that layout.

```python
layout = QuantKVLayout.create(bits=2, head_dim=128, num_kv_heads=8,
                              k_quant=("channel", 128),   # per channel, per 128 tokens
                              v_quant=("token", 128),     # per token over the head dim
                              device=device)
```

Accepted pairs, at `head_dim` 128:

| | token-wise | channel-wise |
|---|---|---|
| K | group `D` (default) or `D/2` | token group 64, 128 or 256 |
| V | group `D` only | token group 64, 128 or 256 |

`QuantKVCache.append` writes token-wise layouts. A channel-wise layout is written
one complete quantization group at a time, through `append_quantized`.

## Support matrix

| | SM90 (Hopper) | SM100 (Blackwell) |
|---|---|---|
| Paged GQA decode | yes | yes |
| `bits` | 2, 4 | 2, 4 |
| Q dtype | bf16 | fp8 e4m3 |
| K/V | packed uint8 pools + scale pool | packed uint8 pools + scale pool |
| head_dim | 128, 256 | 128, 256 |
| page_size | 64 | 64 |
| reductions | none, kernel (split-K with LSE merge) | none, kernel (split-K with LSE merge) |
| KV heads | 1 to 128 at 128 query heads | 1 to 128 at 128 query heads |
| quantized granularity | token 128/64, channel 64/128/256 | token 128/64, channel 64/128/256 |

`get_capabilities()` reports the same set for the running device. Requesting a
configuration outside it raises `ValueError` or `NotImplementedError` naming what
is supported.

## Writing the cache

`QuantKVCache.append` quantizes and writes tokens on the device, one launch for K
and V. `append_quantized` takes integer values an engine has already quantized.
`dequantize` reads rows back as bf16.

Every use of a page must begin with a write at position 0, and no decode of the
layer may run concurrently with an append on another stream.

## Benchmarks

Decode latency in microseconds. Each figure plots the speedup over FlashInfer
bf16 at the same shape, so the dashed line at 1.00 is bf16.

### B200 (SM100)

![Decode speedup over FlashInfer bf16 on B200, at batch 1 and 8](docs/decode-latency-b200.png)

| Batch | Context | int2 | int4 | FlashInfer bf16 | TRT-LLM FP8 | TRT-LLM NVFP4 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32K | **16.9** | 19.4 | 35.4 | 23.6 | 18.5 |
| 1 | 128K | **37.2** | 45.8 | 98.3 | 59.5 | 41.6 |
| 1 | 256K | **63.5** | 77.6 | 171.1 | 101.5 | 69.7 |
| 1 | 512K | **116.0** | 136.3 | 313.6 | 176.3 | 122.5 |
| 8 | 32K | **63.3** | 75.9 | 168.0 | 96.1 | 69.2 |
| 8 | 128K | **220.6** | 256.5 | 595.2 | 309.3 | 246.3 |
| 8 | 256K | **400.5** | 461.9 | 1170.4 | 594.2 | 479.3 |
| 8 | 512K | **760.3** | 874.0 | 2323.3 | 1166.7 | 947.7 |

### H100 (SM90)

TensorRT-LLM's `trtllm-gen` decode is Blackwell only, so this table has no
TensorRT-LLM column.

![Decode speedup over FlashInfer bf16 on H100, at batch 1 and 8](docs/decode-latency-h100.png)

| Batch | Context | int2 | int4 | FlashInfer bf16 |
|---:|---:|---:|---:|---:|
| 1 | 32K | **25.5** | 32.9 | 64.4 |
| 1 | 128K | **72.8** | 79.4 | 199.9 |
| 1 | 256K | **127.9** | 135.3 | 378.4 |
| 1 | 512K | **242.5** | 271.5 | 735.0 |
| 8 | 32K | **94.6** | 117.2 | 385.1 |
| 8 | 128K | **347.1** | 401.7 | 1458.2 |
| 8 | 256K | **676.2** | 926.6 | 2890.4 |
| 8 | 512K | **1330.6** | 1830.2 | 5758.2 |

### Conditions

One query token per request, 32 query heads over 8 KV heads, `head_dim` 128,
page size 64, paged cache. Q is fp8 on the B200 tables and bf16 on the H100 one;
FlashInfer is bf16 everywhere. Each row's five (or four) columns ran on the same
card in one session, through one timing function. The number is the CUDA kernel
span from CUPTI over a cold L2, the median of the fully observed iterations. It
excludes host dispatch and the paged-attention wrapper, so it is not an
end-to-end serving latency.

The columns for this package were measured by driving `PagedDecodeWrapper` from
this release; `benchmarks/decode_latency.csv` holds the values and
`benchmarks/plot_decode.py` renders both figures. The competitor columns were
measured by the project's benchmark harness.

## Testing

```sh
pytest -q
```

Modules that need a device skip when there is none.

## License

Apache-2.0. This package derives from FlashInfer (Apache-2.0) and the NVIDIA
CUTLASS Hopper FMHA example (BSD-3-Clause), and redistributes subsets of QuACK
and flash-attention (both BSD-3-Clause). See [NOTICE](NOTICE).
