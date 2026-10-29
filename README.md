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
from fosca import PagedDecodeWrapper, get_capabilities, pool

device = "cuda"
arch = get_capabilities()["arch"]                 # "sm90" or "sm100"
q_dtype = torch.float8_e4m3fn if arch == "sm100" else torch.bfloat16

batch, kv_heads, q_heads, head_dim, page_size = 8, 8, 32, 128, 64
pages_per_seq, bits = 64, 2

# A logical KV cache: [batch, pages, page_size, kv_heads, head_dim], bf16.
k_log = torch.randn(batch, pages_per_seq, page_size, kv_heads, head_dim,
                    device=device, dtype=torch.bfloat16)
v_log = torch.randn_like(k_log)

# Build the packed pools and the physical block table.
k_pool, v_pool, scale_pool, block_table, _aux = pool.build_pools_flex(
    k_log, v_log, bits, k_store_perm=pool.k_store_perm_sm100_f4(bits),
    v_store_perm=pool.v_store_perm_sm100_f4(bits))

# Page-table inputs.
indptr = torch.arange(batch + 1, device=device, dtype=torch.int32) * pages_per_seq
indices = block_table.reshape(-1).to(torch.int32)
seq_lens = torch.full((batch,), pages_per_seq * page_size, device=device,
                      dtype=torch.int32)

q = torch.randn(batch, q_heads, head_dim, device=device,
                dtype=torch.bfloat16).to(q_dtype)

w = PagedDecodeWrapper(torch.empty(256 * 1024 * 1024, dtype=torch.uint8,
                                   device=device))
w.plan(indptr, indices, seq_lens,
       num_qo_heads=q_heads, num_kv_heads=kv_heads, head_dim=head_dim,
       page_size=page_size, q_data_type=q_dtype, bits=bits)

out = w.run(q, k_pool, v_pool, scale_pool=scale_pool,
            k_pool_order="permuted")     # [batch, q_heads, head_dim]
```

`plan()` compiles the kernel for one static configuration and is called when the
page table or the sequence lengths change. `run()` is called every decode step
and is graph-capturable. Call `plan()` once at model load to pay the JIT cost
before the first request.

Two arguments in `run()` are worth hoisting out of the hot path: passing `out=`
avoids an allocation per call, and passing `pscale_lp=w.pscale_lp(scale_pool)`
avoids a device synchronisation per call.

## Support matrix

| | SM90 (Hopper) | SM100 (Blackwell) |
|---|---|---|
| Paged GQA decode | yes | yes |
| `bits` | 2, 4 | 2, 4 |
| Q dtype | bf16 | fp8 e4m3 |
| K/V | packed uint8 pools + scale pool | packed uint8 pools + scale pool |
| head_dim | 128; 256 with token-wise K and V and an f32 scale pool | 128 |
| page_size | 64 | 64 |
| reductions | none, kernel (split-K with LSE merge) | none, kernel (split-K with LSE merge) |
| GQA group size | up to 8 | up to 8 |
| quantized granularity | token 128/64, channel 64/128/256 | token 128/64, channel 64/128/256 |

`get_capabilities()` reports the same set for the running device. Requesting a
configuration outside it raises `ValueError` or `NotImplementedError` naming what
is supported.

## Building pools

Both architectures ship a builder that turns a bf16 KV cache into the packed
pools their kernel reads.

On Blackwell, `fosca.pool` packs the pools and also carries a reference
implementation of the same attention, so a kernel run can be checked against a
storage-matched oracle:

```python
from fosca import pool

k_pool, v_pool, scale_pool = pool.build_pools(
    k, v, bits,                                  # k, v: [pages, 64, kv_heads, dim]
    k_store_perm=pool.k_store_perm_sm100_f4(bits),
    v_store_perm=pool.v_store_perm_sm100_f4(bits))
```

On Hopper, `fosca.pool_sm90` quantizes and repacks into the chunk-major
pools the SM90 kernel reads, and lays the pages out as one global pool behind a
block table:

```python
from fosca import pool_sm90

cache = pool_sm90.build_pools_sm90(
    k, v, bits,                                  # k, v: [batch, kv_heads, kv_len, dim]
    seq_lens=seq_lens)                           # -> codes_k, codes_v, scales, block_table
```

The stored integer values are permuted in a way the kernel requires and the bytes
do not reveal. Ask the builder for the permutation rather than reproducing it:
`k_store_perm_sm100_f4` / `v_store_perm_sm100_f4` on Blackwell, and the
`order=` argument on Hopper, which defaults to the `chunk16` order the SM90
kernel reads. On Blackwell, `run()` refuses a K pool whose order the caller has
not declared through `k_pool_order`.

## Accuracy

`baselines`-style end-to-end scoring is not part of this release. What the
package ships is the reference implementation in `fosca.pool`, whose
`dequant_pools_flex` inverts a pool and `ref_paged_attention_fp8q_flex` scores
attention over the dequantized logical stream through the block table. Comparing
a kernel run against that reference measures the kernel's own error, because
both sides consume the same integer values, the same scale and offset, and the
same Q dtype.

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
