# cutedsl-decode

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

The same extras work with `pip install "cutedsl-decode[cu13]"` once the package
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
from cutedsl_decode import PagedDecodeWrapper, get_capabilities, pool

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

`cutedsl_decode.pool` turns a bf16 logical cache into the packed pools the
kernel reads, and provides a reference implementation of the same attention so a
kernel run can be checked against a storage-matched oracle.

The stored integer values are permuted in a way the kernel requires and the bytes
do not reveal, so ask the pool builder for the permutation
(`k_store_perm_sm100_f4`, `v_store_perm_sm100_f4`) rather than reproducing it.
`run()` refuses a K pool whose order the caller has not declared through
`k_pool_order`.

Pool construction for SM90 is not yet packaged. That architecture's kernel takes
pre-quantized codes through `cutedsl_decode._sm90.serving_cache`.

## Accuracy

`baselines`-style end-to-end scoring is not part of this release. What the
package ships is the reference implementation in `cutedsl_decode.pool`, whose
`dequant_pools_flex` inverts a pool and `ref_paged_attention_fp8q_flex` scores
attention over the dequantized logical stream through the block table. Comparing
a kernel run against that reference measures the kernel's own error, because
both sides consume the same integer values, the same scale and offset, and the
same Q dtype.

## Benchmarks

Decode latency on one NVIDIA B200, against FlashInfer and TensorRT-LLM's
`trtllm-gen` decode kernels. Lower is better.

![Decode latency against batch size at KV lengths 8K and 32K](docs/decode-latency.png)

| Batch | KV len | int2 | int4 | FlashInfer bf16 | TRT-LLM FP8 | TRT-LLM NVFP4 | int2 vs NVFP4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8K | **11.5** | 12.4 | 16.1 | 12.5 | 13.0 | 1.13x |
| 1 | 32K | **16.6** | 19.7 | 35.3 | 22.9 | 18.5 | 1.12x |
| 8 | 8K | **22.9** | 28.2 | 55.5 | 31.6 | 22.9 | 1.00x |
| 8 | 32K | **63.0** | 75.7 | 166.9 | 95.5 | 69.2 | 1.10x |
| 32 | 8K | **65.1** | 75.2 | 170.2 | 95.5 | 72.2 | 1.11x |
| 32 | 32K | **214.1** | 257.0 | 602.4 | 317.9 | 250.7 | 1.17x |
| 128 | 8K | **214.6** | 248.0 | 600.4 | 314.8 | 223.3 | 1.04x |
| 128 | 32K | **765.9** | 877.7 | 2323.6 | 1189.8 | 832.7 | 1.09x |

What the columns store per cached value, including the scale and offset:

| | int2 | int4 | NVFP4 | FP8 | bf16 |
|---|---:|---:|---:|---:|---:|
| bits per value | 2.5 | 4.5 | 4.5 | 8 | 16 |

Two things follow from that table and are worth stating plainly.

The 2-bit kernel reads about 1.8 times fewer bytes than NVFP4 and finishes
between 1.00 and 1.17 times sooner. It is not bandwidth-bound at these shapes:
the dequantization and the on-chip work absorb most of the traffic saving.

At 4 bits, where this package and NVFP4 store the same number of bits per value,
the two are within a few percent of each other, and TRT-LLM is faster at seven of
the eight shapes. The 2-bit setting is where this kernel wins.

Measurement conditions: one query token per request, 32 query heads over 8 KV
heads, `head_dim` 128, page size 64, paged cache, fp8 Q for this package and for
TRT-LLM, bf16 for FlashInfer. All five columns were measured on the same B200
card on 2026-10-28, in one session, through one timing function. The reported
number is the CUDA kernel span from CUPTI, over a cold L2, as the median of the
fully observed iterations. It excludes host dispatch and the paged-attention
wrapper around the kernel, so it is not an end-to-end serving latency.

Which code produced each column:

- **int2 and int4** were measured by driving `PagedDecodeWrapper` from this
  release. `benchmarks/decode_latency.csv` holds the values and
  `benchmarks/plot_decode.py` renders the figure above.
- **FlashInfer and TRT-LLM** were measured by the project's benchmark harness,
  which is not part of this release. Those three columns cannot be regenerated
  from this repository.

## Testing

```sh
pytest -q            # CPU suite: no GPU and no CUDA toolkit required
pytest -q -m gpu     # GPU tests, on an SM90 or SM100 device
```

A test module whose name ends in `_cpu` must pass with no GPU. The CPU suite
runs in about ten seconds.

## Limitations

- Only the quantized path exists. There is no dense bf16, fp16 or fp8 KV mode.
- `head_dim` 256 on SM90 requires token-wise K and V with an f32 scale pool.
- The benchmark harness that produced the competitor columns in
  [Benchmarks](#benchmarks) is not part of this release.
- Pool construction for SM90 from a bf16 cache is not packaged; that
  architecture's kernel takes pre-quantized codes through
  `cutedsl_decode._sm90.serving_cache`.

## License

Apache-2.0. This package derives from FlashInfer (Apache-2.0) and the NVIDIA
CUTLASS Hopper FMHA example (BSD-3-Clause), and redistributes subsets of QuACK
and flash-attention (both BSD-3-Clause). See [NOTICE](NOTICE).
