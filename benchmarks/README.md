# Benchmarks

Decode latency for this package and for the two kernels it is compared against,
measured by one harness on one GPU in one process.

```sh
pip install -e ".[bench]"                       # adds FlashInfer 0.6.13
python benchmarks/run_bench.py                  # every shape, every kernel
python benchmarks/run_bench.py --shape inkling
python benchmarks/run_bench.py --kernel ours_int2 --batch 1 --context 32768
python benchmarks/run_bench.py --out results.csv
```

`bench.yaml` sets the shapes and the batch and context grid. A shape is the
attention geometry one GPU decodes: a model's head counts divided by its
tensor-parallel width.

## What a number is

One decode step's GPU kernel time. `timer.measure` runs five warm-up calls and
then five timed calls, each preceded by a 1 GiB write that reallocates every L2
set, and captures the loop with the CUDA profiler. Each kernel is charged to the
call whose launch produced it, so the figure is the sum of one call's kernels and
nothing else: host dispatch, stream idle and the L2 flush are outside it. The
median is over the calls the profiler reported whole.

The flush is excluded by exact name, which is why it is compiled here through
NVRTC rather than issued as a `torch` fill.

## Columns

| Name | What runs |
|---|---|
| `ours_int2`, `ours_int4` | `PagedDecodeWrapper` over a token-wise quantized cache |
| `flashinfer_bf16` | FlashInfer's paged bf16 decode, at the faster backend each GPU has |
| `trtllm_fp8`, `trtllm_nvfp4` | TensorRT-LLM's Blackwell decode through FlashInfer |

Every column of a cell decodes the same K, V and page table. Each converts them
to the cache format its own kernel reads, before timing starts.

The three FlashInfer columns are Blackwell-only for the TensorRT-LLM kernels; on
Hopper the harness reports them as skipped. The extra pins FlashInfer 0.6.13,
the release the published columns were measured with: a later one moves its
decode kernels, and its numbers would not be the same comparison.

## Output

One CSV row per cell: GPU, architecture, shape, batch, context, the shape's head
counts and head dimension, kernel, microseconds, and the kernels the call
launched. `decode_latency.csv` is the published table; `plot_decode.py` renders
the figures in `docs/` from it.

## Adding a shape

Add it under `shapes:` in `bench.yaml` with the per-GPU head counts, the head
dimension and the tensor-parallel width, then run that shape by name. The kernel
refuses a head dimension or a query-to-KV head ratio it does not build for, and
the harness reports that shape as skipped rather than failing the run.
