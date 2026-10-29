# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - unreleased

### Added

- `QuantKVLayout` and `QuantKVCache`: a layer's cache layout, resolved against
  the reader that will decode it, and the cache object itself. `append` quantizes
  and writes tokens on the device, `append_quantized` takes values an engine has
  already quantized, and `dequantize` reads rows back as bf16.
- `head_dim` 256 on Blackwell, eager decode.
- KV heads from 1 to 128 at 128 query heads.
- CUDA graph mode, with `graph_workspace_bytes()` and a plan per captured batch.

### Changed

- `plan()` takes a `layout` and `run()` takes the cache. The `k_cache`, `v_cache`,
  `scale_pool`, `pscale_lp` and `k_pool_order` arguments are retired and raise:
  the orders, the scale dtype and the granularity all come from the layout.
- The package is `fosca`, not `cutedsl-decode`. The kernel-cache directory and its
  two environment variables moved with it (`FOSCA_KERNEL_CACHE`,
  `FOSCA_KERNEL_CACHE_DIR`); an existing cache directory is not migrated.
- `numpy` is a declared dependency.

## [0.1.0] - unreleased

First public release.

### Added

- Quantized-KV paged GQA decode for Hopper (SM90) and Blackwell (SM100) behind
  one arch-agnostic `PagedDecodeWrapper`.
- int2 and int4 packed KV pools with per-tensor quantization granularity, chosen
  independently for K and V.
- Split-KV planning for both architectures, with a cost model per head
  dimension.
- A persistent on-disk kernel cache.
- A CPU test suite that needs no GPU, and GPU validation scripts.

### Notes

- The package is licensed under Apache-2.0. It derives from FlashInfer
  (Apache-2.0) and NVIDIA CUTLASS (BSD-3-Clause); see `NOTICE`.
- Only the quantized decode path exists. There is no dense bf16/fp16/fp8 KV
  mode.
- Published latency comparisons are not reproduced by this repository; the
  benchmark harness used for them is not part of this release.
