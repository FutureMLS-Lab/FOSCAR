# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
