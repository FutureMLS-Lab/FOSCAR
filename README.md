# cutedsl-decode

Standalone CuTe DSL **quantized-KV** GQA paged-decode kernels for **Hopper (SM90)** and
**Blackwell (SM100)**, exposed through a single arch-agnostic `PagedDecodeWrapper`.

K and V live in the cache as int2/int4 packed codes; the kernels dequantize on chip and
never materialize a 16-bit KV tile. Internal arch detection and per-arch feature gating are
hidden behind the wrapper: pass your page table, sequence lengths, Q, code pools and scale
pool; get back the attention output. The kernels are pure-Python, JIT-compiled via
`nvidia-cutlass-dsl` (MLIR → CUDA; no C++/ninja build step). Compiled modules are memoized so
the cost is paid once per unique static configuration.

**SM100 INT4 cache compatibility:** V now packs tokens within channel rows,
matching the INT2 orientation. Rebuild old INT4 V pools: the shape is unchanged,
but the byte order is not. See [the pool contract](docs/K_STORE_PERM.md#shared-reduction-axis-packing-for-k-and-v-2026-09-24).
SM100 INT2 pools are unchanged.

**SM90 cache compatibility:** both Hopper code pools are now stored chunk-major,
`[num_pages, num_kv_heads, per/16, 128, 16]`, and a row-major
`[num_pages, num_kv_heads, 128, per]` pool is refused by shape. Convert old pools with
`serving_cache.code_rows_to_order`. Do not `view` or `reshape` them to the new shape:
that passes the shape check and decodes wrong at INT4, where the two orders place the
same bytes differently. See [the pool contract](docs/integration.md#run).

## Why this package

Decode is HBM-bound, and the KV cache is what it reads. At int2 the cache costs 2.5 bits per
value including scales — **6.4x smaller than bf16** and 1.8x smaller than nvfp4 — so the
kernel finishes in a fraction of the bytes.

Accuracy is a knob rather than a fixed cost. `k_quant` / `v_quant` set per-tensor
granularity independently for K and V — token-wise, or channel-wise with token groups of
64/128/256. Channel-wise K cuts end-to-end attention error on outlier-channel data, and
most of that is available at the cheapest setting.

## Install

```bash
pip install "cutedsl-decode[cu12]"   # CUDA 12.x: Hopper only
pip install "cutedsl-decode[cu13]"   # CUDA 13.x: Blackwell, and Hopper on a CUDA 13 driver
```

Blackwell needs the CUDA 13 extra: the DSL's CUDA 12 backend miscompiles the SM100 kernel, so
compiling it under that backend raises.

Not yet published to PyPI — until then, install from a source checkout instead:

```bash
git clone https://github.com/donglinz/cutedsl-decode.git
cd cutedsl-decode
pip install -e ".[cu12]"      # or: uv pip install -e ".[cu12]"
```

Bring your own `torch` matching your CUDA installation (requires `torch>=2.4`). The package
itself contains no compiled extensions; everything is JIT-compiled by `nvidia-cutlass-dsl`
at first use.

Both extras pin `nvidia-cutlass-dsl` 4.8.0, the version SGLang pins, and `baselines/bench.sh`
measures our columns under that pin, in an interpreter of their own (`ours` in
`baselines/venvs.py`). Earlier published measurements were taken under 4.5.2. DSL 4.6 to 4.8
silently miscompiled the SM100 kernel's merge of the token-wise V row sums whenever one split
covered more tiles than the output ring, which the single-split path (`reduction="none"`) does
at most shapes, so the output was wrong without an error. The kernel now reads those sums in
a form these versions compile correctly. `baselines/adapters/ours.py` still refuses to measure
the SM100 kernel under 4.6 or 4.7, because neither has been through the gate set.

## Quickstart

```python
import torch
from cutedsl_decode import PagedDecodeWrapper, get_capabilities

is_sm100 = get_capabilities()["arch"] == "sm100"   # "sm90" on Hopper
q_dtype = torch.float8_e4m3fn if is_sm100 else torch.bfloat16

# --- Problem setup ---
batch_size = 8
num_pages_per_seq = 64    # seq_len / page_size
page_size = 64            # quantized mode requires 64
total_pages = batch_size * num_pages_per_seq
num_qo_heads, num_kv_heads, head_dim = 32, 8, 128   # GQA group 4 (must be <= 8)

bits = 2                  # 2 or 4; REQUIRED
device = "cuda"

# Q: [batch_size, num_qo_heads, head_dim]. fp8 e4m3 on SM100, bf16 on SM90.
q = torch.randn(batch_size, num_qo_heads, head_dim,
                device=device, dtype=torch.bfloat16).to(q_dtype)

# K/V are PACKED CODE POOLS (uint8), not dtype-typed caches, plus a scale pool.
# Build them with `cutedsl_decode._sm90.serving_cache` (the pool contract) or see
# `scripts/probes/repack_sm100.py` for the reference repack. On SM100 the default
# kernel reads a K pool whose codes were permuted by
# `repack_sm100.k_store_perm_sm100_f4` before packing, at both widths
# (docs/K_STORE_PERM.md). On SM90 both pools are chunk-major, in the order
# `IntPingPongDecodeSm90.code_pool_order` names (docs/integration.md).
k_cache, v_cache, scale_pool = build_packed_pools(...)   # uint8 / uint8 / f32

# Page table tensors (int32)
indptr = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * num_pages_per_seq
indices = torch.arange(total_pages, device=device, dtype=torch.int32)   # identity mapping
seq_lens = torch.full((batch_size,), num_pages_per_seq * page_size, device=device, dtype=torch.int32)

# --- Wrapper lifecycle ---
workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
w = PagedDecodeWrapper(workspace)

# plan() compiles the kernel for this static config (once, then cached)
w.plan(
    indptr,
    indices,
    seq_lens,
    num_qo_heads=num_qo_heads,
    num_kv_heads=num_kv_heads,
    head_dim=head_dim,
    page_size=page_size,
    q_data_type=q_dtype,               # SM100: fp8 only. SM90: torch.bfloat16
    bits=bits,                         # REQUIRED (2 or 4)
    sm_scale=head_dim ** -0.5,         # default; can be omitted
)

# run() dispatches the compiled kernel; output shape matches q.
# k_pool_order declares the K order the pool carries, which nothing in its bytes
# reveals: an SM100 plan that reads a permuted pool refuses to run without it, and
# SM90 does not check it.
out = w.run(q, k_cache, v_cache, scale_pool=scale_pool,
            k_pool_order="permuted" if is_sm100 else None)   # [batch_size, num_qo_heads, head_dim]
```

**Key shapes:**
- `q`: `[batch_size, num_qo_heads, head_dim]` — one token per sequence.
- `k_cache`, `v_cache`: the packed code pools (`uint8`), NHD-contiguous, `page_size=64`. On SM90 each (page, kv-head) block is chunk-major, `[per/16, 128, 16]`.
- `scale_pool`: the affine (scale, zero) block; layout from `cutedsl_decode/quant_layout.quant_block_offsets`.
- `seq_lens`: per-sequence KV token counts, `int32`.
- `indptr`: length `batch_size + 1`; `indptr[i]` is the start index into `indices` for sequence `i`.

Call `plan()` whenever the page table or sequence lengths change. Call `run()` every decode
step. To warm the JIT at startup (avoiding first-step latency), call `plan()` once with a dummy
batch during model loading.

## Capabilities and support matrix

```python
from cutedsl_decode import get_capabilities, is_supported

print(is_supported())       # True on SM90 or SM100, False otherwise
print(get_capabilities())
# {'arch': 'sm100', 'paged': True, 'dtypes': ['float8_e4m3fn'],   # SM90: ['bfloat16']
#  'head_dims': '128',                                            # SM90: '128,256'
#  'page_sizes': [64],
#  'reductions': ['none', 'kernel'],
#  'speculative': False, 'atomic_reduction': False, 'skip_softmax': False,
#  'kv_layouts': ['NHD'],
#  'quantized_bits': [2, 4], 'quantized_page_sizes': [64],
#  'quantized_validated': True}    # True on both arches
```

Every value describes the quantized path, which is the only path left. `dtypes` is the
**Q** dtype; K/V are always packed `uint8` code pools.

| Capability | SM90 (Hopper) | SM100 (Blackwell) |
|---|---|---|
| Paged-KV GQA decode | yes | yes |
| `bits` | 2, 4 | 2, 4 |
| GQA group size | up to 8 | up to 8 |
| Q dtype | bf16 | fp8 e4m3 |
| K/V | packed uint8 code pools + scale pool | packed uint8 code pools + scale pool |
| head_dim | 128; 256 with token-wise K and V and an f32 scale pool only | 128 |
| page_size | 64 | 64 |
| reductions | none, kernel (split-K LSE-merge) | none, kernel (split-K LSE-merge) |
| quant granularity (`k_quant`/`v_quant`) | token 128/64, channel 64/128/256; token 256 only at head_dim 256 | token 128/64, channel 64/128/256 |
| o_scale | yes | yes |
| LSE output | no | no |

Requesting something outside this matrix raises `ValueError` or `NotImplementedError` with an
actionable message, so `get_capabilities()` is a convenience rather than a requirement.

## Quantized-KV decode through the wrapper

The int2/int4 kernels are reached from `PagedDecodeWrapper` — pass `bits=2` or `bits=4`
to `plan()`.

```python
w = PagedDecodeWrapper(float_workspace_buffer)
w.plan(indptr, indices, seq_lens,
       num_qo_heads=32, num_kv_heads=8, head_dim=128,
       page_size=64,                       # quantized mode requires 64
       q_data_type=torch.float8_e4m3fn,    # SM100 fold kernel is fp8-Q only
       o_data_type=torch.bfloat16,
       sm_scale=sm_scale / q_scale,        # fold the per-tensor Q descale in here
       bits=2,                             # or 4
       max_kv_len=kv_len)                  # else plan() syncs to read it
# Serving loop: hoist BOTH of these out of the hot path.
out       = torch.empty(batch, 1, 32, 128, device="cuda", dtype=torch.bfloat16)
pscale_lp = w.pscale_lp(mScale)     # else run() does a .max().item() DEVICE SYNC

out = w.run(q_fp8, mKq, mVq, out=out, scale_pool=mScale, pscale_lp=pscale_lp,
            k_pool_order="permuted")   # SM100 int2 (default feed) and int4 read a permuted K pool
```

**The `out=` and `pscale_lp=` arguments above are load-bearing for latency, not
optional polish.** Omit `out=` and every call allocates the output; omit `pscale_lp=` and
every call does a `.max().item()` device sync to derive it. Either one lands
inside whatever a serving integration measures, and neither is visible from the
call signature — this quick-start deliberately shows the fast path because the
short form teaches the slow one.

`k_pool_order=` is required on SM100 wherever the planned kernel reads a permuted K
pool, which is int4 at every configuration and int2 with the default E2M1 feed: nothing
in the pool's bytes reveals the order, so `run()` refuses an undeclared one. See
[docs/K_STORE_PERM.md](docs/K_STORE_PERM.md) for the order a pool writer has to meet.

In quantized mode `k_cache`/`v_cache` carry the **packed uint8 code pools** and the affine
`(scale, zero)` block arrives as `scale_pool`. With `kv_splits=None` the split count *and*
the reduction mode come from the arch cost model
(`_sm100/scheduler.plan_splits_sm100`, `_sm90/scheduler.plan_splits`), which are keyed by
`bits`, the SM90 one also by head dimension, and were calibrated against the quantized kernels.
`k_quant` / `v_quant` select the per-tensor `(axis, group)` granularity (see the capabilities
table above); the default `("token", head_dim)` pair is the shipped fast path.

At head_dim 128, `k_quant=("token", head_dim // 2)` additionally needs a **bfloat16** scale
pool — the token-64 layout doubles the scale ring's shared-memory section and halving the
element is what funds it — so build the pool with
`make_serving_cache_flex(..., sdtype=torch.bfloat16)` and pass
`kernel_kwargs={"sdtype": torch.bfloat16}` to `plan()`. The wrapper validates the pair
rather than choosing for you: a mismatch is an error naming both arguments, and
`run()` rejects a `scale_pool` whose dtype disagrees with what `plan()` compiled for.
Until 2026-07-30 the SM90 path forwarded no `sdtype` at all, so this granularity was
advertised by `get_capabilities()`, accepted by `validate_quant_flex`, and unreachable
through the public wrapper — every such `plan()` died on a constructor assert about a
shared-memory budget the caller had no way to influence.

SM90 status: **validated on H100 SXM5.** The path is wired to
`IntPingPongDecodeSm90` through the same API, cross-compiles under `CUTE_DSL_ARCH=sm_90a`
(`scripts/test_wrapper_quant_sm90_compile.py`), and is gated on hardware by
`baselines/validate.py --arch sm90`, by
`baselines/check_sm90_split_parity.py`, which scores all three split-K merge
mechanisms — direct write, external `SplitKCombine`, and the in-kernel fused v2
tree — at every split count the planner can emit, uniform and ragged, and by the
[end-to-end accuracy gate](#end-to-end-accuracy-gate), which scores all 20
granularity pairings × int2/int4 × three shapes against a naive reference. SM100 is
gated bitwise against the in-repo benchmark driver by
`scripts/test_wrapper_quant_sm100.py`.

## Performance

Measured latencies are not kept in this README. The comparison against the other decode
kernels is produced by `bash baselines/bench.sh --set roofline`, which runs the
architecture's named column set under the repository's one timing protocol;
[`docs/BASELINES_METHODOLOGY.md`](docs/BASELINES_METHODOLOGY.md) explains that protocol.

## Quantized-KV granularity on SM90

`IntPingPongDecodeSm90` supports per-tensor quantization granularity via
`k_quant`/`v_quant` = `(axis, group)`: token-wise 128 (one `(s,z)` per token, the shipped
default), **token-wise 64** for K (one `(s,z)` per token per D-half, bf16 scale pool),
and **channel-wise** with token-groups of 64/128/256 (one `(s,z)` per channel per group —
the KIVI-style config, which is more accurate on outlier-channel K).  K and V choose
independently; channel-wise K folds `sm2·s_d` into the gemm-B operand (Q′ planes built
per group by the otherwise-idle IO warps — per-warp plane ownership: warp 10 builds
even groups, warp 11 odd, concurrently) so the hot softmax fold becomes a single FADD;
channel-wise V drains group-local accumulators through per-channel scales; token-64 K
splits the D-contraction into two wgmma accumulator chains with per-half scales (the
hot fold stays 2 FFMA via a bias pre-fold over per-half Q row-sums).  All configs are
gated against f64 grouped-dequant references (`scripts/test_pingpong_quant_flex.py`,
including GQA-hint-specialized builders).

The accuracy side of the same grid is measured on hardware by the
[end-to-end accuracy gate](#end-to-end-accuracy-gate), which scores every one of
these configs against a naive reference and separates the granularity's cost from
the kernel's own residual instead of quoting them together.

## End-to-end accuracy gate

A single "accuracy" number for a quantized kernel hides which half of the error is
the *format* and which half is the *kernel*.
[`baselines/accuracy_gate.py`](baselines/accuracy_gate.py) separates them by scoring
three things over one fixture — **(1)** textbook f64 attention over bf16 Q/K/V with no
quantization; **(2)** the same textbook attention over K/V round-tripped through the
affine `(axis, group)` contract, Q cast to the dtype the kernel really takes (bf16 on
SM90, fp8_e4m3 on SM100); **(3)** ours, through `PagedDecodeWrapper.plan(bits=…)` →
`run()` at the planner's own split choice. Both references live in one DSL-free file,
[`baselines/accuracy_reference.py`](baselines/accuracy_reference.py).

**(3) vs (2) is the gate** — the reference consumes the same integer codes, the same
`(s, z)` at the same pool precision and the same query dtype, so only the kernel's own
doing is left in it. **(2) vs (1) is the format's cost**: reported, never gated. It moves
with the data, so a bar on the total would have to be loose enough to catch nothing — the
trap [`baselines/validate.py`](baselines/validate.py) records having avoided once already.

The gate runs the full envelope (K ∈ {token-128, token-64, channel-64/128/256} ×
V ∈ {token-128, channel-64/128/256}; V token-64 is designed but not wired) × int2/int4 ×
three shapes: uniform, **ragged** (masked tails and empty split partitions), and **batch 1
at KV=8192**, where the cost model splits the KV so the split-K merge is exercised rather
than assumed. The gated statistic is the per-(batch, kv-head) worst, which strictly
subsumes the global norm. Two bars because the population is two: token-wise K sits on the
query-dtype floor and is held to 5.0e-3, while channel-wise K carries per-group Q′-plane
rounding the naive reference deliberately does not model and is held to 2.5e-2.

**Both bars are proven every run, not asserted.** The gate injects the failure it
exists to catch — one request's V pages taken from one page over, the shape of the bug
fixed in `656c274` — under each bar's own quantization and query dtype, and prints what
that reads as a multiple of the tolerance. It also prints a sensitivity floor it
deliberately does *not* gate: the smallest version of the same bug, one swapped page
pair, reads below the kernel's own residual, because a decode softmax averages it away.
That limit is printed rather than implied; corruption at that scale is caught by
[`baselines/check_sm90_split_parity.py`](baselines/check_sm90_split_parity.py)
and the flex gates, which score against bit-exact references.

Reproduce with:

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -m baselines.accuracy_gate \
    --out data/h100_accuracy_gate_<date>.csv
python -m baselines.accuracy_gate --quick   # one config, one shape, for iteration
```

**SM100: the pools are host-verified, and this gate has no B200 run.** Pools go through
the same `repack_sm100.build_pools_flex` path the B200 gates use, with the K and V store
permutations taken from the kernel's own resolvers, and every config must pass a *full
round trip* on the host — codes and `(s, z)` must equal the reference exactly, then
`dequant_pools_flex` inverts the pools and the result is gathered back through the block
table, which exercises the scale-block layout, both store permutations and the page
scatter. The tolerances are therefore calibrated on SM90 only. B200 kernel error is
measured instead by [`baselines/accuracy_arms.py`](baselines/accuracy_arms.py), whose
`kernel_rel_l2` scores each arm against its own storage-matched oracle.

This gate covers **our** int2/int4 kernels only. The competitor int2/int4 columns are
scored by [`baselines/validate.py`](baselines/validate.py), whose loose cross-format
tolerance is the price of covering formats whose output dtype is fp8.

## Further reading

- [docs/integration.md](docs/integration.md) — wiring into an SGLang-style attention backend
- [docs/design.md](docs/design.md) — kernel internals and arch dispatch

## Research kernels (SM90)

Beyond the production `PagedDecodeWrapper`, the research-grade kernels live under
`cutedsl_decode._sm90`. One exception:
`int_decode_pingpong` is reachable from the public wrapper via `plan(bits=...)`
(see "Quantized-KV decode through the wrapper") and is **validated on H100
hardware**, not merely compile-checked. Everything else here stays internal:

- `int_decode_pingpong` — current-gen int2/int4 quantized-KV paged decode (TMA producer
  WG + two symmetric ping-pong full-attention consumer WGs). Prepare-phase affine
  scale/zero folding, 1-LOP3-per-pair unpack, log2-domain softmax; `serving_cache` holds
  the packed-code + interleaved-scale pool contract.
- `splitk_combine` — fused flash-decode LSE combine for split-K partials.
- `warpmma_decode` — earlier-gen configurable int2/int4 quantized-KV decode
  (token/channel/seq quant granularity), superseded by the pingpong kernels for speed.

The required upstream helpers (quack, flash_attn-cute; BSD-3) are vendored as a minimal
subset under `cutedsl_decode._vendor/` — the package is fully self-contained:
`pip install -e .` and run any harness with no `PYTHONPATH` or `sys.path` setup. Harnesses:
`scripts/` (tests/benches) and `scripts/probes/` (mechanic-proving probes).

## License

BSD-3-Clause. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for attribution of adapted code
from the NVIDIA CUTLASS Hopper FMHA example and FlashInfer.
