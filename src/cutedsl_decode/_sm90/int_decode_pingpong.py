# Copyright 2026 Donglin Zhuang
# SPDX-License-Identifier: Apache-2.0
"""PING-PONG int2/int4 quantized-KV GQA decode (SM90): two SYMMETRIC full-attention
consumer warpgroups per CTA, i.e. an intra-CTA split-K.

It replaced a functional QK->PV pipelined kernel for the batched-decode regime, and was
faster there; that kernel was retired on 2026-09-30 and is in git at
`2237269:src/cutedsl_decode/_sm90/int_decode.py`. In the pipelined kernel the QK->PV
P-handoff is a DEPENDENCY rather than a capacity gap, so no work migration can fill PV's
idle; here the dependency is deleted instead. Each consumer WG runs the COMPLETE chain
(K-unpack -> gemm0 -> log2-domain online softmax -> P-tilde -> V-unpack -> gemm1
O0/O1/lc) on its own tiles, and the two independent streams overlap on the SM's pipes
free-running.

Development record, including the measured comparison and the fold ablation numbers:
docs/kernel/int_decode_pingpong_sm90.md.

STRUCTURE (384 threads, one CTA per (kv-head, batch) x split-K partition)

  warps 8-11  IO/producer WG (Q staging, TMA K/V and scale rings)
  warps 0-3   consumer A — full attention on EVEN tiles
  warps 4-7   consumer B — full attention on ODD tiles

Tiles INTERLEAVE rather than splitting into halves, so both consumers stream
concurrently from the in-order ring: stage t%S is owned by the WG matching t's parity,
and each WG walks its own PipelineState starting at stage wg and advancing by 2 per
tile, which is why num_stages must be EVEN. Only the owner releases its stage, so
per-stage consumer arrivals stay 4 and the producer is untouched.

  kernel                  smem views, pipeline construction, warp dispatch
  _stage_q                Q (and, under channel-K, the per-row Q sums) into smem
  _tma_k_warp             warp 8: the K code ring plus the interleaved scale block
  _tma_v_warp             warp 9: the V code ring only
  _build_q_planes         warps 10-11: the channel-K Q'-plane builder
  _consumer_wg            the complete attention chain for one WG's tiles
  _unpack_k_and_prepare   K unpack plus the affine prepare; RETURNS the five values the
                          later phases of the same iteration consume
  _online_softmax         mask, running max, exp2, correction factor
  _merge_and_store        per-WG l/c recombine, CTA-tail merge, store

Four consumer phases stay INLINE on purpose, because extracting them would cost more
than it buys: `_unpack_v_and_prepare` would need 35 parameters to move 71 lines, and
`_gemm_pv` is 12 lines.

PER-WG STATE

The handoff double-buffer slot dimension becomes the WG dimension (sP / sWarpMax /
sLcFactors [..., consumer_wg_idx]). Each WG keeps its running max in
sRunningMax[q, consumer_wg_idx] and rescales its own accumulators with a per-thread
`corr` register, which is valid because the gemm0 and gemm1 C-fragments map the same
column to the same (thread, elem). Because every thread derives its own `corr`, every
thread reads the old max before the softmax barrier after which warp 0 stores the new
one; a read after it is a race (`_online_softmax`). Same-WG STS -> wgmma-descriptor needs
`fence.proxy.async` plus one WG barrier (ids 2/3), since the cross-WG `_PRDY`
release/acquire that used to cover this is gone.

CTA-TAIL MERGE (log2-max domain)

m = max(mA, mB), w_x = exp2(m_x - m); l, c and O combine with w;
out = (O_A*wA + O_B*wB + c)*bmm2/l; mLSE = m*ln2 + log(l). B stashes O to sMergeOut and
raw l/c/m to sMergeL/sMergeC/sRunningMax; ONE _HBAR rendezvous; A merges and writes.
Empty splits degrade gracefully, because m = -1e30 gives w = 0.

CROSS-PROXY ORDERING

TMA writes and wgmma shared-memory reads go through the async proxy, while LDS and STS go
through the generic proxy. A same-address pair split across the two proxies is outside
the PTX memory model's proxy-preserved causality order, so neither an mbarrier
release/acquire nor a named barrier orders it on its own (PTX ISA "Proxies", "Causality
Order"). Every handoff from the generic proxy to the async proxy therefore carries
`fence.proxy.async.shared::cta`, executed by every thread that touched the bytes. A proxy
fence acts only within its own thread, while the thread that publishes is often another
one (a stage release arrives from lane 0 of each warp only, and qBar hands sQ to other
warps), so a barrier between the fences and the publishing thread completes the chain:

  LDS of a TMA stage, then release to the TMA producer. K (codes and the scale block):
      fence, warp barrier, release, on both the fold and the fold-off path. V: released
      after the sP fence and the per-WG barrier, which already cover its code LDS. On the
      grouped body (HEAD DIMENSION) each stage is read once per group, so K and V are both
      released after the last group's copy, each with its own fence and warp barrier; the
      sP fence runs in the first group and cannot cover the V reads that follow it.
      Without this ordering the refill may land before the reads, which then return the
      next occupant's bytes; the SM100 scale ring failed exactly this way
      (docs/kernel/int_fold_decode.md).
  STS, then a wgmma read: sP (fence, per-WG barrier), the channel-K Q' planes and pad
      rows (fence, RDY / qBar), and the token-K sQ staging (fence, qBar).

The reverse direction needs no fence, because observed TMA completion and wgmma
completion are each followed by an implicit generic-async proxy fence. The only TMA
destinations are sKCodes, sVCodes and sKvQuantParams; every other shared buffer
(sLcFactors, sLcOnes, sWarpMax, sRunningMax, sMergeL/C/Out, sLGroup, sBias, sQRowSum) is
generic on both sides and needs only the ordinary barrier, and sBuildQ is read only by the
lane that wrote it, so it needs none.

HEAD DIMENSION

The kernel builds head_dim 128 and 256 (`quant_layout.SM90_HEAD_DIMS`), and the constructor
chooses the body at trace time through `group_channels`, which every site that differs reads:

  whole-D body (D=128, `group_channels` None): the code this kernel has always compiled. One
      QK wgmma group reads a 32-word register A operand unpacked from the thread's whole K row,
      two 64-channel accumulators hold the output, and one PV group covers both halves.
  grouped body (D=256, `group_channels` 128): both contractions run as D/128 groups of 128
      channels, each doing the whole-D body's work on its channels. A thread copies one
      group's packed words of its K or V row from the shared-memory stage at a time
      (`_code_group_words`, `_load_code_group`), unpacks them into the same register A
      operand, issues the group's wgmmas and waits for them to retire before the next group
      rewrites the operand, so the A operand and the live packed words stay at one group's
      worth. The QK groups chain into one score accumulator; each PV group's sums join its
      own output halves of a (4, D/64) f32 register tensor, the only register state that
      grows with D. The lc wgmma runs after the PV groups, from an A operand loaded per tile
      (`pv_tile_restart`). The epilogue stashes, merges and stores every output half, and the
      fused combine runs its second pass once per 128 channels.

At D=256 the constructor admits one layout: token-wise K and V over all 256 channels, the
fold, an f32 scale pool and 2 or 4 ring stages, and resolves the default depth to 2 at both
widths. `quant_layout.sm90_wide_refusal` names every
other quantization pair, scale dtype, fold setting and ring depth, and the constructor raises
ValueError with it before `validate_quant_flex` and every assert. The arguments that layout
leaves free (the bit width, `fused_combine`, `phase_lock`, `int2_fast_mantissa_injection`,
`heads_per_kv`) stay free.

REGISTERS (the hard constraint)

At 2 CTAs per SM the setmaxnregs split MUST sum to an exact multiple of 384 within the
pool or the increment STALLS FOREVER. The shipped split is IO dealloc 32, consumers
alloc 104, so 32*128 + 104*256 = 30720 = EXACTLY 80*384, under the 85*384 2-CTA pool.
`min_blocks_per_sm=2` is asserted.

`scripts/_sass_census_sm90.py` checks that every configuration of the PTX fingerprint,
`scripts/_ptx_fingerprint_sm90.py`, compiles within that split without spilling registers or
serializing its wgmma chain. It compiles each of those configurations, reads the device image
inside the object that `export_to_c` writes for it, and fails a configuration whose image has
a nonzero `STACK` or `LOCAL` figure in `cuobjdump -res-usage`, any `STL` or `LDL`
instruction, an `HGMMA` count other than the PTX's `wgmma.mma_async` count, or a
`WARPGROUP.DEPBAR` count other than the PTX's `wgmma.wait_group` count at any immediate. The
numbers it measures depend on the assembler the DSL vendors and can move with any change to
device code, so they are recorded with their date and commit in
docs/kernel/int_decode_pingpong_sm90.md.

The fingerprint's rows include fourteen at head dimension 256, so the census holds the grouped
body to the same split: each ring depth of the admitted layout at both widths, with the fused
combine off and on and, at int2, the phase lock on and off, all at four query heads per kv head,
and two int2 rows at eight and at two (the GQA ratio is sampled at 2, 4 and 8, as at D=128).
All fourteen pass at 128-channel groups, with 72 `HGMMA` against 72 `wgmma.mma_async` and
8 `WARPGROUP.DEPBAR` against 8 `wgmma.wait_group` (docs/kernel/int_decode_pingpong_sm90.md,
"Head dimension 256").

Two roles once spilled registers to local memory under that split. Neither spills in any
configuration that the census and its compile matrix checked on 2026-09-28, which cover 1 to 5
and 8 query heads per kv head but not 6 or 7. The channel-wise-K Q'-plane builder runs in the
IO warpgroup, with 29 registers for values. It stores each plane element through one of at
most four lane terms plus an immediate (`_plane_store_split`, which needs both planes aligned
to the swizzle's period under channel-wise K), and it stages its query block once per CTA in
`sBuildQ`. Its loop used to carry an address per store and the hoisted query or a query
pointer. Each consumer has 101 registers for values. Under channel-wise V it reads the lc
wgmma's constant operand as one register quad loaded from `sLcOnes` (`lc_ones_quad`), where
ptxas used to hold a separate four-register copy of two values for each lc wgmma, and it reads
each fragment's channel-wise V scales at static offsets from one address.
docs/kernel/int_decode_pingpong_sm90.md, "Register spills in the channel-wise paths
(2026-09-28)", records the measurements.

CODEGEN

The consumer body lives ONCE in `_consumer_wg`, a `@cute.jit` helper taking all state as
explicit parameters (the DSL bans closures capturing locals in dynamic control flow) and
`wg` as a trace-time Constexpr. Each branch call still inlines a fully specialized copy,
so barrier ids and buffer planes stay immediates, and the PTX is byte-identical to the
hand-inlined form.

The class docstring carries the quant-flex envelope and the required keyword-only
``heads_per_kv``; ``__call__`` carries the launch tensors and their shapes. Read both:
the non-legacy scale pool is the sectioned block rather than the interleaved [N, 4]
one, and ``phase_lock`` and ``int2_fast_mantissa_injection`` apply to int2 only.

The sUnpackMagic register-resident unpack magic (1 LOP3 per pair), the log2-domain
softmax (sm2 = sm*log2e, with the mLSE *ln2 fixup), and the prepare-phase affine
s'/z'/sV' folding came verbatim from the pipelined kernel. The host contracts are
documented where they are defined: ``__call__`` for the launch tensors and their TMA
views, `serving_cache.py` for the pool layout and the code pool orders (this kernel
reads the chunk16 order that `code_pool_order` names), and `scripts/probes/repack_rs.py`
for the plane-word bytes. `scripts/probes/fold_numerics.py` gates the fold algebra and
`scripts/probes/probe_rs_bf16.py` the register-operand wgmma.

WGMMA REGISTER OPERANDS

Every wgmma reads registers that went through a `_wgmma_fence` after an ordinary instruction
last wrote them: the A words the unpack wrote, the lc wgmma's A operand, and the PV accumulators
the max correction rescaled. The fence asm takes each of those registers as a tied in/out operand
and the wgmmas read its outputs, so every such write precedes the fence in the PTX, as the PTX
ISA's `wgmma.fence` requires. A fence without operands let LLVM place the unpack after it.

The PV chains fence per k-step: each step's A words just before its wgmmas, and the
accumulators and the lc operand that every step reads at the first step, so a step's unpack need
not precede the earlier steps' wgmmas. A QK chain fences per k-step in the leading consumer under
the phase lock (`qk_fence_per_step`) and once per chain otherwise. That is the QK issue order the
assembler built at D=128 with the fold for the operand-free fence, and of the four placements
measured it is the one closest to the previous speed. Over the 50 timed D=128 cases it is faster
than before in both run orders in one case and slower in one; one fence per chain is slower in 19,
by up to 2.7%, one per QK chain in both consumers is slower in 12, INT2 with token-wise K and V by
up to 1.9%, and per k-step in both consumers spilled at D=256 (docs/kernel/int_decode_pingpong_sm90.md,
"wgmma register operands", which defines the percentages). A chain added later passes its own
registers the same way.

Under the per-tile restart (`pv_tile_restart`) the PV and lc sums of a tile start from zero, and
a wgmma with scale-D 0 ignores its accumulator, so those chains pass only their A words. The lc
chain runs after the PV chain there and fences its A words per k-step, as the PV chain does.

MODIFYING THIS KERNEL

`scripts/_check_roles.py` lists the extraction hazards and runs a check for each; the
first four compile and fail at trace time, and the fifth runs silently wrong.
`int_fold_decode.py`'s module docstring states them. Any refactor must also pass
`scripts/_anchor_pingpong_outputs.py`, which gates bit-identity over 33 configurations,
because a tolerance gate passes a refactor that perturbs every low mantissa bit.

NUMERICS

The numerics gate is `scripts/test_pingpong_quant_flex.py`, which compares every
quantization pair against an f64 attention reference. This kernel has no debug_s
folded-S tap.

The H100 wgmma adds each 16-product step into its f32 accumulator by truncation, so a long
chain of steps over terms of one sign loses accuracy in one direction. The fold with
token-wise V forms its output from two long sums, (O + c) / l: the PV sum O, whose terms are
never negative, and the lc wgmma's correction sum c. On V data centred near zero every term of
c is negative and both sums are far larger than the output, so they cancel. The PV and lc
wgmmas of that path therefore restart at every tile, and the tile's sums join f32 totals in
registers (`pv_tile_restart`, in the constructor).

FOLD ABLATION (`fold_scale_zero`)

Default True is the production fold path. False dequantizes K/V to REAL bf16 in the
unpack (magic-y then packed-bf16 fma; int2 prefolds beta into z'', while int4 must keep
the exact HSUB2 because the bf16 z-128s cancellation is a documented 1.25e-1 failure, and
so does int2 under `int2_fast_mantissa_injection=False`, whose beta is int4's 128),
prefolds sm2 into the K scales so the softmax pre-EX2 is a pure mask-select, reads
per-token V (s,z) pairs BEFORE the K-stage release and carries them packed across
softmax, and feeds the lc-gemm a ones-row so l = sum p and c = 0. On H100 at int2 with
the token-128 pair, it is slower than the fold wherever the planner runs one split
(B >= 32, fold 1.08-1.09x faster). It is faster at B=1, where the planner splits 16 ways
with the fused combine (0.92-0.94x). See data/measured/h100_fold_ab_grid_20260907.csv in
the paper repository, measured at a5312d5.
"""
try:
    from cutedsl_profiling.annotate import annotate
except Exception:                       # profiling optional; annotations become no-ops
    class _NoOpCM:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __call__(self, fn): return fn          # usable as no-op decorator
    def annotate(name): return _NoOpCM()
    annotate.range = lambda name: _NoOpCM()
    annotate.range_push = lambda name: None
    annotate.range_pop = lambda: None
    annotate.mark = lambda name: None
    annotate.kernel = lambda name: _NoOpCM()

import torch, cutlass, cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import warpgroup, cpasync
from cutlass.utils import LayoutEnum
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu.tcgen05.helpers import smem_descriptor_to_int
import cutlass.utils.hopper_helpers as sm90u
from cutedsl_decode.kernel_contract import GQAPagedDecode, normalize_sdtype
from cutedsl_decode._vendor.quack import sm90_utils
from cutedsl_decode._vendor.flash_attn.cute import copy_utils
from cutedsl_decode._sm90.serving_cache import (
    CODE_CHUNK_BYTES, code_block_word_strides, code_row_bytes, code_row_chunks,
    code_v_half_bytes, require_code_pool)
from cutlass.pipeline import PipelineTmaAsync, make_pipeline_state
from cutlass.pipeline import CooperativeGroup, Agent, PipelineUserType

BF16 = cutlass.BFloat16
_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453
_HBAR = 1   # named-barrier id for the CTA-tail merge rendezvous (A+B, 256 threads)
_PLCK = 5   # phase-lock order-barrier pair (int2, phase_lock=True): A arrives 5 / waits 6, B arrives 6 / waits 5
# named-barrier ids in use: 0 = CTA-wide, 1 = _HBAR, 2/3 = per-WG internal, 4 = qBar,
# 5/6 = _PLCK pair (qk-mma issue alternation, one-tile skew, A leads; absent when the
# `phase_lock` constructor flag is off, and at int4 where the pair is never compiled).
# Channel-K adds 7/8 (plane-0 RDY, A/B), 9/10 (plane-0 FREE), 12/13 (plane-1 RDY), 14/15 (plane-1 FREE),
# all count 160 = builder warp 32 + consumer WG 128.  Id 11 is FREE (was the builder gang's
# internal bias barrier before per-warp plane ownership made the reduce warp-local).

MAGIC4 = 0x43004300   # bf16x2 (128.0, 128.0); 0x4300|q == bf16(128+q) for q in [0,16)
MASK4 = 0x000F000F    # isolate low 4 bits of each bf16 lane (int4 codes)
# int2 LINEAR-MAGIC (probe-gated): bf16 lane = 1s+8e+7m; 0x4080 = 4.0 (e=129, m=0);
# m = q<<5 = 32q  =>  y = 4·(1 + 32q/128) = 4 + q, EXACT in bf16.  After `word >> (2p)` the code
# pair sits at bits 0-1/16-17 of the u32 plane word; << 5 lands it at bits 5-6/21-22 = mantissa
# bits 5-6 of each bf16 lane (MASK2L selects exactly those; plane-word garbage dies in the mask).
# Unpack = one shift and one LOP3 per operand word (ptxas merges `>> 2p` and `<< 5` into one net
# shift), with no floating-point op; a raw-q feed would need a bf16x2 sub = HADD2.
# Fold (fold_numerics): y = q + 4  =>  s' = s, z' = z − 4s.  The base-4 encoding (vs the
# original base-2 y = q/2 + 2) makes alpha=1, so the two per-column ALPHA_INV multiplies in
# QK-prepare vanish behind the const_expr guard — int2 joins int4's multiply-free affine path
# with the SAME z' = z − 4s cancellation (s' = s amplifies strictly less than s' = 2s).
# WHY THIS IS THE int2 DEFAULT (`int2_fast_mantissa_injection=True`): of the eight operand words
# one u32 code word yields, the three at shift 0, 2 and 4 move LEFT by 5, 3 and 1, and ptxas
# emits those as IMAD.SHL.U32 (x32, x8) and IMAD.IADD (x + x), which issue on the FMA pipe; the
# other five are SHF.R.U32.HI. Per code word that is 13 ALU-pipe unpack instructions (5 SHF +
# 8 LOP3) and 3 on the FMA pipe, where the opt-in below needs 15 on the ALU pipe (7 SHF + 8 LOP3)
# and none on the FMA pipe, and this kernel's unpack saturates the ALU pipe (inferred from these
# counts and the timing, not from a pipe-utilization profile). On H100 (2026-10-06) the opt-in
# measured 2.7% slower with the fold and 3.8% slower with the fold off; int4 is unchanged.
MASK2L = 0x00600060
MAGIC2L = 0x40804080  # bf16x2 (4.0, 4.0)
# THE OPT-IN int2 FEED, y = 128 + q (`int2_fast_mantissa_injection=False`), int4's magic and
# offset: after `word >> sh` the code pair sits at bits 0-1/16-17, MASK2 keeps exactly those, and
# OR-ing MAGIC4 gives bf16(128 + q), EXACT. Every shift is a right shift (none at sh = 0), all on
# the ALU pipe. Fold: y = q + 128  =>  s' = s, z' = z − 128s, as at int4.
MASK2 = 0x00030003    # isolate low 2 bits of each bf16 lane (int2 codes, the 128 + q feed)
# sScale ring stream slots (all four staged by warp 8, all four consumed by the QK WG):
_KS, _KZ, _VS, _VZ = 0, 1, 2, 3


# ---- shared-memory layout of the lc strip ----
# Plain Python, so `tests/test_sm90_lc_refill_banks_cpu.py` reads these functions out
# of this file and checks their bank arithmetic without the DSL.
def _lc_factor_layout(page_size):
    """(shape, stride) of `sLcFactors` in shared memory, in bf16 elements over (plane, token,
    consumer warpgroup).

    Each 16-token step kt of the lc wgmma is one contiguous 128-byte row of the strip, stored
    in the order the lc wgmma's A registers want it: word e of lane t's 16 bytes in row kt holds
    plane t//4 + 2(e%2) at token pair 8kt + t%4 + 4(e//2), which is what the lc wgmma's A
    operand `rLcATile[4kt + e]` holds. Eight lanes therefore read one row of a step with one
    16-byte load each, on 32 distinct banks, and each load lands in the four registers the
    wgmma reads: lanes 0 to 7 of warp 0 on the whole-D body, and every quarter-warp on the
    grouped body, where lane t reads lane t % 8's bytes.
    The plane mode splits into (plane%2, plane//2) and the token mode into (token%2, token
    pair%4, token bit 3, token//16). CuTe decomposes a flat coordinate into each, so the
    writer keeps its (plane, token, warpgroup) subscripts, and its eight 2-byte stores per
    warp still touch four distinct words on four banks."""
    return (((2, 2), (2, 4, 2, page_size // 16), 2),
            ((32, 2), (1, 8, 4, 64), 4 * page_size))


def _lc_factor_words(page_size):
    """(shape, stride) of the lc operand load's Int32 view of `sLcFactors`, over (word e, lane t,
    step kt, consumer warpgroup): the slice [None, t, kt, wg] is lane t's 16-byte load."""
    return ((4, 8, page_size // 16, 2), (1, 4, 32, 2 * page_size))


def _lc_ones_layout():
    """(shape, stride) of `sLcOnes`, in Int32 words over (word e, row).

    Under channel-wise V the lc wgmma's A operand is a constant: the bf16 pair (1.0, 1.0) in
    words 0 and 2 of consumer warpgroup threads 0-3, which hold row 0 of the A operand, the row
    of l, and zeros everywhere else. Row 0 of `sLcOnes` holds (ones, 0, ones, 0) and its row 1
    zeros; each thread copies its row once per CTA with one 16-byte load, so the four words are
    four distinct registers that every lc wgmma of the tile loop reads."""
    return ((4, 2), (1, 4))


# ---- the channel-K Q'-plane builder's shared-memory addressing ----
# Plain Python, so `tests/test_sm90_build_q_planes_cpu.py` reads these functions out of this
# file and checks them without the DSL.
def _swizzle_period(swizzle):
    """Bytes after which `Swizzle<bits, base, shift>`, applied to byte addresses, repeats:
    2**(bits + base + shift), 1,024 for the 128-byte swizzle (3, 4, 3)."""
    bits, base, shift = swizzle
    return 1 << (bits + base + shift)


def _swizzle_of(composed_layout):
    """(bits, base, shift) of a swizzled shared-memory layout's `Swizzle`, the one its
    allocation hands the pointer (`smem.allocate_tensor(..., swizzle=layout.inner)`)."""
    sw = composed_layout.inner
    return int(sw.num_bits), int(sw.num_base), int(sw.num_shift)


def _plane_store_split(offset, heads, quarters, swizzle, elem_bytes, alignment):
    """Trace-time split of the channel-K builder's Q'-plane store offsets.

    Builder lane `lane` stores channel `d = lane + 32 * qtr` of head row `h`. A plane is
    stored under `swizzle`, which the shared-memory pointer applies to each full byte
    address, so an address's (h, qtr) part does not commute with the XOR as far as the
    compiler can see, and it keeps one address register per (h, qtr) across the builder's
    group loop: 16 at four heads and D = 128, of the 29 registers the IO warpgroup has for
    values. With the plane base aligned to the swizzle's period (`_swizzle_period`), the
    swizzled address is the plane base plus the swizzled offset within the plane, and every
    store splits into a trace-time part and a lane term,

        swizzled element offset of (h, 32 * qtr + lane) = base_elem + (lane ^ lane_xor),

    with base_elem a multiple of 32 and lane_xor in [0, 32). The builder computes
    `lane ^ lane_xor` once for each distinct lane_xor, four at most under the 128-byte
    swizzle, and every base_elem becomes the store's immediate.

    `offset(h, d)` is the unswizzled element offset of (h, d) in the plane layout, `swizzle`
    the (bits, base, shift) of its byte-address swizzle, and `alignment` the plane base's
    byte alignment. Returns {(h, qtr): (base_elem, lane_xor)} over h < heads and
    qtr < quarters. Every lane of every pair is checked, so a plane aligned below the period,
    or a layout or swizzle for which the split does not hold, raises ValueError at trace time
    rather than storing a channel to the wrong address."""
    period = _swizzle_period(swizzle)
    if alignment % period != 0:
        raise ValueError(
            f"a Q' plane aligned to {alignment} bytes: the store split needs the base aligned "
            f"to the swizzle's period, {period} bytes")
    bits, base, shift = swizzle
    mask = ((1 << bits) - 1) << base
    quarter_bytes = 32 * elem_bytes

    def swizzled(h, d):
        address = elem_bytes * offset(h, d)
        return address ^ ((address >> shift) & mask)

    split = {}
    for h in range(heads):
        for qtr in range(quarters):
            first = swizzled(h, 32 * qtr)
            base_bytes = first - first % quarter_bytes
            lane_xor = (first - base_bytes) // elem_bytes
            for lane in range(32):
                if swizzled(h, 32 * qtr + lane) != base_bytes + elem_bytes * (lane ^ lane_xor):
                    raise ValueError(
                        f"the Q'-plane store of (h={h}, d={32 * qtr + lane}) is not a trace-time "
                        f"part plus a lane XOR under swizzle {swizzle}; the builder's store "
                        f"addresses do not describe this layout")
            split[(h, qtr)] = (base_bytes // elem_bytes, lane_xor)
    return split


def _build_q_layout(heads_per_kv, head_dim):
    """(shape, stride) of `sBuildQ`, in Int32 words over (word, lane, quarter, builder warp).

    Before its group loop each builder lane stages, once per CTA, the query values it
    multiplies: channel `lane + 32 * qtr` of every real head, two heads to a bf16x2 word,
    the low half the even head. The words of one (lane, quarter) are contiguous and their
    count is padded to a power of two, so one 4-, 8- or 16-byte access moves all of them,
    and consecutive lanes are consecutive, so a warp's access covers one contiguous run."""
    used = (heads_per_kv + 1) // 2
    words = 1 << (used - 1).bit_length()
    quarters = head_dim // 32
    return (words, 32, quarters, 2), (1, words, 32 * words, 32 * words * quarters)


# ---- inline-PTX wgmma mechanics (kept from the fp8 skeleton; bf16 RS asm lifted from probe_rs_bf16) ----
@dsl_user_op
def _fence_acq_rel_gpu(*, loc=None, ip=None):
    # gpu-scope memory fence: publishes this thread's prior global stores before the
    # split-counter arrive (threadFenceReduction pattern — every WG-A thread fences,
    # the WG barrier then orders the fences before the elected thread's atomic).
    llvm.inline_asm(None, [], "fence.acq_rel.gpu;", "", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def _wgmma_fence(*operands, loc=None, ip=None):
    """`wgmma.fence.sync.aligned`, with every register of `operands` passed through it.

    The PTX ISA requires a `wgmma.fence` between an ordinary instruction's access to a register
    and a `wgmma.mma_async` that reads that register as its A operand or its accumulator. A fence
    asm with no operands orders only memory and other side effects, so LLVM was free to place the
    pure integer unpack that writes an A word, or the rescale of an accumulator, after it, and it
    did: at 1,000 of the 2,040 wgmmas of the D=128 fingerprint rows
    (profile/h100_planb_prc_20261001/PROVENANCE.md, "Register-operand order").

    So each element of each register tensor in `operands`, Int32 A words or Float32
    accumulators, is an output of the fence asm and an input tied to that output ("=r" or "=f",
    then the output's index, because MLIR's llvm.inline_asm has no "+"), and the fenced value is
    written back into the tensor. A wgmma that reads the tensor afterwards reads the fence's
    output, so the write is ordered before the fence by a data dependency. This is CUTLASS's
    `warpgroup_fence_operand`, folded into the fence itself. The asm text names no operand, so
    the fence is the same one PTX instruction.

    The kernel passes, before the wgmmas that read them, the registers that ordinary
    instructions write (module docstring, WGMMA REGISTER OPERANDS). With no operands this is the
    bare fence, which the RS probes use."""
    regs = []
    for t in operands:
        if t.element_type is not Int32 and t.element_type is not Float32:
            raise TypeError(f"_wgmma_fence: a {t.element_type} register tensor is neither A "
                            f"words (Int32) nor an accumulator (Float32)")
        regs += [(t, i, t.element_type) for i in range(cute.size(t))]
    if not regs:
        llvm.inline_asm(None, [], "wgmma.fence.sync.aligned;", "", has_side_effects=True,
                        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
        return
    types = [T.i32() if dtype is Int32 else T.f32() for _, _, dtype in regs]
    constraints = ",".join(["=r" if dtype is Int32 else "=f" for _, _, dtype in regs]
                           + [str(k) for k in range(len(regs))])
    st = llvm.inline_asm(
        llvm.StructType.get_literal(types),
        [dtype(t[i]).ir_value(loc=loc, ip=ip) for t, i, dtype in regs],
        "wgmma.fence.sync.aligned;", constraints, has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    for k, (t, i, dtype) in enumerate(regs):
        t[i] = dtype(llvm.extractvalue(types[k], st, [k], loc=loc, ip=ip))


@dsl_user_op
def _wgmma_commit(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "wgmma.commit_group.sync.aligned;", "", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def _wgmma_wait0(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "wgmma.wait_group.sync.aligned 0;", "", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def _wgmma_n8k16_bf16(acc, a0, a1, a2, a3, smem_b_desc, scaleD: cutlass.Constexpr,
                      transB: cutlass.Constexpr, *, loc=None, ip=None):
    # RS-form bf16 wgmma m64n8k16: A in registers (a0..3 = 4x u32 = 8 bf16), B from smem (GMMA
    # descriptor), acc = 4 f32 in/out.  The acc is passed through tied constraints, because MLIR
    # llvm.inline_asm has NO "+f": $0-3 = acc out, $4-7 = tied acc-in (alias $0-3), $8-11 = a0..3,
    # $12 = smem_b_desc.  scaleD is materialized as a predicate via setp (bf16 wgmma scale-d operand is a
    # predicate); trailing imm = imm-trans-b (0 = B K-major as stored, validated by the probe gate).
    accv = [Float32(acc[i]).ir_value(loc=loc, ip=ip) for i in range(4)]
    av = [Int32(x).ir_value(loc=loc, ip=ip) for x in (a0, a1, a2, a3)]
    constraints = ",".join(["=f"] * 4 + [str(i) for i in range(4)] + ["r"] * 4 + ["l"])
    st = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        accv + av + [Int64(smem_b_desc).ir_value(loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.b32 p, " + str(int(scaleD)) + ", 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{$0,$1,$2,$3}, {$8,$9,$10,$11}, $12, p, 1, 1, " + str(int(transB)) + ";\n}\n",
        constraints, has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    for i in range(4):
        acc[i] = Float32(llvm.extractvalue(T.f32(), st, [i], loc=loc, ip=ip))


# ---- packed-bf16 helpers (the fold_scale_zero=False ablation path, and the channel-K
# builder's staged query words) ----
@dsl_user_op
def _cvt2_bf16x2(hi: Float32, lo: Float32, *, loc=None, ip=None):
    # pack two f32 into one bf16x2 word (low half = lo, high half = hi)
    rst = llvm.inline_asm(
        T.i32(), [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f", has_side_effects=False,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Int32(rst)


@dsl_user_op
def _hsub2_bf16(a: Int32, b: Int32, *, loc=None, ip=None):
    # packed bf16x2 subtract: a - b (words hold two bf16 lanes)
    rst = llvm.inline_asm(
        T.i32(), [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
        "sub.rn.bf16x2 $0, $1, $2;", "=r,r,r", has_side_effects=False,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Int32(rst)


@dsl_user_op
def _hfma2_bf16(a: Int32, b: Int32, c: Int32, *, loc=None, ip=None):
    # packed bf16x2 fma: a*b + c
    rst = llvm.inline_asm(
        T.i32(), [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip),
                  Int32(c).ir_value(loc=loc, ip=ip)],
        "fma.rn.bf16x2 $0, $1, $2, $3;", "=r,r,r,r", has_side_effects=False,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Int32(rst)


@dsl_user_op
def _bf16x2_lo_f32(w: Int32, *, loc=None, ip=None):
    # low bf16 lane of a packed word -> f32 (bf16 bits are the f32's high half): 1 SHF
    rst = llvm.inline_asm(
        T.f32(), [Int32(w).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nshl.b32 t, $1, 16;\nmov.b32 $0, t;\n}\n",
        "=f,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(rst)


@dsl_user_op
def _bf16x2_hi_f32(w: Int32, *, loc=None, ip=None):
    # high bf16 lane of a packed word -> f32: 1 LOP
    rst = llvm.inline_asm(
        T.f32(), [Int32(w).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nand.b32 t, $1, 0xFFFF0000;\nmov.b32 $0, t;\n}\n",
        "=f,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(rst)


class IntPingPongDecodeSm90(GQAPagedDecode):
    """Ping-pong int2/int4 paged-decode (SM90): two symmetric full-attention consumer WGs
    per CTA over interleaved tiles + CTA-tail log2-max merge.  See the module docstring for
    the design.
    Constraints: head_dim in {128, 256} (256 in one layout, the module docstring's HEAD DIMENSION),
    q_head_tile=8, page_size=64, num_stages EVEN, min_blocks_per_sm=2 (32/104 setmaxnregs split
    built in).

    ``q_head_tile`` is the number of Q head rows one threadblock packs, fixed at 8 here
    because the inline RS wgmma is m64n8k16.  It is NOT the GQA ratio.  The ratio is
    ``heads_per_kv``, a trace-time constructor argument, and when it is smaller than the
    tile the rows from the ratio up to the tile are zero-padded.  The constructor REFUSES
    a ratio larger than the tile, which is why the ratio has to be known at construction:
    a larger one silently leaves the heads at and above the tile absent from the output.

    ``IntDecodeSm100`` uses the same name for the same quantity, the Q head rows one
    threadblock packs.  What differs is what each kernel does when the ratio exceeds it:
    SM100 tiles the heads across several threadblocks, SM90 refuses."""

    #: THE CODE-POOL ORDER THIS KERNEL READS, and the predicate every writer asks. A
    #: (page, kv-head) block is [per/16 chunks, 128 threads, 16 bytes], chunk c of thread t
    #: holding bytes [16c, 16c+16) of that thread's plane-word row (`serving_cache`, CODE
    #: POOL ORDERS). It is a property of the kernel, not a choice: there is no knob, because
    #: a knob would let a caller pair a pool with the wrong reader. The TMA copies the block
    #: as one contiguous run whatever its order; the order exists for the consumer's
    #: shared-memory read, one 16-byte load per chunk at 16t + 2048c, so every warp load
    #: covers 512 consecutive bytes. At D=128 that is one chunk at int2, where the order
    #: coincides with the row order, and two at int4 (K: channels 0-63 and 64-127; V: the
    #: two output halves), where the 32-byte row pitch put two lanes of every 8-lane group
    #: on the same four banks (8 wavefronts per load against an ideal of 4, measured with
    #: ncu in profile/h100_chunk16_bankconf_20260925/).
    #: Everything that depends on the order is derived from this attribute rather than
    #: restating it: the consumer's shared-memory view (`_code_ring_words`), the refusal
    #: in `__call__` of a pool whose shape is not this order's, the fake tensors of
    #: `pingpong_runtime.compile_constructed`, and IntAppend's scatter.
    code_pool_order = "chunk16"

    def __init__(self, head_dim, page_size=64, q_head_tile=8, num_stages=None, bits=4, sdtype=torch.float32,
                 min_blocks_per_sm=None, fold_scale_zero=True,
                 fused_combine=False, k_quant=None, v_quant=None,
                 phase_lock=True,
                 int2_fast_mantissa_injection=True,
                 *, heads_per_kv):
        # THE DEFAULT QUANTIZATION IS TOKEN-WISE OVER THE FULL HEAD DIMENSION, resolved here
        # and nowhere else: `compile_pingpong` passes None through, so the default has one
        # spelling. A literal ("token", 128) default was the D=128 pair at every head
        # dimension, so a kernel built at D=256 without a pair was refused instead of built in
        # the one layout its grouped body implements.
        if k_quant is None:
            k_quant = ("token", head_dim)
        if v_quant is None:
            v_quant = ("token", head_dim)
        # k_quant / v_quant (trace-time): per-tensor quant granularity (axis, group).
        #   ("token", D) stores one (s,z) per token over all D channels. It is the shipped
        #     path and the default, and when both tensors use it the legacy interleaved scale
        #     pool and code paths compile byte-identical.
        #   ("token", D/2) is allowed for K only, and it stores one (s,z) per token for each
        #     half of D, which splits the accS quads. It requires sdtype=bf16: its ring section
        #     holds two pairs per token, and at bf16 those take the bytes of one f32 pair.
        #     V with ("token", D/2) is deferred, and `validate_quant_flex` refuses it on this
        #     architecture and on SM100 alike.
        #   ("channel", Gt) — one (s,z) per channel over Gt tokens, Gt in {64,128,256}
        #                     (page-aligned; pages of one group carry duplicate scales).
        #                     K: IO builds sm2·s∘Q planes + z·Q bias, fold becomes an
        #                     FADD.  V: per-group accumulator drain + z·l term.
        # ONE quantization envelope for both decode kernels, refused in ONE place.
        # Scattering it across inline asserts is how the V granularity check came to
        # admit token-64 while a second assert twenty-five lines below refused it, and
        # a reader of only the first produced five RUN-FAIL cells on a pristine build.
        # `validate_quant_flex` is where a reader looking for the envelope will find
        # it; keep it there.
        # NORMALIZED FIRST, because the head-dimension envelope below and the token-64 check
        # after it compare it, and a caller passing the string spelling would otherwise fail
        # there rather than at the contract.
        sdtype_name = normalize_sdtype(sdtype)
        from ..quant_layout import (SM90_HEAD_DIMS, quant_block_offsets, sm90_wide_refusal,
                                    validate_quant_flex)
        # TMA ring depth (correctness-neutral prefetch), resolved here, before the head-dimension
        # envelope below, which admits a depth.
        #   D=128: the bit-aware default inherited from the stage sweep of the retired pipelined
        #     kernel (2237269:src/cutedsl_decode/_sm90/int_decode.py, CONFIG), where int2 won at 4
        #     (its small codes leave SMEM headroom, and A1's barrier-relief unlocked deeper
        #     prefetch vs the mem-latency stall) and int4 at 2 (2x-larger codes + closer to
        #     BW-bound).
        #   Any other admitted head dimension (256): 2 at both widths. At the planner's split, B in
        #     {1, 8, 128} and KV in {8192, 16384}, 4 stages was slower than 2 at five shapes at int4
        #     (0.9-4.7%) and three at int2 (0.4-3.5%), and faster at one at each width (1.1%, 1.3%),
        #     so the rule, 4 only where it is faster somewhere and slower nowhere, keeps 2
        #     (profile/h100_planb_prd_20261002/PROVENANCE.md).
        if num_stages is None:
            num_stages = (4 if bits == 2 else 2) if head_dim == 128 else 2
        # THE HEAD-DIMENSION ENVELOPE, refused FIRST, before `validate_quant_flex` and every
        # assert below. A head dimension outside `SM90_HEAD_DIMS` raises ValueError, and so, at
        # an admitted head dimension other than 128, does every quantization pair, scale dtype,
        # fold setting and ring depth outside the one layout `sm90_wide_layouts` names, under
        # `python -O` too. `sm90_wide_refusal` normalizes the pairs itself, and only a scale
        # dtype spelling that neither kernel accepts is refused before it, by
        # `normalize_sdtype`. This is a memory-safety boundary: a head dimension the epilogue
        # does not tile would write past its output row or leave channels unwritten. At 128 the
        # kernel builds every pair `validate_quant_flex` admits; at any other admitted head
        # dimension it runs the grouped body (`group_channels` below).
        if head_dim not in SM90_HEAD_DIMS:
            raise ValueError(
                f"head_dim must be one of {SM90_HEAD_DIMS} on this kernel; got {head_dim}")
        if head_dim != 128:
            refusal = sm90_wide_refusal(head_dim, k_quant, v_quant, sdtype=sdtype_name,
                                        fold_scale_zero=fold_scale_zero, num_stages=num_stages)
            if refusal is not None:
                raise ValueError(refusal)
        k_quant, v_quant = validate_quant_flex(k_quant, v_quant, page_size, head_dim)
        # THE BODY, CHOSEN BY THE HEAD DIMENSION AT TRACE TIME, and the one rule every site
        # that differs reads. None selects the whole-D body, which is what D=128 has always
        # compiled: one QK wgmma group over every k-step, two 64-channel output accumulators,
        # and one PV group over both halves. Any other admitted head dimension runs the
        # GROUPED body, which cuts the QK contraction and the PV contraction into groups of
        # this many channels, each group doing the whole-D body's work on its channels, so
        # the register A operands and the packed words a thread holds at one time stay at one
        # group's worth and only the D/16 f32 output accumulators grow with D. 128 at both
        # widths: the SASS census of the launched D=256 binaries passes at 128
        # (docs/kernel/int_decode_pingpong_sm90.md, "Head dimension 256").
        self.group_channels = None if head_dim == 128 else 128
        self.k_axis, self.k_group = k_quant
        self.v_axis, self.v_group = v_quant
        if (self.k_axis, self.k_group) == ("token", head_dim // 2):
            assert sdtype_name == "bf16", \
                "token-64 doubles its ring section; sdtype=bf16 funds it (smem budget)"
        # heads_per_kv is TRACE-TIME: the GQA ratio does not change between launches,
        # so it is a constructor argument rather than a kernel operand. Do not add a
        # runtime route beside it: the same number arriving by two routes leaves the
        # signature unable to say which one wins. `IntDecodeSm100` has never had a
        # runtime GQA ratio either.
        #
        # ONE EXEMPTION, and it is not an invitation to a second. `__call__` derives
        # `merge_head_limit_dyn` and the CTA-tail merge predicate reads it instead of
        # this constant. That is a code-generation fix, measured, and it is clamped to
        # this value so it cannot decide anything this constant would not have decided;
        # it is not a second route to the ratio and nothing may select against it.
        #
        # Keyword-only and required on purpose.  There is no defensible default: a wrong
        # ratio does not fail, it silently reads the wrong Q rows.
        # SM90's OWN RULE, not a shared one: SM100 packs the grouped heads on the tiled
        # mode and launches `ceil_div(q_heads // kv_heads, q_head_tile)` head tiles per
        # KV head, so a ratio larger than the tile is SPREAD over several threadblocks
        # there rather than refused. Here the inline RS wgmma produces exactly
        # `q_head_tile` head columns and there is nowhere else for the rest to go.
        assert 1 <= heads_per_kv <= q_head_tile, (
            f"heads_per_kv must be in [1, {q_head_tile}]: the inline RS wgmma produces "
            f"exactly {q_head_tile} head columns, so a larger ratio would leave the "
            f"heads at and above {q_head_tile} silently absent from the output. Got "
            f"{heads_per_kv}.")
        self.heads_per_kv = heads_per_kv
        self.qoff = quant_block_offsets(k_quant, v_quant, N=page_size, D=head_dim)
        assert fold_scale_zero or self.qoff["legacy"], \
            "fold_scale_zero=False (dequant ablation arm) supports only the legacy token-128/token-128 config"
        assert self.qoff["ring_tot"] <= 512 and self.qoff["ring_tot"] % 128 == 0, \
            "ring section must fit a (128, ring_tot/128) TMA box"

        # fold_scale_zero (ablation flag, trace-time): True = production affine fold (never
        # dequantize; s'/z' applied post-gemm via the softmax FFMA + rcp/zr lc-gemm rows).
        # False = classic dequant: K/V codes are dequantized to REAL bf16 values in the unpack
        # (y-magic, sub beta, packed-bf16 fma with per-row K / per-token V scales; sm2 is
        # constant-prefolded into the K scales), softmax is a pure mask-select + max-reduce,
        # P-tilde is raw p, and the lc-gemm degenerates to an l-row of ones (c = 0).
        # fused_combine (trace-time): compile the XQA-style in-kernel last-CTA split
        # reduction (v2 tree form; see the epilogue block; scheduler.plan_fused_combine
        # is the enable policy). Enable ONLY for multi-split plans: compiled into a
        # single-split kernel the block adds register pressure although its runtime
        # guard never fires.
        self.fused_combine = fused_combine
        self.fold = fold_scale_zero
        # phase_lock (tuning flag, trace-time; int2 only): compile the order-barrier pair
        # that forces the two consumer warpgroups' QK-mma issue windows into strict
        # alternation with a one-tile skew. The barriers only order issue, so both
        # settings must produce bit-identical output, and a two-arm pair of
        # `baselines/bench_sweep.py` checks that (`compare_bench_sweeps.py --pair` marks
        # `outputs_agree`) as well as timing the two arms.
        #
        # At D=128 its effect on the shipped kernel is unmeasured: the last paired A/B there
        # predates a5312d5, which changed the consumer loop bodies, and covered only the
        # legacy token-128/token-128 pair. docs/kernel/int_decode_pingpong_sm90.md names the
        # instrument to re-run it with; the old record is
        # 3f7f180:docs/kernel/int_decode_pingpong_sm90.md:86-123. At D=256 it was compared at
        # int2 and 2 stages, and it stays on there, because switching it off was slower at two
        # shapes and faster at none (profile/h100_planb_prd_20261002/PROVENANCE.md).
        self.phase_lock = bool(phase_lock)
        # WHETHER THE FLAG REACHES EMITTED CODE, exposed rather than left for a caller to
        # re-derive. The pair is compiled only at int2, so `phase_lock` is inert at int4 and
        # the two settings then compile the SAME kernel. An A/B that read only
        # `phase_lock` off the two objects would see them differ and time one kernel
        # against itself, which is the "applied as a dead statement" failure this
        # repository keeps paying for. The barrier guards below read this predicate, so
        # there is one rule and not two. Derived from the constructor's `bits`, which is
        # the value `self.bits` is assigned further down, and `_validate_shape` just
        # below rejects any width that is neither 2 nor 4.
        self.phase_lock_active = self.phase_lock and bits == 2
        # int2_fast_mantissa_injection (encoding flag, trace-time; int2 only): the value the int2
        # unpack feeds the wgmma A operand. True, the default, feeds y = 4 + q (MASK2L, MAGIC2L,
        # BETA_FOLD 4), the faster unpack on Hopper (the module comment above MASK2L says why).
        # False feeds y = 128 + q, int4's constant and offset (MAGIC4 with MASK2, BETA_FOLD 128),
        # and with the fold off it subtracts the magic exactly, as int4 does, where the default
        # prefolds beta into the bf16 zero. False is kept by owner decision (2026-10-06) so that
        # the 128 + q construction the paper presents exists in the kernel. On H100 it is 2.7%
        # slower with the fold and 3.8% slower with the fold off, and its fold error grows with
        # the tokens one split covers (9.15e-3 at B=2 over one 32768-token split, against about
        # 2.5e-3 for the default; docs/kernel/int_decode_pingpong_sm90.md,
        # "`int2_fast_mantissa_injection`"). Constructor-only, like `phase_lock` and
        # `fold_scale_zero`: `compile_pingpong` does not take it, so its default is spelled here
        # and nowhere else.
        self.int2_fast_mantissa_injection = bool(int2_fast_mantissa_injection)
        # WHETHER THE FLAG REACHES EMITTED CODE, exposed as `phase_lock_active` is and for the
        # same reason. int4 has one feed, so the flag is inert there and both settings compile
        # the SAME int4 kernel; it is accepted rather than refused, like `phase_lock`, and an A/B
        # that reads this predicate off its two objects sees that they do not differ. Every
        # encoding site below reads this predicate, so there is one rule and not two.
        self.int2_fast_mantissa_injection_active = self.int2_fast_mantissa_injection and bits == 2
        # THE LC WGMMA'S ONES-ROW OPERAND AS ONE REGISTER QUAD, derived from the
        # configuration. On the fold path with channel-wise V the lc wgmma computes l alone,
        # and its A operand is a constant (`_lc_ones_layout`). Written as two values, each
        # passed twice to every lc wgmma, it made ptxas hold a separate four-register copy of
        # them for each of a tile's four lc wgmmas, all live with the PV wgmmas in flight: 16
        # registers for two values at the consumer's register peak. Loaded once per CTA from
        # `sLcOnes`, it is four distinct registers that all four read. The fold-off path keeps
        # the two-value form, because it admits only the token-128 pair.
        self.lc_ones_quad = self.fold and self.v_axis == "channel"
        # THE FOLD'S TOKEN-WISE-V SUMS RESTART AT EVERY TILE, derived from the configuration.
        # On that path the PV accumulator sums P~ * (q + beta), whose terms are never negative,
        # and the lc accumulator's c rows sum P~ * (z/s - beta); the output is (O + c) / l. On
        # V data centred near zero, such as the Gaussian cases measured, every c term is
        # negative and the output is the small difference of two sums each about 1,200 (int2)
        # and 7,000 (int4) times its size (docs/kernel/int_decode_pingpong_sm90.md, "Fold
        # accuracy with token-wise V"). The H100 wgmma adds a 16-product step into its f32
        # accumulator by truncation: it aligns the accumulator and the products on the largest,
        # keeps 26 bits and cuts the sum toward zero. A sum of terms of one sign carried
        # through the whole split therefore loses part of an accumulator ulp at every step,
        # always in the same direction, and the cancellation multiplies that loss (7.5e-2 at
        # int2 over 32768 tokens in one split on the Gaussian case of
        # baselines/check_sm90_split_parity.py, against a 5e-3 bar). So each tile's PV and lc
        # wgmmas start from zero (scale-d 0 on the tile's first step), and the tile's sums
        # join the running f32 totals in one rounded multiply-add that also applies the max
        # correction, total = total * corr + tile. The lc wgmma runs after the PV wgmmas have
        # retired, from an A operand loaded per tile, so that neither that operand nor the lc
        # sums hold registers while the PV wgmmas are in flight (issued together, int4 and
        # head dimension 256 spilled). Channel-wise V already restarts per V group, and the
        # fold-off path sums the dequantized values themselves, with no offset beta for a
        # second sum to cancel. The grouped body (`group_channels`) admits only this path,
        # and implements only the restart.
        self.pv_tile_restart = self.fold and self.v_axis == "token"
        assert self.group_channels is None or self.pv_tile_restart, \
            "the grouped body implements only the per-tile restart of the fold with token-wise V"
        # The four shape arguments both kernels take, checked in ONE place. `bits` and
        # `page_size` were asserted here with a different message from SM100's for the
        # same rule, and `sdtype` was asserted against a DIFFERENT VOCABULARY: this
        # kernel refused "bf16" and SM100 refused torch.bfloat16, so one configuration
        # could not be handed to both. The head dimension was refused above, before this
        # check asserts, because an assert is removed by `python -O`.
        self._validate_shape(page_size, head_dim, q_head_tile, bits)
        # ...and then what only THIS kernel narrows.
        assert q_head_tile == 8, "inline RS wgmma is m64n8k16, so q_head_tile must be 8; the 16-wide legacy path was stripped"
        # PAGED-ONLY: the repacked N(=64)-token tile IS the page, and mPT[(b, gt)] indexes
        # the GLOBAL page (NP) dim of the code and scale pools. There is no contiguous path;
        # it was exactly an identity block table (block_table[b,gt]=base_b+gt).
        # `self.sdtype` STAYS A TORCH DTYPE, because that is what this kernel's tensor
        # construction wants, and changing it would be a body change rather than a
        # contract one. What changed is the door: either spelling gets in.
        self.sdtype = torch.float32 if sdtype_name == "f32" else torch.bfloat16
        self.sdtype_name = sdtype_name
        self.scale_element_type = Float32 if sdtype_name == "f32" else BF16   # CuTe element type of the scale pool
        self.scale_element_bytes = 4 if sdtype_name == "f32" else 2             # scale element bytes (tx_count)
        self.min_blocks_per_sm = min_blocks_per_sm
        # pingpong runs at 2 CTA/SM with the built-in 32/104 setmaxnregs split (module docstring).
        self._min_blocks = 2 if min_blocks_per_sm is None else min_blocks_per_sm
        assert self._min_blocks == 2, "pingpong: min_blocks_per_sm must be 2 (32/104 setmaxnregs split built in)"
        # `num_stages` was resolved to its default before the head-dimension envelope.
        assert num_stages % 2 == 0, "pingpong: stage ownership follows tile parity — num_stages must be EVEN"
        self.head_dim = head_dim; self.page_size = page_size; self.q_head_tile = q_head_tile; self.num_stages = num_stages; self.bits = bits
        # affine feed constants (trace-time): the wgmma A operand holds y = alpha·q + beta
        # — int4 (1,128) magic, int2 (1,4) base-4 linear-magic, or int2 (1,128) under
        # `int2_fast_mantissa_injection=False`.  Fold at use (fold_numerics):
        #   s' = ALPHA_INV·s,  z' = z − BETA_FOLD·s   (ALPHA_INV = 1/alpha, BETA_FOLD = beta/alpha)
        # alpha=1 for every feed: the ALPHA_INV multiplies in QK-prepare are const_expr-elided.
        self.ALPHA_INV = 1.0
        self.BETA_FOLD = 4.0 if self.int2_fast_mantissa_injection_active else 128.0
        # repacked plane-word row sizes (bytes per warpgroup-thread row, see repack_rs.py), from
        # the one pool geometry every writer uses (`serving_cache`, CODE POOL GEOMETRY):
        self.k_code_bytes_per_thread, self.v_code_bytes_per_thread = code_row_bytes(
            head_dim, page_size, bits)
        self.num_threads = 384   # 12 warps: QK WG(0-3) ∥ PV WG(4-7) ∥ IO WG(8-11):
        # warp 8 = K producer, warp 9 = V producer, warps 8-11 collectively stage Q (bf16, no convert).

    def _mma_qk(self):   # RS maps for gemm0: S^T[Nkv,PG] = unpacked-K[Nkv,D] (RMEM) @ Q[PG,D]^T (SMEM desc)
        return sm90u.make_trivial_tiled_mma(
            BF16, BF16, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K, Float32,
            atom_layout_mnk=(self.page_size // 64, 1, 1), tiler_mn=(64, self.q_head_tile),
            a_source=warpgroup.OperandSource.RMEM)

    def _mma_pv(self):   # RS maps for gemm1: O^T[64,PG] = V^T[64,Nkv] (RMEM) @ P^T (sP B-descriptor)
        return sm90u.make_trivial_tiled_mma(
            BF16, BF16, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K, Float32,
            atom_layout_mnk=(1, 1, 1), tiler_mn=(64, self.q_head_tile),
            a_source=warpgroup.OperandSource.RMEM)

    def _code_ring_words(self, sCodes, code_bytes_per_thread):
        """Int32 view (thread, (word, chunk), stage) of a code ring in shared memory.

        A stage holds one pool block verbatim, in the order `code_pool_order` names, and
        `code_block_word_strides` says where each row word sits in it: for chunk16, thread
        t's row is 4 words per chunk at word 4t + 512c.  Slicing one thread keeps that chunk
        stride, and one `autovec_copy` of the slice issues one 16-byte load per chunk into
        the thread's row registers in row order."""
        chunks = code_row_chunks(code_bytes_per_thread)
        words = CODE_CHUNK_BYTES // 4
        thread, chunk = code_block_word_strides(self.code_pool_order, code_bytes_per_thread)
        return cute.make_tensor(
            cute.recast_ptr(sCodes.iterator, dtype=Int32),
            cute.make_layout((128, (words, chunks), self.num_stages),
                             stride=(thread, (1, chunk), 128 * words * chunks)))

    def _group_word_count(self, code_bytes_per_thread):
        """Words of one thread's row that one wgmma group of the grouped body reads:
        `group_channels` channels of K (bits * group_channels / 64 words), or the
        group_channels / 64 output halves of V, which is the same count."""
        return code_bytes_per_thread // 4 // (self.head_dim // self.group_channels)

    def _code_group_words(self, sCodes, code_bytes_per_thread):
        """Int32 view (thread, group word, group, stage) of a code ring in shared memory, for the
        grouped body.

        Group g holds channels W*g to W*(g+1)-1 of K, W = `group_channels`, or the W/64 output
        halves of V from half g*W/64 onward, which are row words [g*n, (g+1)*n) with
        n = `_group_word_count`. Row word w of thread t is at word 4t + 512*(w // 4) + w % 4 in
        the chunk16 order (`code_block_word_strides`), so at 128 channels a group is two whole
        16-byte chunks at int4 (K k-steps 8g to 8g+7; V halves 2g and 2g+1) and one at int2,
        and at 64 channels half a chunk at int2. The group mode is split into groups within a
        chunk and chunk groups, and one of the two has extent 1 at any group size, so one
        `autovec_copy` of a slice issues one 16-byte load per whole chunk, or one 8-byte load."""
        chunks = code_row_chunks(code_bytes_per_thread)
        words = CODE_CHUNK_BYTES // 4
        thread, chunk = code_block_word_strides(self.code_pool_order, code_bytes_per_thread)
        n = self._group_word_count(code_bytes_per_thread)
        in_chunk, chunks_per_group = min(n, words), max(n // words, 1)
        groups_per_chunk = max(words // n, 1)
        groups = self.head_dim // self.group_channels
        return cute.make_tensor(
            cute.recast_ptr(sCodes.iterator, dtype=Int32),
            cute.make_layout(
                (128, (in_chunk, chunks_per_group), (groups_per_chunk, groups // groups_per_chunk),
                 self.num_stages),
                stride=(thread, (1, chunk), (n, chunk * chunks_per_group), 128 * words * chunks)))

    def _load_code_group(self, sCodeWords, code_bytes_per_thread, wg_thread_idx, group, stage):
        """One group's words of this thread's row of a code ring stage, copied from shared memory
        into a fresh register tensor of `_group_word_count` words (`_code_group_words`)."""
        n = self._group_word_count(code_bytes_per_thread)
        rWords = cute.make_rmem_tensor(n, Int32)
        sRow = sCodeWords[(wg_thread_idx, None, group, stage)]
        cute.autovec_copy(cute.make_tensor(sRow.iterator.align(4 * min(n, CODE_CHUNK_BYTES // 4)),
                                           sRow.layout), rWords)
        return rWords

    @cute.jit
    def _consumer_wg(self, consumer_wg_idx: cutlass.Constexpr, thread_idx, seq_len, tile_count, split_idx, tiles_per_cta,
                     split_count, kv_head_idx, batch_idx, heads_per_kv, merge_head_limit_dyn, softmax_scale, output_scale, pipeline_k, pipeline_v, sUnpackMagic, sQ, sQRowSum,
                     sKCodes, sVCodes, sKvQuantParams, sP, sWarpMax, sRunningMax, sMergeL, sMergeC, sMergeOut, sLcFactors, sQPlane1, sBias, sLGroup, sLcOnes, mScaleRaw, mPT, mOp, mOut, mLSE, mCnt, cnt_index):
        """One ping-pong consumer warpgroup: the COMPLETE attention chain (K-unpack -> gemm0 ->
        log2-domain online softmax -> P-tilde -> V-unpack -> gemm1 O0/O1/lc) on this WG's tiles,
        then the CTA-tail merge (B stashes, A merges + writes).

        Intra-CTA split-K: WG A (consumer_wg_idx=0) owns EVEN tiles, WG B
        (consumer_wg_idx=1) owns ODD
        tiles (interleaved so both stream concurrently from the shared ring; each WG strides its
        pipeline state by 2 and releases only its own stages, so the producer machinery is
        untouched). Each WG runs against its own sP/sWarpMax/sLcFactors/sRunningMax plane
        (= the old handoff slot dim, now the consumer_wg_idx dim; the skew-2 _PRDY/_PFREE
        protocol and cross-WG sCorr are GONE --
        each WG rescales its own accumulators with per-thread corr registers).  At the CTA tail
        the two partial softmaxes merge in the log2-max domain: m = max(mA,mB), w_x =
        exp2(m_x - m) applied to O, l and c (B stashes to smem, A merges and writes out).

        CODEGEN: called once per consumer branch with consumer_wg_idx a trace-time
        cutlass.Constexpr, so the body still inlines twice and WG-derived barrier ids / buffer
        planes / merge-role
        const_exprs stay immediates.  All state arrives as EXPLICIT parameters: the DSL bans
        closures capturing locals in dynamic control flow, but a plain @cute.jit call with
        arguments traces fine (A/B-verified vs the hand-inlined form: byte-identical PTX,
        int2 and int4)."""
        D, Nkv, PG, S = self.head_dim, self.page_size, self.q_head_tile, self.num_stages
        W = self.group_channels    # channels per wgmma group on the grouped body; None at D=128
        k_code_bytes_per_thread, v_code_bytes_per_thread = self.k_code_bytes_per_thread, self.v_code_bytes_per_thread    # repacked row bytes (per warpgroup thread)
        if cutlass.const_expr(self._min_blocks == 2):
            # pingpong register split at 2 CTA/SM: 32*128 (IO) + 104*256 (consumers)
            # = 30720 = EXACTLY 80*384 (pool feasibility is a hard deadlock gate, see IO
            # branch).
            cute.arch.setmaxregister_increase(104)
        wg_thread_idx = thread_idx - 128 * consumer_wg_idx
        unpack_magic = sUnpackMagic[0]
        mma_qk = self._mma_qk(); thr_qk = mma_qk.get_slice(wg_thread_idx)
        tQkCoord = thr_qk.partition_C(cute.make_identity_tensor((Nkv, PG)))
        # The K ring as the consumer reads it: one thread's whole row on the whole-D body, and
        # one group's words of it on the grouped body (`_code_group_words`).
        if cutlass.const_expr(W is None):
            sKCodeWords = self._code_ring_words(sKCodes, k_code_bytes_per_thread)
        else:
            sKCodeWords = self._code_group_words(sKCodes, k_code_bytes_per_thread)
        if wg_thread_idx < PG:
            sRunningMax[(wg_thread_idx, consumer_wg_idx)] = Float32(-1e30)
        cute.arch.barrier(barrier_id=2 + consumer_wg_idx, number_of_threads=128)   # per-WG: sRunningMax init
        cute.arch.barrier(barrier_id=4, number_of_threads=384)        # qBar (IO + BOTH WGs)
        warp_idx_in_wg = wg_thread_idx // 32; lane = wg_thread_idx % 32
        # the QK k-steps one wgmma group issues: every k-step of D on the whole-D body
        qk_k_steps = D // 16 if cutlass.const_expr(W is None) else W // 16
        # how often a QK chain fences its A words: per k-step in the leading consumer under the
        # phase lock, once per chain otherwise (WGMMA REGISTER OPERANDS, module docstring)
        qk_fence_per_step = self.phase_lock_active and consumer_wg_idx == 0
        # Names bound ONLY inside a `cutlass.const_expr` guard below, declared here
        # so the consumer's phase methods can take them as parameters. A call
        # argument is evaluated unconditionally, so an unbound name raises "free
        # variable referenced before assignment in enclosing scope" even when every
        # USE sits behind the same guard. Same idiom, and same reason, as the SM100
        # kernel's pre-init block.
        rKScaleHi = rKZeroHi = None
        rOutGlobLo = rOutGlobHi = rLTotal = None
        rOutAccumLo = rOutAccumHi = rOutAccum = None
        k_dequant_scale_log2 = None
        rKScale = cute.make_rmem_tensor(4, Float32); rKZero = cute.make_rmem_tensor(4, Float32)
        if cutlass.const_expr(self.k_axis == "token" and self.k_group == D // 2):
            rKScaleHi = cute.make_rmem_tensor(4, Float32)
            rKZeroHi = cute.make_rmem_tensor(4, Float32)   # dead after the prepare bias pre-fold
        rVScale = cute.make_rmem_tensor(4, Float32); rVZero = cute.make_rmem_tensor(4, Float32)
        rOldMax = cute.make_rmem_tensor(4, Float32); rNewMax = cute.make_rmem_tensor(4, Float32)
        rMaxCorrection = cute.make_rmem_tensor(4, Float32)
        mma_pv = self._mma_pv(); thr_pv = mma_pv.get_slice(wg_thread_idx)
        if cutlass.const_expr(W is None):
            sVCodeWords = self._code_ring_words(sVCodes, v_code_bytes_per_thread)
        else:
            sVCodeWords = self._code_group_words(sVCodes, v_code_bytes_per_thread)
        pv_k_steps = Nkv // 16
        v_words_per_half = code_v_half_bytes(Nkv, self.bits) // 4   # V^T row = D/64 stacked halves
        accum_fragment_size = PG // 2
        if cutlass.const_expr(W is None):
            rOutAccumLo = cute.make_rmem_tensor(accum_fragment_size, Float32); rOutAccumHi = cute.make_rmem_tensor(accum_fragment_size, Float32)
            for z in cutlass.range_constexpr(accum_fragment_size):
                rOutAccumLo[z] = Float32(0.0); rOutAccumHi[z] = Float32(0.0)
        else:
            # (fragment element, output half): half h holds output channels 64h to 64h+63, so
            # every PV wgmma writes the slice rOutAccum[None, h], which `_wgmma_n8k16_bf16`
            # reads and writes element by element; all D/64 halves are live across the tile loop
            rOutAccum = cute.make_rmem_tensor((accum_fragment_size, D // 64), Float32)
            for z in cutlass.range_constexpr(accum_fragment_size * (D // 64)):
                rOutAccum[z] = Float32(0.0)
        rLcAccum = cute.make_rmem_tensor(accum_fragment_size, Float32)
        for z in cutlass.range_constexpr(accum_fragment_size):
            rLcAccum[z] = Float32(0.0)
        if cutlass.const_expr(self.fold):
            lc_words_shape, lc_words_stride = _lc_factor_words(Nkv)
            sLcWords = cute.make_tensor(cute.recast_ptr(sLcFactors.iterator, dtype=Int32),
                                        cute.make_layout(lc_words_shape, stride=lc_words_stride))
        tPvCoord = thr_pv.partition_C(cute.make_identity_tensor((64, PG)))
        softmax_scale_log2 = softmax_scale * Float32(_LOG2E)           # log2-domain softmax
        # The l-gemm A operand where the lc wgmma computes l alone (fold off, and fold on with
        # channel-wise V): bf16 1.0s on output row 0 ONLY (a0/a2 of warp-0 group-0 threads),
        # zeros elsewhere -> C row 0 = l = sum_t p_t, row 1 = 0 = c; the fold epilogue's
        # hi/lo row split then reads (l, 0) unchanged. The measured alternative, a shuffle
        # plus sWSum l-reduce with no lc-gemm, is slower: the softmax-window shuffles cost
        # more than 4 wgmma issue slots.
        rLcOnes = None
        if cutlass.const_expr(self.lc_ones_quad):
            # fold on: this thread's row of `sLcOnes`, one 16-byte load before the tile loop,
            # four distinct registers that every lc wgmma reads (`lc_ones_quad` in the
            # constructor)
            rLcOnes = cute.make_rmem_tensor(4, Int32)
            sLcOnesRow = sLcOnes[(None, Int32(0) if wg_thread_idx < 4 else Int32(1))]
            cute.autovec_copy(cute.make_tensor(sLcOnesRow.iterator.align(16), sLcOnesRow.layout), rLcOnes)
        if cutlass.const_expr(not self.fold):
            # fold off: the two values, the ones word and the zero word, each passed twice to
            # every lc wgmma; a register tensor, so that the PV fence can pass them through it
            rLcOneZero = cute.make_rmem_tensor(2, Int32)
            rLcOneZero[0] = Int32(0); rLcOneZero[1] = Int32(0)
            if wg_thread_idx < 4:
                rLcOneZero[0] = Int32(0x3F803F80)
        if cutlass.const_expr(not self.fold):
            k_dequant_scale_log2 = softmax_scale_log2 * Float32(self.ALPHA_INV)   # K dequant scale prefold: s*alpha_inv*sm2
        # per-tensor quant granularity (trace-time; see __init__)
        K_CH = self.k_axis == "channel"
        V_CH = self.v_axis == "channel"
        K_TOK64 = self.k_axis == "token" and self.k_group == D // 2   # per-token per-D-half (s,z)
        LEGACY_RING = self.qoff["legacy"]
        # flat ring index = BASE + token*STRIDE (+1 for z); trace-time constants (no closures
        # in dynamic control flow — DSL 4.5.2 ban)
        K_BASE = _KS if LEGACY_RING else (self.qoff["k_ring"] if self.qoff["k_ring"] is not None else 0)
        K_STR = 4 if LEGACY_RING or K_TOK64 else 2   # tok-64 ring row = (s_lo, z_lo, s_hi, z_hi)
        V_BASE = _VS if LEGACY_RING else (self.qoff["v_ring"] if self.qoff["v_ring"] is not None else 0)
        V_STR = 4 if LEGACY_RING else 2
        gtk = (self.k_group // Nkv) if K_CH else 1     # tiles per K group
        gtv = (self.v_group // Nkv) if V_CH else 1     # tiles per V group
        if cutlass.const_expr(V_CH):
            # two-level PV accumulation: group-local rOutAccum drains into the s_d-scaled
            # global accumulators at V-group crossings (plus the z_d·l_G zero-term); both
            # levels ride the per-tile max correction.  rLTotal accumulates the drained
            # per-group l for the final normalization (rLcAccum is group-local here).
            rOutGlobLo = cute.make_rmem_tensor(accum_fragment_size, Float32)
            rOutGlobHi = cute.make_rmem_tensor(accum_fragment_size, Float32)
            rLTotal = cute.make_rmem_tensor(accum_fragment_size, Float32)
            rVDrain = cute.make_rmem_tensor(4 * accum_fragment_size, Float32)   # prefetched drain scales
            for z in cutlass.range_constexpr(accum_fragment_size):
                rOutGlobLo[z] = Float32(0.0); rOutGlobHi[z] = Float32(0.0); rLTotal[z] = Float32(0.0)
            for z in cutlass.range_constexpr(4 * accum_fragment_size):
                rVDrain[z] = Float32(0.0)
        # stride-2 pipeline state: start at stage wg, advance 2 per consumed tile.
        consumer_state = make_pipeline_state(PipelineUserType.Consumer, S)
        if cutlass.const_expr(consumer_wg_idx == 1):
            consumer_state.advance()
        wg_tile_count = (tile_count + Int32(1) - consumer_wg_idx) // 2   # A: ceil(steps/2), B: floor(steps/2)
        wg_tile_idx = Int32(0)
        prev_k_group = Int32(-1)
        while wg_tile_idx < wg_tile_count:
            # Same reason as the block above, for names bound inside THIS loop. They
            # are declared here rather than before the loop on purpose: a name bound
            # in the loop body must be pre-declared in the loop body, and hoisting the
            # declaration outside would make it loop-carried IR.
            k_plane = None
            rScoreAccum2 = None
            tPvAWords = tPvB = None
            global_tile_idx = split_idx * tiles_per_cta + 2 * wg_tile_idx + consumer_wg_idx   # this WG's global tile
            if cutlass.const_expr(K_CH):
                k_group_idx = global_tile_idx // Int32(gtk)
                k_plane = k_group_idx % 2
                if k_group_idx != prev_k_group:
                    # barrier ids are PER (wg x plane parity): the builder runs up to two
                    # groups ahead, and consecutive groups sharing one id would accumulate
                    # arrivals across phases (partial release / phase slip).
                    # FREE for the previous group is arrived EARLY (right after that tile's
                    # qk _wgmma_wait0, sBias hoisted to registers in prepare) — granting the
                    # builder ~2 tile-slots of lookahead instead of zero; the late-FREE form
                    # exposed the whole build chain on the consumer critical path.
                    # RDY: plane built. One barrier whose id is computed (7 + WG for plane 0,
                    # 12 + WG for plane 1): written as two branch-selected barriers, NVVM 23
                    # predicates both, and the pair slows INT2 channel-wise K.
                    cute.arch.barrier(barrier_id=7 + consumer_wg_idx + 5 * k_plane, number_of_threads=160)
                    prev_k_group = k_group_idx
            if cutlass.const_expr(V_CH):
                v_group_idx = global_tile_idx // Int32(gtv)
                v_group_ends = (wg_tile_idx + 1 == wg_tile_count) | ((global_tile_idx + 2) // Int32(gtv) != v_group_idx)
                if (wg_tile_idx == 0) | ((global_tile_idx - 2) // Int32(gtv) != v_group_idx):
                    # first own tile of a V group: fresh group-local accumulators
                    for z in cutlass.range_constexpr(accum_fragment_size):
                        rOutAccumLo[z] = Float32(0.0); rOutAccumHi[z] = Float32(0.0); rLcAccum[z] = Float32(0.0)
            with annotate.range("TMA-wait-K"):
                pipeline_k.consumer_wait(consumer_state)
            with annotate.range("QK-prepare"):
                tQkAWords, tQkB, tQkB1, sV2, zV2 = self._unpack_k_and_prepare(
                    D,
                    K_BASE,
                    K_CH,
                    K_STR,
                    K_TOK64,
                    Nkv,
                    PG,
                    V_BASE,
                    V_CH,
                    V_STR,
                    consumer_state,
                    consumer_wg_idx,
                    k_code_bytes_per_thread,
                    k_dequant_scale_log2,
                    k_plane,
                    lane,
                    pipeline_k,
                    pv_k_steps,
                    qk_k_steps,
                    rKScale,
                    rKScaleHi,
                    rKZero,
                    rKZeroHi,
                    rVScale,
                    rVZero,
                    sBias,
                    sKCodeWords,
                    sKvQuantParams,
                    sLcFactors,
                    sQ,
                    sQPlane1,
                    sQRowSum,
                    softmax_scale_log2,
                    tQkCoord,
                    thr_qk,
                    unpack_magic,
                    wg_thread_idx,
                )
            with annotate.range("qk-mma"):
                rScoreAccum = cute.make_rmem_tensor(4, Float32)
                if cutlass.const_expr(K_TOK64):
                    rScoreAccum2 = cute.make_rmem_tensor(4, Float32)   # hi-half contraction (kt >= 4)
                if cutlass.const_expr(self.phase_lock_active):
                    # phase-lock (int2-only): force the two consumers'
                    # qk-mma issue windows into strict alternation with a one-tile skew
                    # (A leads).  Each WG ARRIVES on its own id after commit (A: _PLCK,
                    # B: _PLCK+1) and WAITS on the partner's id before issuing; A's first
                    # tile skips the wait (skew primer).  Counting-barrier safety: every
                    # trip = partner's arrive (128) + own sync (128) = 256, and a WG cannot
                    # re-arrive before its own wait consumed the partner's arrive, so ids
                    # never alias across tiles; the ragged tail (A owns an odd last tile)
                    # leaves at most one dangling 128-count at kernel end (ids 5/6 unused
                    # elsewhere).  Empty/one-tile splits never wait on a partner that
                    # lacks the pairing tile (A's tile ii waits on B's tile ii-1, which
                    # exists whenever A's tile ii does; and vice versa).
                    if cutlass.const_expr(consumer_wg_idx == 0):
                        if wg_tile_idx > 0:
                            cute.arch.barrier(barrier_id=_PLCK + 1, number_of_threads=256)
                    else:
                        cute.arch.barrier(barrier_id=_PLCK, number_of_threads=256)
                if cutlass.const_expr(W is None):
                    # the A words the unpack wrote, per chain or per k-step (`qk_fence_per_step`);
                    # the score accumulator is written by the chain alone, whose first wgmma
                    # ignores it (scale-D 0)
                    if cutlass.const_expr(not qk_fence_per_step):
                        _wgmma_fence(tQkAWords)
                    for kt in cutlass.range_constexpr(qk_k_steps):
                        rQkAFrag = tQkAWords[None, None, kt]
                        if cutlass.const_expr(qk_fence_per_step):
                            _wgmma_fence(rQkAFrag)
                        smem_b_desc = smem_descriptor_to_int(tQkB[None, None, kt].iterator)
                        if cutlass.const_expr(K_CH):
                            smem_b_desc1 = smem_descriptor_to_int(tQkB1[None, None, kt].iterator)
                            smem_b_desc = smem_b_desc if k_plane == 0 else smem_b_desc1
                        if cutlass.const_expr(K_TOK64):
                            # split the D contraction at channel 64: kt 0-3 -> lo accumulator,
                            # kt 4-7 -> hi (scaleD=0 restarts each chain; both trace-time)
                            _wgmma_n8k16_bf16(rScoreAccum if kt < 4 else rScoreAccum2,
                                              rQkAFrag[0], rQkAFrag[1], rQkAFrag[2], rQkAFrag[3], smem_b_desc,
                                              0 if kt % 4 == 0 else 1, 0)
                        else:
                            _wgmma_n8k16_bf16(rScoreAccum, rQkAFrag[0], rQkAFrag[1], rQkAFrag[2], rQkAFrag[3], smem_b_desc,
                                              0 if kt == 0 else 1, 0)
                    _wgmma_commit()
                    if cutlass.const_expr(self.phase_lock_active):
                        # release the partner's issue window while this group's mma drains
                        cute.arch.barrier_arrive(barrier_id=_PLCK + consumer_wg_idx, number_of_threads=256)
                    _wgmma_wait0()
                else:
                    # THE GROUPED QK CONTRACTION. Group g is channels W*g to W*g+W-1: its packed
                    # words are unpacked into the same register A operand (group 0's in
                    # `_unpack_k_and_prepare`, at the whole-D body's place), and its W/16 wgmmas
                    # read Q from shared memory through the B descriptors of k-blocks
                    # (W/16)*g + kt and chain into one score accumulator, scale-D 0 only on the
                    # very first. Every group ends in wait_group 0, so no A register is
                    # rewritten while a group that reads it is pending, and only one group's
                    # packed words are ever live.
                    k_groups = D // W
                    for g in cutlass.range_constexpr(k_groups):
                        if cutlass.const_expr(g > 0):
                            # group g's words, read from this thread's row of the K stage in
                            # shared memory after group g-1 retired
                            rKGroupWords = self._load_code_group(
                                sKCodeWords, k_code_bytes_per_thread, wg_thread_idx, g,
                                consumer_state.index)
                        if cutlass.const_expr(g == k_groups - 1):
                            # the last read of the K stage, codes and scales alike: fence, warp
                            # barrier, release (CROSS-PROXY ORDERING), as the whole-D body does
                            # after its one copy
                            cute.arch.fence_proxy("async.shared", space="cta")
                            cute.arch.sync_warp()
                            pipeline_k.consumer_release(consumer_state)
                        if cutlass.const_expr(g > 0):
                            for kt in cutlass.range_constexpr(qk_k_steps):
                                rQkAFrag = tQkAWords[None, None, kt]
                                for p in cutlass.range_constexpr(4):
                                    if cutlass.const_expr(self.bits == 4):
                                        rQkAFrag[p] = ((rKGroupWords[kt] >> (4 * p)) & Int32(MASK4)) | unpack_magic
                                    elif cutlass.const_expr(self.int2_fast_mantissa_injection_active):
                                        t2 = (rKGroupWords[kt // 2] >> (2 * (4 * (kt % 2) + p))) << 5
                                        rQkAFrag[p] = (t2 & Int32(MASK2L)) | unpack_magic
                                    else:
                                        rQkAFrag[p] = ((rKGroupWords[kt // 2] >> (2 * (4 * (kt % 2) + p))) & Int32(MASK2)) | unpack_magic
                        # group g's A words, as on the whole-D body
                        if cutlass.const_expr(not qk_fence_per_step):
                            _wgmma_fence(tQkAWords)
                        for kt in cutlass.range_constexpr(qk_k_steps):
                            rQkAFrag = tQkAWords[None, None, kt]
                            if cutlass.const_expr(qk_fence_per_step):
                                _wgmma_fence(rQkAFrag)
                            smem_b_desc = smem_descriptor_to_int(tQkB[None, None, qk_k_steps * g + kt].iterator)
                            _wgmma_n8k16_bf16(rScoreAccum, rQkAFrag[0], rQkAFrag[1], rQkAFrag[2], rQkAFrag[3], smem_b_desc,
                                              0 if g == 0 and kt == 0 else 1, 0)
                        _wgmma_commit()
                        if cutlass.const_expr(self.phase_lock_active and g == k_groups - 1):
                            # one arrive per tile, after the last group's commit
                            cute.arch.barrier_arrive(barrier_id=_PLCK + consumer_wg_idx, number_of_threads=256)
                        _wgmma_wait0()
                if cutlass.const_expr(K_CH):
                    # early FREE: the plane's last reads (wgmma B operand) completed at wait0
                    # and the bias lives in registers — release on this WG's LAST own tile of
                    # the group.  One arrive per (WG, group); arrivals on one parity id are
                    # separated by a full builder round-trip (RDY(g+2) needs the build of g+2
                    # which needed this FREE), so counting phases stay aligned.
                    if (wg_tile_idx + 1 == wg_tile_count) | ((global_tile_idx + 2) // Int32(gtk) != k_group_idx):
                        # FREE: 9 + WG for plane 0, 14 + WG for plane 1, computed as RDY's is
                        cute.arch.barrier_arrive(barrier_id=9 + consumer_wg_idx + 5 * k_plane, number_of_threads=160)
            global_tile_idx = split_idx * tiles_per_cta + 2 * wg_tile_idx + consumer_wg_idx   # this WG's global tile
            valid_token_count = cutlass.min(Int32(Nkv), seq_len - global_tile_idx * Nkv)
            with annotate.range("softmax"):
                self._online_softmax(
                    K_CH,
                    K_TOK64,
                    V_CH,
                    consumer_wg_idx,
                    lane,
                    rKScale,
                    rKScaleHi,
                    rKZero,
                    rMaxCorrection,
                    rNewMax,
                    rOldMax,
                    rScoreAccum,
                    rScoreAccum2,
                    rVScale,
                    sP,
                    sQRowSum,
                    sRunningMax,
                    sWarpMax,
                    tQkCoord,
                    valid_token_count,
                    warp_idx_in_wg,
                    wg_thread_idx,
                )
            with annotate.range("TMA-wait-V"):
                pipeline_v.consumer_wait(consumer_state)
            with annotate.range("PV-prepare"):
                if cutlass.const_expr(V_CH):
                    # V-DRAIN SCALE PREFETCH. On group-ending tiles the drain's 16 per-channel
                    # scale LDGs are issued here in PV-prepare, so they land under the V unpack
                    # and PV wgmma instead of forming a cold serial chain inside the drain.
                    # Full record: git show 3f7f180:src/cutedsl_decode/_sm90/int_decode_pingpong.py (lines 744-746).
                    if v_group_ends:
                        # REFUTED, do not re-propose: pointing BOTH WGs at one shared
                        # page's duplicate scale copy, in group-start (cold) and
                        # group-last (warm) variants, is slower on int2 V-ch128/256.
                        # The int2 V-channel consumer is perturbation-sensitive at the
                        # 104-register cap and the halved line traffic buys nothing
                        # robust. Keep the per-WG group-ending page.
                        v_page_pf = mPT[(Int64(batch_idx), global_tile_idx)]
                        # The page's scale rows, located once by its 64-bit page coordinate
                        # (64-BIT INDEXING, `kernel`), so the offsets inside them, bounded by
                        # HKV times the scale block, stay 32-bit. Each fragment's four scales
                        # (s and z of both 64-channel halves) are then read at static offsets
                        # from one address, the fragment's first scale, which the loads carry
                        # as immediates. Indexing the pool with page, kv head and channel at
                        # once instead let consumer B's copy of this code give every load its
                        # own 64-bit address and keep the kv-head term as a 64-bit value across
                        # the tile loop, which spilled at the register cap at V-channel-64,
                        # where every tile drains.
                        gVDrainPage = mScaleRaw[(Int64(v_page_pf), None, None)]
                        for j in cutlass.range_constexpr(accum_fragment_size):
                            dl_p, _ = tPvCoord[j]
                            gVDrainFrag = cute.domain_offset((kv_head_idx, self.qoff["v_ch"] + dl_p), gVDrainPage)
                            rVDrain[4 * j + 0] = gVDrainFrag[(0, 0)].to(Float32)
                            rVDrain[4 * j + 1] = gVDrainFrag[(0, D)].to(Float32)
                            rVDrain[4 * j + 2] = gVDrainFrag[(0, 64)].to(Float32)
                            rVDrain[4 * j + 3] = gVDrainFrag[(0, D + 64)].to(Float32)
                if cutlass.const_expr(W is None):
                    if cutlass.const_expr(not self.pv_tile_restart):
                        # under the per-tile restart the totals are rescaled where the tile's
                        # sums join them, after pv-mma
                        for j in cutlass.range_constexpr(accum_fragment_size):
                            rOutAccumLo[j] = rOutAccumLo[j] * rMaxCorrection[j]   # cols match: cS_pv[j][1] == cS_qk[j][1]
                            rOutAccumHi[j] = rOutAccumHi[j] * rMaxCorrection[j]
                            rLcAccum[j] = rLcAccum[j] * rMaxCorrection[j]
                            if cutlass.const_expr(V_CH):
                                rOutGlobLo[j] = rOutGlobLo[j] * rMaxCorrection[j]
                                rOutGlobHi[j] = rOutGlobHi[j] * rMaxCorrection[j]
                                rLTotal[j] = rLTotal[j] * rMaxCorrection[j]
                    rVCodeWords = cute.make_rmem_tensor(v_code_bytes_per_thread // 4, Int32)
                    sVCodeRow = sVCodeWords[(wg_thread_idx, None, consumer_state.index)]
                    cute.autovec_copy(cute.make_tensor(sVCodeRow.iterator.align(16), sVCodeRow.layout), rVCodeWords)
                    # NOT released here: the V stage is released below, behind the sP fence
                    # and the per-WG barrier (CROSS-PROXY ORDERING).
                    _, tPvA0, tPvB = sm90_utils.partition_fragment_ABC(
                        thr_pv, (64, PG, Nkv), None, sP[(None, None, consumer_wg_idx)])
                    _, tPvA1, _ = sm90_utils.partition_fragment_ABC(
                        thr_pv, (64, PG, Nkv), None, sP[(None, None, consumer_wg_idx)])
                    tPvA0Words = cute.recast_tensor(tPvA0, Int32); tPvA1Words = cute.recast_tensor(tPvA1, Int32)
                    for kt in cutlass.range_constexpr(pv_k_steps):
                        rPvA0Frag = tPvA0Words[None, None, kt]; rPvA1Frag = tPvA1Words[None, None, kt]
                        for p in cutlass.range_constexpr(4):
                            if cutlass.const_expr(self.bits == 4):
                                yV0 = ((rVCodeWords[kt] >> (4 * p)) & Int32(MASK4)) | unpack_magic
                                yV1 = ((rVCodeWords[v_words_per_half + kt] >> (4 * p)) & Int32(MASK4)) | unpack_magic
                            elif cutlass.const_expr(self.int2_fast_mantissa_injection_active):
                                sh = 2 * (4 * (kt % 2) + p)
                                yV0 = (((rVCodeWords[kt // 2] >> sh) << 5) & Int32(MASK2L)) | unpack_magic
                                yV1 = (((rVCodeWords[v_words_per_half + kt // 2] >> sh) << 5) & Int32(MASK2L)) | unpack_magic
                            else:
                                sh = 2 * (4 * (kt % 2) + p)
                                yV0 = ((rVCodeWords[kt // 2] >> sh) & Int32(MASK2)) | unpack_magic
                                yV1 = ((rVCodeWords[v_words_per_half + kt // 2] >> sh) & Int32(MASK2)) | unpack_magic
                            if cutlass.const_expr(self.fold):
                                rPvA0Frag[p] = yV0
                                rPvA1Frag[p] = yV1
                            elif cutlass.const_expr(not self.int2_fast_mantissa_injection_active):
                                # real V values: token pair = words a0/a1 (pr=0) vs a2/a3 (pr=1);
                                # both D-halves share the token pair
                                swV = sV2[2 * kt + p // 2]; zwV = zV2[2 * kt + p // 2]
                                rPvA0Frag[p] = _hfma2_bf16(_hsub2_bf16(yV0, unpack_magic), swV, zwV)
                                rPvA1Frag[p] = _hfma2_bf16(_hsub2_bf16(yV1, unpack_magic), swV, zwV)
                            else:
                                # int2 (4 + q feed): beta prefolded into zV2 -> single HFMA2 per word
                                swV = sV2[2 * kt + p // 2]; zwV = zV2[2 * kt + p // 2]
                                rPvA0Frag[p] = _hfma2_bf16(yV0, swV, zwV)
                                rPvA1Frag[p] = _hfma2_bf16(yV1, swV, zwV)
                    # same-thread STS->wgmma-descriptor proxy switch + cross-thread visibility
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(barrier_id=2 + consumer_wg_idx, number_of_threads=128)
                    # The same fence also orders every thread's V-code LDS above against the
                    # V warp's TMA refill, and the barrier orders all 128 fences before the
                    # lane-0 arrives, so the V release costs no fence of its own. A separate
                    # fence at the LDS cost about 1-2% of decode time on H100 (exploratory
                    # A/B, profile/h100_proxy_fence_20260924).
                    pipeline_v.consumer_release(consumer_state)
                else:
                    # THE GROUPED PV CONTRACTION, prepared here and issued in pv-mma. Build group
                    # 0: its packed words (output halves 0 to W/64-1) unpacked into the W/64
                    # register A partitions, which every later group rewrites after the previous
                    # group retired. The output halves and the lc accumulator are rescaled in
                    # pv-mma, where each group's sums join them (`pv_tile_restart`).
                    # NOT released here: the V stage is released in pv-mma, after the last
                    # group's copy (CROSS-PROXY ORDERING).
                    rVGroupWords = self._load_code_group(
                        sVCodeWords, v_code_bytes_per_thread, wg_thread_idx, 0, consumer_state.index)
                    tPvAWords = []
                    for hh in cutlass.range_constexpr(W // 64):
                        _, tPvA, tPvB = sm90_utils.partition_fragment_ABC(
                            thr_pv, (64, PG, Nkv), None, sP[(None, None, consumer_wg_idx)])
                        tPvAWords.append(cute.recast_tensor(tPvA, Int32))
                    for hh in cutlass.range_constexpr(W // 64):
                        for kt in cutlass.range_constexpr(pv_k_steps):
                            rPvAFrag = tPvAWords[hh][None, None, kt]
                            for p in cutlass.range_constexpr(4):
                                if cutlass.const_expr(self.bits == 4):
                                    rPvAFrag[p] = ((rVGroupWords[hh * v_words_per_half + kt] >> (4 * p)) & Int32(MASK4)) | unpack_magic
                                elif cutlass.const_expr(self.int2_fast_mantissa_injection_active):
                                    sh = 2 * (4 * (kt % 2) + p)
                                    rPvAFrag[p] = (((rVGroupWords[hh * v_words_per_half + kt // 2] >> sh) << 5) & Int32(MASK2L)) | unpack_magic
                                else:
                                    sh = 2 * (4 * (kt % 2) + p)
                                    rPvAFrag[p] = ((rVGroupWords[hh * v_words_per_half + kt // 2] >> sh) & Int32(MASK2)) | unpack_magic
                    # same-thread STS->wgmma-descriptor proxy switch + cross-thread visibility
                    # of sP; it does not order the V stage's release, which follows the last
                    # group's copy in pv-mma with its own fence
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(barrier_id=2 + consumer_wg_idx, number_of_threads=128)
            with annotate.range("pv-mma"):
                if cutlass.const_expr(W is None):
                    # Under the per-tile restart (`pv_tile_restart`) this tile's PV sums start
                    # from zero in tensors of their own and join the running totals after the
                    # wait, and its lc wgmma follows; every other path accumulates the PV and lc
                    # sums into the totals in place.
                    rPvSumLo = rOutAccumLo; rPvSumHi = rOutAccumHi
                    if cutlass.const_expr(self.pv_tile_restart):
                        rPvSumLo = cute.make_rmem_tensor(accum_fragment_size, Float32)
                        rPvSumHi = cute.make_rmem_tensor(accum_fragment_size, Float32)
                    for kt in cutlass.range_constexpr(pv_k_steps):
                        smem_b_desc = smem_descriptor_to_int(tPvB[None, None, kt].iterator)
                        rPvA0Mma = tPvA0Words[None, None, kt]; rPvA1Mma = tPvA1Words[None, None, kt]
                        # this k-step's A words, which the V unpack wrote; at the first step also
                        # the lc operand that every step reads and, where the sums accumulate in
                        # place, the three accumulators the max correction rescaled. The
                        # restart's sums start from zero (scale-D 0), so it passes none.
                        pv_operands = [rPvA0Mma, rPvA1Mma]
                        if cutlass.const_expr(kt == 0 and self.lc_ones_quad):
                            pv_operands.append(rLcOnes)
                        elif cutlass.const_expr(kt == 0 and not self.fold):
                            pv_operands.append(rLcOneZero)
                        if cutlass.const_expr(kt == 0 and not self.pv_tile_restart):
                            pv_operands.append(rOutAccumLo)
                            pv_operands.append(rOutAccumHi)
                            pv_operands.append(rLcAccum)
                        _wgmma_fence(*pv_operands)
                        _wgmma_n8k16_bf16(rPvSumLo, rPvA0Mma[0], rPvA0Mma[1], rPvA0Mma[2], rPvA0Mma[3], smem_b_desc,
                                          0 if self.pv_tile_restart and kt == 0 else 1, 0)
                        _wgmma_n8k16_bf16(rPvSumHi, rPvA1Mma[0], rPvA1Mma[1], rPvA1Mma[2], rPvA1Mma[3], smem_b_desc,
                                          0 if self.pv_tile_restart and kt == 0 else 1, 0)
                        if cutlass.const_expr(self.lc_ones_quad):
                            _wgmma_n8k16_bf16(rLcAccum, rLcOnes[0], rLcOnes[1], rLcOnes[2], rLcOnes[3], smem_b_desc, 1, 0)
                        elif cutlass.const_expr(not self.fold):
                            _wgmma_n8k16_bf16(rLcAccum, rLcOneZero[0], rLcOneZero[1], rLcOneZero[0], rLcOneZero[1], smem_b_desc, 1, 0)
                    _wgmma_commit(); _wgmma_wait0()
                    if cutlass.const_expr(self.pv_tile_restart):
                        # THE LC WGMMA, after the PV wgmmas have retired, so that neither its A
                        # operand nor its sums are live while they are in flight (issued with
                        # them, int4 spilled). One 16-byte load per step: lane t reads bytes
                        # [16t, 16t+16) of step kt's 128-byte row, its four A words in order. Only
                        # threads 0 to 7 load: A row r feeds only C row r, and the rows of l and c,
                        # which they hold and the epilogue reads, take exactly their factors; no
                        # other row is read. The PV sums join their totals while it runs.
                        rLcATile = cute.make_rmem_tensor((4, pv_k_steps), Int32)
                        if wg_thread_idx < 8:
                            for kt in cutlass.range_constexpr(pv_k_steps):
                                sLcRow = sLcWords[(None, wg_thread_idx, kt, consumer_wg_idx)]
                                cute.autovec_copy(cute.make_tensor(sLcRow.iterator.align(16), sLcRow.layout),
                                                  rLcATile[(None, kt)])
                        rLcTile = cute.make_rmem_tensor(accum_fragment_size, Float32)
                        for kt in cutlass.range_constexpr(pv_k_steps):
                            smem_b_desc = smem_descriptor_to_int(tPvB[None, None, kt].iterator)
                            # this k-step's A words, which the load above wrote; the sums start
                            # from zero (scale-D 0), so no accumulator is passed
                            _wgmma_fence(rLcATile[(None, kt)])
                            _wgmma_n8k16_bf16(rLcTile, rLcATile[4 * kt], rLcATile[4 * kt + 1],
                                              rLcATile[4 * kt + 2], rLcATile[4 * kt + 3], smem_b_desc,
                                              0 if kt == 0 else 1, 0)
                        _wgmma_commit()
                        for j in cutlass.range_constexpr(accum_fragment_size):
                            rOutAccumLo[j] = rOutAccumLo[j] * rMaxCorrection[j] + rPvSumLo[j]   # cols match: cS_pv[j][1] == cS_qk[j][1]
                            rOutAccumHi[j] = rOutAccumHi[j] * rMaxCorrection[j] + rPvSumHi[j]
                        _wgmma_wait0()
                        for j in cutlass.range_constexpr(accum_fragment_size):
                            rLcAccum[j] = rLcAccum[j] * rMaxCorrection[j] + rLcTile[j]
                else:
                    v_groups = D // W
                    for g in cutlass.range_constexpr(v_groups):
                        if cutlass.const_expr(g > 0):
                            # group g's words, read after group g-1 retired
                            rVGroupWords = self._load_code_group(
                                sVCodeWords, v_code_bytes_per_thread, wg_thread_idx, g,
                                consumer_state.index)
                        if cutlass.const_expr(g == v_groups - 1):
                            # the last read of the V stage was this group's copy (group 0's in
                            # PV-prepare): fence, warp barrier, release (CROSS-PROXY ORDERING),
                            # as on the K side
                            cute.arch.fence_proxy("async.shared", space="cta")
                            cute.arch.sync_warp()
                            pipeline_v.consumer_release(consumer_state)
                        if cutlass.const_expr(g > 0):
                            for hh in cutlass.range_constexpr(W // 64):
                                for kt in cutlass.range_constexpr(pv_k_steps):
                                    rPvAFrag = tPvAWords[hh][None, None, kt]
                                    for p in cutlass.range_constexpr(4):
                                        if cutlass.const_expr(self.bits == 4):
                                            rPvAFrag[p] = ((rVGroupWords[hh * v_words_per_half + kt] >> (4 * p)) & Int32(MASK4)) | unpack_magic
                                        elif cutlass.const_expr(self.int2_fast_mantissa_injection_active):
                                            sh = 2 * (4 * (kt % 2) + p)
                                            rPvAFrag[p] = (((rVGroupWords[hh * v_words_per_half + kt // 2] >> sh) << 5) & Int32(MASK2L)) | unpack_magic
                                        else:
                                            sh = 2 * (4 * (kt % 2) + p)
                                            rPvAFrag[p] = ((rVGroupWords[hh * v_words_per_half + kt // 2] >> sh) & Int32(MASK2)) | unpack_magic
                        # this group's PV sums start from zero and join the running totals after
                        # the wait (`pv_tile_restart`)
                        rPvSum = cute.make_rmem_tensor((accum_fragment_size, W // 64), Float32)
                        for kt in cutlass.range_constexpr(pv_k_steps):
                            smem_b_desc = smem_descriptor_to_int(tPvB[None, None, kt].iterator)
                            # this k-step's A words of group g, which the unpack wrote; the group's
                            # sums start from zero (scale-D 0), so no accumulator is passed
                            pv_group_operands = []
                            for hh in cutlass.range_constexpr(W // 64):
                                pv_group_operands.append(tPvAWords[hh][None, None, kt])
                            _wgmma_fence(*pv_group_operands)
                            for hh in cutlass.range_constexpr(W // 64):
                                rPvAMma = tPvAWords[hh][None, None, kt]
                                _wgmma_n8k16_bf16(rPvSum[None, hh], rPvAMma[0], rPvAMma[1],
                                                  rPvAMma[2], rPvAMma[3], smem_b_desc, 0 if kt == 0 else 1, 0)
                        _wgmma_commit(); _wgmma_wait0()
                        for j in cutlass.range_constexpr(accum_fragment_size):
                            for hh in cutlass.range_constexpr(W // 64):
                                rOutAccum[j, (W // 64) * g + hh] = \
                                    rOutAccum[j, (W // 64) * g + hh] * rMaxCorrection[j] + rPvSum[j, hh]   # cols match: cS_pv[j][1] == cS_qk[j][1]
                    # THE LC WGMMA, after the PV groups have joined their totals, so that neither
                    # its A operand nor its sums are live while a group's PV wgmmas are in flight
                    # (issued with group 0's, this body spilled). Every thread reads
                    # row t % 8 of the strip with one 16-byte load per step; A row r feeds only C
                    # row r, so the rows of l and c, which threads 0 to 7 hold and the epilogue
                    # reads, take exactly their factors, and the copies in threads 8 to 127 fill
                    # rows nothing reads. The sums start from zero and join the totals as the PV
                    # sums do.
                    rLcATile = cute.make_rmem_tensor((4, pv_k_steps), Int32)
                    for kt in cutlass.range_constexpr(pv_k_steps):
                        sLcRowTile = sLcWords[(None, wg_thread_idx % 8, kt, consumer_wg_idx)]
                        cute.autovec_copy(cute.make_tensor(sLcRowTile.iterator.align(16), sLcRowTile.layout),
                                          rLcATile[(None, kt)])
                    rLcTile = cute.make_rmem_tensor(accum_fragment_size, Float32)
                    for kt in cutlass.range_constexpr(pv_k_steps):
                        smem_b_desc = smem_descriptor_to_int(tPvB[None, None, kt].iterator)
                        # this k-step's A words, as on the whole-D body
                        _wgmma_fence(rLcATile[(None, kt)])
                        _wgmma_n8k16_bf16(rLcTile, rLcATile[4 * kt], rLcATile[4 * kt + 1],
                                          rLcATile[4 * kt + 2], rLcATile[4 * kt + 3], smem_b_desc,
                                          0 if kt == 0 else 1, 0)
                    _wgmma_commit(); _wgmma_wait0()
                    for j in cutlass.range_constexpr(accum_fragment_size):
                        rLcAccum[j] = rLcAccum[j] * rMaxCorrection[j] + rLcTile[j]
            if cutlass.const_expr(V_CH):
                # V-group drain: when this WG's NEXT own tile leaves the group (or this was its
                # last tile), fold the group-local accumulators into the globals:
                #   out += s_d ∘ pv_G + z_d · l_G     (all in this WG's current-max log2 domain)
                # l_G comes from the ones-row lc accumulator; its holder rows publish through
                # sLGroup so every thread sees its columns' l.
                if v_group_ends:
                    if wg_thread_idx < 8:
                        ev_l = Float32(0.0); od_l = Float32(0.0)
                        ev_h = Int32(0); od_h = Int32(0)
                        for j in cutlass.range_constexpr(accum_fragment_size):
                            r_, h_ = tPvCoord[j]
                            is_ev = h_ % 2 == 0
                            ev_l = ev_l + (rLcAccum[j] if is_ev else Float32(0.0))
                            od_l = od_l + (Float32(0.0) if is_ev else rLcAccum[j])
                            ev_h = h_ if is_ev else ev_h
                            od_h = od_h if is_ev else h_
                        r0_, _ = tPvCoord[0]
                        if r0_ % 8 == 0:          # the ones-row holders
                            sLGroup[(ev_h, consumer_wg_idx)] = ev_l
                            sLGroup[(od_h, consumer_wg_idx)] = od_l
                    cute.arch.barrier(barrier_id=2 + consumer_wg_idx, number_of_threads=128)
                    for j in cutlass.range_constexpr(accum_fragment_size):
                        dl_, q_ = tPvCoord[j]
                        s_lo = rVDrain[4 * j + 0]
                        z_lo = rVDrain[4 * j + 1]
                        s_hi = rVDrain[4 * j + 2]
                        z_hi = rVDrain[4 * j + 3]
                        l_g = sLGroup[(q_, consumer_wg_idx)]
                        # the V y-feed is q + BETA_FOLD (magic unpack), so the zero fold is
                        # z' = z - BETA*s; the +BETA*l term inside pv cancels exactly (the
                        # ones-gemm shares the same bf16 P operand and y is exact bf16).
                        z_lo = z_lo - Float32(self.BETA_FOLD) * s_lo
                        z_hi = z_hi - Float32(self.BETA_FOLD) * s_hi
                        rOutGlobLo[j] = rOutGlobLo[j] + s_lo * rOutAccumLo[j] + z_lo * l_g
                        rOutGlobHi[j] = rOutGlobHi[j] + s_hi * rOutAccumHi[j] + z_hi * l_g
                        rLTotal[j] = rLTotal[j] + rLcAccum[j]
            consumer_state.advance(); consumer_state.advance()       # stride 2: skip the other WG's stage
            wg_tile_idx += 1
        # ---- per-WG l/c recombine (RAW l and c — the merge applies the weights) ----
        with annotate.range("epilogue"):
            self._merge_and_store(
                V_CH,
                accum_fragment_size,
                batch_idx,
                cnt_index,
                consumer_wg_idx,
                heads_per_kv,
                merge_head_limit_dyn,
                kv_head_idx,
                mCnt,
                mLSE,
                mOp,
                mOut,
                output_scale,
                rLTotal,
                rLcAccum,
                rOutAccum,
                rOutAccumHi,
                rOutAccumLo,
                rOutGlobHi,
                rOutGlobLo,
                sMergeC,
                sMergeL,
                sMergeOut,
                sRunningMax,
                split_count,
                split_idx,
                tPvCoord,
                wg_thread_idx,
            )

    def _plane_store_offsets(self, sQ_layout, alignment):
        """`_plane_store_split` for this kernel's Q' planes, at trace time: the element offsets
        of the plane layout `sQ_layout.outer`, the byte-address swizzle `sQ_layout.inner`, and
        `alignment`, the byte alignment of the two plane bases. CuTe evaluates `crd2idx` of a
        static layout at Python integers to a Python integer, so this runs as plain Python."""
        return _plane_store_split(
            lambda h, d: cute.crd2idx((h, d), sQ_layout.outer), self.heads_per_kv,
            self.head_dim // 32, _swizzle_of(sQ_layout), BF16.width // 8, alignment)

    @cute.jit
    def _build_q_planes(
        self,
        D,
        Nkv,
        batch_idx,
        heads_per_kv,
        kv_head_idx,
        mPT,
        mQ,
        mScaleRaw,
        sBias,
        sBuildQ,
        sQ,
        sQPlane1,
        sQ_layout,
        softmax_scale,
        split_idx,
        thread_idx,
        tile_count,
        tiles_per_cta,
        warp_idx,
    ):
        """The channel-K Q'-plane builder (IO warps 10-11).

        Idle unless
        `k_axis == "channel"`. Warp 10 owns EVEN groups (plane 0), warp 11 ODD groups
        (plane 1), so the two planes build CONCURRENTLY rather than serially through one
        64-lane gang: plane building is latency-bound on dependent load, convert and store
        chains, so halving the threads per plane costs little per-plane latency while
        doubling plane concurrency.

        REGISTERS. The IO warpgroup runs at 32 registers, 29 of them for values. What the
        group loop holds per lane is the next own group's channel scale/zero (`rPfScale`,
        2 * D / 32 values), at most four plane-store lane terms (`_plane_store_split`), one
        bias partial per real head, and a few scalars (the staged-query address, sm2 and the
        loop bounds). The query block the builds multiply is staged once per CTA in `sBuildQ`
        and read back by every build, rather than held in registers or reloaded from global
        memory.
        """
        if cutlass.const_expr(self.k_axis == "channel"):
            # this request's [HQ, D] query, located once by its 64-bit request coordinate
            # (64-BIT INDEXING, `kernel`); the offsets inside it are bounded by HQ * D
            gQPlanes = mQ[(Int64(batch_idx), None, None)]
          # ===== Q'-plane builders (per-warp plane ownership; idle otherwise) =====
            # Warp 10 owns EVEN groups (plane 0), warp 11 ODD groups (plane 1): the two
            # planes build CONCURRENTLY instead of serially through one 64-thread gang
            # (transplanted from the B200 kernel: plane building is latency-shaped —
            # load, cvt and STS chains — so halving the threads per plane costs little
            # per-plane latency while doubling plane concurrency).  Each lane covers D/32
            # channels (d = lane + 32·qtr) and the bias reduce is fully warp-local —
            # the cross-warp sBiasPart round-trip and builder-internal barrier (old id
            # 11) are gone.
            # Per K token-group G: plane G%2 <- bf16(sm2·s_d·Q[h,d]) (the gemm0 B operand
            # absorbs scale AND sm2), sBias[h, G%2] <- sm2·Σ_d z_d·Q[h,d] − BETA·Σ_d
            # bf16(sm2·s_d·Q[h,d]) (the fold's whole bias, the second sum cancelling the
            # +BETA the y = q + BETA feed adds through the rounded plane values — softmax
            # fold becomes accS + bias).  Channel scales are read straight
            # from the paged pool by LDG (they bypass the TMA ring); pages of one group
            # carry duplicate copies, so any in-range page of the group serves.
            # Handshake per group g with each consumer WG that owns >= 1 tile in g:
            #   builder warp arrives RDY_wg (id 7/8 | 12/13, count 160) after the build;
            #   the WG bar.syncs RDY_wg before its first tile of g;
            #   the WG arrives FREE_wg (id 9/10 | 14/15, count 160) after its last tile;
            #   builder warp bar.syncs FREE_wg before REBUILDING its plane (g >= g0+2;
            #   the warp's previous own group is exactly g-2 — parities alternate).
            # Arrival conditions on both sides are the same range arithmetic — matched
            # pairs by construction, ragged tails included.
            builder_lane = thread_idx % 32
            builder_plane = warp_idx - 10      # warp-uniform: 0 = even groups, 1 = odd
            gtk = self.k_group // Nkv          # tiles per K group (trace-time)
            NQ = D // 32                       # channel quarters: lane l covers d = l + 32·qtr
            # GQA specialization: the h loops run exactly heads_per_kv iterations,
            # REAL head count (pad rows/bias were zero-initialized once at start)
            HPK = self.heads_per_kv
            QW = _build_q_layout(HPK, D)[0][0]  # staged bf16x2 words per (lane, quarter)
            # PLANE-STORE ADDRESSES: each store is one of at most four lane terms plus a
            # trace-time immediate (`_plane_store_split`, which checks every lane of every
            # (head, quarter) against the plane's own layout and swizzle, and needs the plane
            # bases at the swizzle's period, as `kernel` allocates them under channel-wise K).
            # The stores go through unswizzled flat views of the same planes.
            plane_base_alignment = min(sQ.iterator.alignment, sQPlane1.iterator.alignment)
            plane_stores = self._plane_store_offsets(sQ_layout, plane_base_alignment)
            sQFlat = cute.make_tensor(cute.recast_ptr(sQ.iterator, dtype=BF16),
                                      cute.make_layout(cute.cosize(sQ_layout.outer)))
            sQPlane1Flat = cute.make_tensor(cute.recast_ptr(sQPlane1.iterator, dtype=BF16),
                                            cute.make_layout(cute.cosize(sQ_layout.outer)))
            sm2b = softmax_scale * Float32(_LOG2E)
            bias_partial = cute.make_rmem_tensor(HPK, Float32)   # pre-loop alloc (DSL rmem contract)
            rPfScale = cute.make_rmem_tensor(2 * NQ, Float32)    # (s,z) x NQ quarters, one OWN group ahead
            for h in cutlass.range_constexpr(HPK):
                bias_partial[h] = Float32(0.0)
            for h in cutlass.range_constexpr(2 * NQ):
                rPfScale[h] = Float32(0.0)
            # THE QUERY BLOCK, staged once per CTA into this lane's own words of sBuildQ
            # (`_build_q_layout`): channel qd = lane + 32·qtr of every real head, two heads to
            # a bf16x2 word (the f32 -> bf16x2 repack is exact, because mQ is bf16-valued),
            # one vector store per quarter. Every build reads the words back with one vector
            # load per quarter. A lane reads only the words it wrote, so program order is
            # the only ordering the reads need.
            for qtr in cutlass.range_constexpr(NQ):
                qd = builder_lane + 32 * qtr
                rQStage = cute.make_rmem_tensor(QW, Int32)
                for w in cutlass.range_constexpr(QW):
                    if cutlass.const_expr(2 * w < HPK):
                        q_lo = gQPlanes[(kv_head_idx * heads_per_kv + 2 * w, qd)].to(Float32)
                        # an odd ratio pads the high half of its last word, which every build
                        # loads with the rest and extracts no head from
                        q_hi = (gQPlanes[(kv_head_idx * heads_per_kv + 2 * w + 1, qd)].to(Float32)
                                if 2 * w + 1 < HPK else Float32(0.0))
                        rQStage[w] = _cvt2_bf16x2(q_hi, q_lo)
                    else:
                        # padding to a power-of-two count, loaded with the rest and never extracted
                        rQStage[w] = Int32(0)
                sBuildQStore = sBuildQ[(None, builder_lane, qtr, builder_plane)]
                cute.autovec_copy(rQStage, cute.make_tensor(sBuildQStore.iterator.align(4 * QW),
                                                            sBuildQStore.layout))
            if tile_count > 0:
                first_tile = split_idx * tiles_per_cta
                last_tile = first_tile + tile_count - 1
                g0 = first_tile // Int32(gtk)
                g1 = last_tile // Int32(gtk)
                g = g0 + ((g0 + builder_plane) % 2)   # this warp's first own group
                # prefetch the first own group's channel scales; every later own group's
                # loads are issued right after the PREVIOUS build's last rPfScale read
                # (single-buffered: no snapshot registers), so the cold LDG chain — the
                # builder's dominant serial cost per crossing — hides behind the bias
                # butterfly + publish + the next FREE wait instead of sitting on the
                # consumers' critical path.  Unconditional across Gt (the per-warp
                # layout already pays the 2·D/32 prefetch registers; the old gang's Gt=256
                # inline-LDG path is slower here, because a lane's cold chain doubles
                # with per-warp ownership and nothing hides it).
                # unconditional with a CLAMPED tile: a warp owning zero groups
                # (single-group range) harmlessly reads in-range scales it never
                # uses, and `pf_page`/`d0` stay defined outside any dynamic if
                # (first assignment inside one is a DSL trap when the name was
                # pre-declared by a const_expr branch)
                pf_page = mPT[(Int64(batch_idx), cutlass.min(cutlass.max(g * Int32(gtk), first_tile), last_tile))]
                for qtr in cutlass.range_constexpr(NQ):
                    d0 = builder_lane + 32 * qtr
                    rPfScale[2 * qtr] = mScaleRaw[(Int64(pf_page), kv_head_idx, self.qoff["k_ch"] + d0)].to(Float32)
                    rPfScale[2 * qtr + 1] = mScaleRaw[(Int64(pf_page), kv_head_idx, self.qoff["k_ch"] + D + d0)].to(Float32)
                while g <= g1:
                    lo = cutlass.max(g * Int32(gtk), first_tile)
                    hi = cutlass.min((g + 1) * Int32(gtk) - 1, last_tile)
                    # WG w owns a tile in [lo, hi] iff exists t with (t - first) % 2 == w
                    span = hi - lo + 1
                    own_a = (span > 1) | ((lo - first_tile) % 2 == 0)
                    own_b = (span > 1) | ((lo - first_tile) % 2 == 1)
                    if g >= g0 + 2:            # this plane last used by own group g-2
                        plo = cutlass.max((g - 2) * Int32(gtk), first_tile)
                        phi = cutlass.min((g - 1) * Int32(gtk) - 1, last_tile)
                        pspan = phi - plo + 1
                        if (pspan > 1) | ((plo - first_tile) % 2 == 0):
                            if builder_plane == 0:
                                cute.arch.barrier(barrier_id=9, number_of_threads=160)
                            else:
                                cute.arch.barrier(barrier_id=14, number_of_threads=160)
                        if (pspan > 1) | ((plo - first_tile) % 2 == 1):
                            if builder_plane == 0:
                                cute.arch.barrier(barrier_id=10, number_of_threads=160)
                            else:
                                cute.arch.barrier(barrier_id=15, number_of_threads=160)
                    for h in cutlass.range_constexpr(HPK):
                        bias_partial[h] = Float32(0.0)
                    for qtr in cutlass.range_constexpr(NQ):
                        s_d = rPfScale[2 * qtr]
                        z_d = rPfScale[2 * qtr + 1]
                        ss = sm2b * s_d
                        # this lane's staged query words for quarter qtr, one vector load
                        rQWords = cute.make_rmem_tensor(QW, Int32)
                        sBuildQLoad = sBuildQ[(None, builder_lane, qtr, builder_plane)]
                        cute.autovec_copy(cute.make_tensor(sBuildQLoad.iterator.align(4 * QW),
                                                           sBuildQLoad.layout), rQWords)
                        for h in cutlass.range_constexpr(HPK):
                            w_q = rQWords[h // 2]
                            qv = _bf16x2_lo_f32(w_q) if h % 2 == 0 else _bf16x2_hi_f32(w_q)
                            qp = (qv * ss).to(BF16)
                            plane_elem, lane_xor = plane_stores[(h, qtr)]
                            if builder_plane == 0:
                                sQFlat[plane_elem + (builder_lane ^ lane_xor)] = qp
                            else:
                                sQPlane1Flat[plane_elem + (builder_lane ^ lane_xor)] = qp
                            # bias = sm2·Σ z_d·Q − BETA·Σ bf16(sm2·s_d·Q): the y = q + BETA
                            # feed's +BETA part rides the ROUNDED plane values, so the
                            # cancellation must subtract those same rounded products (f32
                            # z·Q against bf16 s·Q mismatches by BETA·Σ rounding — 5.6e-2
                            # at int4's BETA=128).
                            bias_partial[h] += sm2b * (z_d * qv) - Float32(self.BETA_FOLD) * qp.to(Float32)
                    # issue the next OWN group's (g+2) scale loads NOW — after this
                    # build's last rPfScale read, before the bias butterfly whose
                    # shuffle chains (plus publish + the next FREE wait) hide the
                    # LDG latency.  (g+2)*gtk <= g1*gtk <= last_tile, so the mPT
                    # read stays in range.
                    if g + 2 <= g1:
                        pf_page = mPT[(Int64(batch_idx), (g + 2) * Int32(gtk))]
                        for qtr in cutlass.range_constexpr(NQ):
                            d0 = builder_lane + 32 * qtr
                            rPfScale[2 * qtr] = mScaleRaw[(Int64(pf_page), kv_head_idx, self.qoff["k_ch"] + d0)].to(Float32)
                            rPfScale[2 * qtr + 1] = mScaleRaw[(Int64(pf_page), kv_head_idx, self.qoff["k_ch"] + D + d0)].to(Float32)
                    # bias reduce: warp-local 32-lane butterfly (each lane holds D/32
                    # channels' partials, so the full sum needs no cross-warp step)
                    for h in cutlass.range_constexpr(HPK):
                        v = bias_partial[h]
                        v += cute.arch.shuffle_sync_bfly(v, offset=1)
                        v += cute.arch.shuffle_sync_bfly(v, offset=2)
                        v += cute.arch.shuffle_sync_bfly(v, offset=4)
                        v += cute.arch.shuffle_sync_bfly(v, offset=8)
                        v += cute.arch.shuffle_sync_bfly(v, offset=16)
                        bias_partial[h] = v
                    if builder_lane == 0:
                        for h in cutlass.range_constexpr(HPK):
                            sBias[(h, builder_plane)] = bias_partial[h]
                    # STS -> wgmma-descriptor proxy switch, then publish to the consumers
                    cute.arch.fence_proxy("async.shared", space="cta")
                    if own_a:
                        if builder_plane == 0:
                            cute.arch.barrier_arrive(barrier_id=7, number_of_threads=160)
                        else:
                            cute.arch.barrier_arrive(barrier_id=12, number_of_threads=160)
                    if own_b:
                        if builder_plane == 0:
                            cute.arch.barrier_arrive(barrier_id=8, number_of_threads=160)
                        else:
                            cute.arch.barrier_arrive(barrier_id=13, number_of_threads=160)
                    g += 2

    @cute.jit
    def _tma_v_warp(
        self,
        S,
        batch_idx,
        kv_head_idx,
        mPT,
        mVq,
        pipeline_v,
        sVCodes,
        split_idx,
        tile_count,
        tiles_per_cta,
        tma_atom_v_codes,
        v_tma_cols,
    ):
        """The V TMA producer warp (IO warp 9): the V code ring only.

        V scales ride the K
        pipeline, so this warp drives one ring and nothing else.
        """
        sVCodeTmaView = cute.make_tensor(sVCodes.iterator,
                                cute.make_layout((256, v_tma_cols, S), stride=(1, 256, 256 * v_tma_cols)))
        raw_v_code_copy, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_v_codes, 0, cute.make_layout(1), mVq[(None, None, kv_head_idx, None)], sVCodeTmaView)
        copy_v_codes = copy_utils.tma_producer_copy_fn(raw_v_code_copy, pipeline_v)
        producer_state = make_pipeline_state(PipelineUserType.Producer, S)
        i = Int32(0)
        while i < tile_count:
            global_tile_idx = split_idx * tiles_per_cta + i
            page_idx = mPT[(Int64(batch_idx), global_tile_idx)]

            with annotate.range("TMA-acquire-V"):
                pipeline_v.producer_acquire(producer_state)

            with annotate.range("TMA-load-V"):
                copy_v_codes(page_idx, producer_state)
                producer_state.advance(); i += 1

    @cute.jit
    def _tma_k_warp(
        self,
        RING_ELEMS,
        S,
        batch_idx,
        k_tma_cols,
        kv_head_idx,
        mKq,
        mPT,
        mScale,
        pipeline_k,
        sKCodes,
        sKvQuantParams,
        split_idx,
        tile_count,
        tiles_per_cta,
        tma_atom_k_codes,
        tma_atom_quant_params,
    ):
        """The K TMA producer warp (IO warp 8): the K code ring plus the scale block.

        The K stage barrier
        carries codes AND the one interleaved paged scale block, so this warp issues two
        TMA ops per stage and their tx bytes accumulate on the same mbarrier. The copy
        coordinate is the GLOBAL PAGE id `mPT[(b, gt)]` for both.
        """
        # smem-side TMA views: byte-identical regroupings of the ring stages, each of which
        # receives one pool block verbatim (no swizzle; the consumer's view, in the order
        # `code_pool_order` names, is `_code_ring_words`). mKq/mScale here are the TMA coordinate
        # tensors built in __call__: codes sliced per (hk) over the GLOBAL pool -> [256, CH,
        # NP]; scales sliced per (hk) -> [N*4, NP] (the per-page contiguous block).  The
        # copy coordinate is the GLOBAL PAGE id mPT[(b, gt)] for BOTH (codes + scales paged).
        sKCodeTmaView = cute.make_tensor(sKCodes.iterator,
                                cute.make_layout((256, k_tma_cols, S), stride=(1, 256, 256 * k_tma_cols)))
        raw_k_code_copy, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_k_codes, 0, cute.make_layout(1), mKq[(None, None, kv_head_idx, None)], sKCodeTmaView)
        copy_k_codes = copy_utils.tma_producer_copy_fn(raw_k_code_copy, pipeline_k)
        # flat [ring_tot, S] ring view (legacy = token-major t*4+slot, byte-identical).
        if cutlass.const_expr(RING_ELEMS > 256):
            sQuantParamTmaView = cute.make_tensor(sKvQuantParams.iterator,
                                    cute.make_layout((128, RING_ELEMS // 128, S), stride=(1, 128, RING_ELEMS)))
            raw_quant_param_copy, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_quant_params, 0, cute.make_layout(1), mScale[(None, None, kv_head_idx, None)], sQuantParamTmaView)
        else:
            sQuantParamTmaView = cute.make_tensor(sKvQuantParams.iterator,
                                    cute.make_layout((RING_ELEMS, S), stride=(1, RING_ELEMS)))
            raw_quant_param_copy, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_quant_params, 0, cute.make_layout(1), mScale[(None, kv_head_idx, None)], sQuantParamTmaView)
        copy_quant_params = copy_utils.tma_producer_copy_fn(raw_quant_param_copy, pipeline_k)
        producer_state = make_pipeline_state(PipelineUserType.Producer, S)
        i = Int32(0)
        while i < tile_count:
            global_tile_idx = split_idx * tiles_per_cta + i   # split-K: this CTA's global tile index
            page_idx = mPT[(Int64(batch_idx), global_tile_idx)]               # GLOBAL page id (page == repacked 64-token tile)
            # acquire = arrive_and_expect_tx(codes + N*4*sizeof(sdtype)) on the stage barrier;
            # both TMA ops below complete-tx on that same barrier (no producer_commit for TMA).
            with annotate.range("TMA-acquire-K"):
                pipeline_k.producer_acquire(producer_state)

            with annotate.range("TMA-load-K"):
                copy_k_codes(page_idx, producer_state)
                if cutlass.const_expr(self.qoff["ring_tot"] > 0):
                    copy_quant_params(page_idx, producer_state)    # scales PAGED: same global page id as the codes
                producer_state.advance(); i += 1

    @cute.jit
    def _stage_q(
        self,
        D,
        PG,
        batch_idx,
        heads_per_kv,
        io_thread_idx,
        kv_head_idx,
        mQ,
        sBias,
        sQ,
        sQPlane1,
        sQRowSum,
        softmax_scale,
    ):
        """Stage Q (and, under channel-K, the per-row Q sums) into smem.

        All four IO warps
        participate; the caller keeps the `annotate.range("load-Q")` wrapper so profiling
        attribution is unchanged, and keeps the qBar barrier that follows.
        """
        # this request's [HQ, D] query, located once by its 64-bit request coordinate
        # (64-BIT INDEXING, `kernel`); the offsets inside it are bounded by HQ * D
        gQ = mQ[(Int64(batch_idx), None, None)]
        if cutlass.const_expr(self.k_axis == "token"):
            for i in cutlass.range(io_thread_idx, PG * D, 128, unroll=1):   # Q[g,D] bf16 -> sQ (pad rows>=g; NO convert)
                r, c = i // D, i % D
                sQ[(r, c)] = gQ[(kv_head_idx * heads_per_kv + r, c)] if r < heads_per_kv else BF16(0.0)
            # sQ is the gemm0 B operand, read by wgmma (async proxy): fence these STS
            # before qBar publishes them, as the channel-K pad init below does
            # (CROSS-PROXY ORDERING).
            cute.arch.fence_proxy("async.shared", space="cta")
        # sQSum[h] = Σ_d Q[h,d]: f32 accumulation over the SAME bf16 values the wgmma B operand
        # holds -> the K fold S = (s·Sq + z'·QSum)·sm is exact algebra vs the f32-acc wgmma.
        # sm2 fold (both widths): sQRowSum is stored PRE-SCALED by sm2 = sm·log2e and
        # rKScale picks up sm2 in QK-prepare, so the hot softmax fold drops its trailing
        # multiply.
        if cutlass.const_expr(self.k_axis == "channel" and self.heads_per_kv < PG):
            # one-time pad init for the specialized builder: rows [heads_per_kv, PG) of
            # BOTH planes + both sBias entries are never written again (the builder
            # loops h < heads_per_kv), so zero them here, fenced before qBar publishes
            # them to the consumers' wgmma (async-proxy) reads.
            for i in cutlass.range(io_thread_idx, (PG - self.heads_per_kv) * D, 128, unroll=1):
                r, c = self.heads_per_kv + i // D, i % D
                sQ[(r, c)] = BF16(0.0)
                sQPlane1[(r, c)] = BF16(0.0)
            if io_thread_idx < 2 * (PG - self.heads_per_kv):
                sBias[(self.heads_per_kv + (io_thread_idx % (PG - self.heads_per_kv)),
                       io_thread_idx // (PG - self.heads_per_kv))] = Float32(0.0)
            cute.arch.fence_proxy("async.shared", space="cta")
        if cutlass.const_expr(self.fold and self.k_axis == "token" and self.k_group == self.head_dim // 2):
            # tok-64: per-D-half QSums (both sm2-pre-scaled, same contract as below)
            q_sum_lo = Float32(0.0); q_sum_hi = Float32(0.0)
            if io_thread_idx < heads_per_kv:
                for head_dim_idx in cutlass.range(0, D // 2, 1, unroll=8):
                    q_sum_lo += gQ[(kv_head_idx * heads_per_kv + io_thread_idx, head_dim_idx)].to(Float32)
                    q_sum_hi += gQ[(kv_head_idx * heads_per_kv + io_thread_idx, D // 2 + head_dim_idx)].to(Float32)
            if io_thread_idx < PG:
                sQRowSum[(io_thread_idx, 0)] = q_sum_lo * (softmax_scale * Float32(_LOG2E))
                sQRowSum[(io_thread_idx, 1)] = q_sum_hi * (softmax_scale * Float32(_LOG2E))
        elif cutlass.const_expr(self.fold and self.k_axis == "token"):     # non-fold / k-channel: no rank-1 K term, sQSum dead
            q_row_sum = Float32(0.0)
            if io_thread_idx < heads_per_kv:
                for head_dim_idx in cutlass.range(0, D, 1, unroll=8):
                    q_row_sum += gQ[(kv_head_idx * heads_per_kv + io_thread_idx, head_dim_idx)].to(Float32)
            if io_thread_idx < PG:
                sQRowSum[io_thread_idx] = q_row_sum * (softmax_scale * Float32(_LOG2E))   # padded heads: 0·sm2 = 0

    @cute.jit
    def _merge_and_store(
        self,
        V_CH,
        accum_fragment_size,
        batch_idx,
        cnt_index,
        consumer_wg_idx,
        heads_per_kv,
        merge_head_limit_dyn,
        kv_head_idx,
        mCnt,
        mLSE,
        mOp,
        mOut,
        output_scale,
        rLTotal,
        rLcAccum,
        rOutAccum,
        rOutAccumHi,
        rOutAccumLo,
        rOutGlobHi,
        rOutGlobLo,
        sMergeC,
        sMergeL,
        sMergeOut,
        sRunningMax,
        split_count,
        split_idx,
        tPvCoord,
        wg_thread_idx,
    ):
        """The consumer tail: per-WG l/c recombine, then the CTA-tail merge and store.

        WG B stashes
        its unnormalized O, raw l/c and running max to smem and stops; WG A waits on the
        one `_HBAR` rendezvous, merges both partial softmaxes in the log2-max domain
        (m = max(mA, mB), w_x = exp2(m_x - m), applied to O, l and c), normalizes and
        writes out. Under `fused_combine` the last CTA to arrive also runs the in-kernel
        split reduction. Empty splits degrade gracefully: m = -1e30 gives w = 0.
        """
        if wg_thread_idx < 8:
            even_sum = Float32(0.0); odd_sum = Float32(0.0)
            even_head = Int32(0); odd_head = Int32(0)
            for j in cutlass.range_constexpr(accum_fragment_size):
                output_row, head_col = tPvCoord[j]
                is_even_head = head_col % 2 == 0
                lc_j = rLTotal[j] if cutlass.const_expr(V_CH) else rLcAccum[j]
                even_sum = even_sum + (lc_j if is_even_head else Float32(0.0))
                odd_sum = odd_sum + (Float32(0.0) if is_even_head else lc_j)
                even_head = head_col if is_even_head else even_head
                odd_head = odd_head if is_even_head else head_col
            first_output_row, _ = tPvCoord[0]
            if first_output_row % 8 == 0:                  # rcp rows -> raw l per column
                sMergeL[(even_head, consumer_wg_idx)] = even_sum
                sMergeL[(odd_head, consumer_wg_idx)] = odd_sum
            else:                              # zr rows -> raw c per column
                sMergeC[(even_head, consumer_wg_idx)] = even_sum
                sMergeC[(odd_head, consumer_wg_idx)] = odd_sum
        # B stashes its unnormalized O accumulators for the merge
        if cutlass.const_expr(consumer_wg_idx == 1):
            if cutlass.const_expr(self.group_channels is None):
                for j in cutlass.range_constexpr(accum_fragment_size):
                    output_row, head_idx = tPvCoord[j]
                    sMergeOut[(0, output_row, head_idx)] = rOutGlobLo[j] if cutlass.const_expr(V_CH) else rOutAccumLo[j]
                    sMergeOut[(1, output_row, head_idx)] = rOutGlobHi[j] if cutlass.const_expr(V_CH) else rOutAccumHi[j]
            else:
                # every output half h at sMergeOut[h, ...], (D/64, 64, PG) f32 (`kernel`)
                for j in cutlass.range_constexpr(accum_fragment_size):
                    output_row, head_idx = tPvCoord[j]
                    for h in cutlass.range_constexpr(self.head_dim // 64):
                        sMergeOut[(h, output_row, head_idx)] = rOutAccum[j, h]
        # both consumer WGs rendezvous: sMergeL/sMergeC/sMergeOut/sRunningMax published
        cute.arch.barrier(barrier_id=_HBAR, number_of_threads=256)
        # ---- CTA-tail merge (WG A only): log2-max-domain partial-softmax combine ----
        if cutlass.const_expr(consumer_wg_idx == 0):
            for j in cutlass.range_constexpr(accum_fragment_size):
                output_row, head_idx = tPvCoord[j]
                # THE ONE COMPARE that reads the runtime limit rather than
                # `heads_per_kv`, and it must stay that way: see `__call__`, where the
                # limit is built and clamped. The value has to be live across the tile
                # loop above for the fix to hold, so deriving the same quantity here
                # instead does not work -- measured, the loop bodies go back to 425/417.
                if head_idx < merge_head_limit_dyn:
                    max_a = sRunningMax[(head_idx, 0)]; max_b = sRunningMax[(head_idx, 1)]
                    merged_max = cute.arch.fmax(max_a, max_b)
                    weight_a = cute.arch.exp2(max_a - merged_max); weight_b = cute.arch.exp2(max_b - merged_max)
                    merged_l = cute.arch.fmax(sMergeL[(head_idx, 0)] * weight_a + sMergeL[(head_idx, 1)] * weight_b, Float32(1e-30))
                    merged_c = sMergeC[(head_idx, 0)] * weight_a + sMergeC[(head_idx, 1)] * weight_b
                    normalization_scale = output_scale / merged_l
                    if cutlass.const_expr(self.group_channels is None):
                        out_lo_j = rOutGlobLo[j] if cutlass.const_expr(V_CH) else rOutAccumLo[j]
                        out_hi_j = rOutGlobHi[j] if cutlass.const_expr(V_CH) else rOutAccumHi[j]
                        merged_out_lo = out_lo_j * weight_a + sMergeOut[(0, output_row, head_idx)] * weight_b + merged_c
                        merged_out_hi = out_hi_j * weight_a + sMergeOut[(1, output_row, head_idx)] * weight_b + merged_c
                        if split_count == 1:
                            mOut[(Int64(batch_idx), kv_head_idx * heads_per_kv + head_idx, output_row)] = (merged_out_lo * normalization_scale).to(cutlass.BFloat16)
                            mOut[(Int64(batch_idx), kv_head_idx * heads_per_kv + head_idx, 64 + output_row)] = (merged_out_hi * normalization_scale).to(cutlass.BFloat16)
                        else:
                            mOp[(Int64(split_idx), Int64(batch_idx), kv_head_idx * heads_per_kv + head_idx, output_row)] = merged_out_lo * normalization_scale
                            mOp[(Int64(split_idx), Int64(batch_idx), kv_head_idx * heads_per_kv + head_idx, 64 + output_row)] = merged_out_hi * normalization_scale
                    else:
                        # output half h stores channel 64h + output_row, once per head and split:
                        # the D/64 halves tile the head's D channels
                        rMergedOut = cute.make_rmem_tensor(self.head_dim // 64, Float32)
                        for h in cutlass.range_constexpr(self.head_dim // 64):
                            rMergedOut[h] = rOutAccum[j, h] * weight_a + sMergeOut[(h, output_row, head_idx)] * weight_b + merged_c
                        if split_count == 1:
                            for h in cutlass.range_constexpr(self.head_dim // 64):
                                mOut[(Int64(batch_idx), kv_head_idx * heads_per_kv + head_idx, 64 * h + output_row)] = (rMergedOut[h] * normalization_scale).to(cutlass.BFloat16)
                        else:
                            for h in cutlass.range_constexpr(self.head_dim // 64):
                                mOp[(Int64(split_idx), Int64(batch_idx), kv_head_idx * heads_per_kv + head_idx, 64 * h + output_row)] = rMergedOut[h] * normalization_scale
                    if output_row == 0:
                        # mLSE is written unconditionally: the host contract (shared with the
                        # production kernel) exposes LSE for every invocation, including
                        # single-split runs whose callers may consume it independently of the
                        # combine path.
                        mLSE[(Int64(split_idx), Int64(batch_idx), kv_head_idx * heads_per_kv + head_idx)] = merged_max * Float32(_LN2) + cute.math.log(merged_l)
            # ---- fused flash-decode combine (XQA-style last-CTA reduction) ----
            # A separate SplitKCombine launch adds a second kernel to every multi-split
            # decode; trtllm XQA also fuses its multi-block reduction into kernel_mha.
            # After WG-A publishes this split's mOp/mLSE, one elected thread bumps
            # the (batch, kv-head) split counter with acq_rel; the LAST
            # CTA to arrive re-reads all splits' partials (still L2-hot) and runs the
            # SplitKCombine online LSE-merge in-kernel, then resets the counter so the
            # buffer is reusable without host re-zeroing (CUDA-graph friendly).
            # mOp/mLSE stay written exactly as before — external combine still works.
        if cutlass.const_expr(self.fused_combine):
            # TRACE-TIME GATED. Compiling this block into a single-split kernel
            # inflates WG-A's register allocation even though the runtime guard never
            # fires, so callers enable it exactly when the scheduler plans OUTER > 1.
            # Full record: git show 3f7f180:src/cutedsl_decode/_sm90/int_decode_pingpong.py (lines 1372-1375).
            #
            # v2 (tree form): v1's one-WG serial online merge ran one DEPENDENT L2
            # round-trip per split and lost to the standalone combine at OUTER >= 32.
            # Here BOTH consumer WGs split the q-heads (A: even, B: odd) and the
            # merge is two flat passes — a global max over split LSEs, then
            # exp2-weighted sums into 4 independent partials — so the split loads
            # have no serial correction chain and pipeline 4 wide.
            rAccP = cute.make_rmem_tensor(4, Float32)
            rWsP = cute.make_rmem_tensor(4, Float32)
            rLBlk = cute.make_rmem_tensor(16, Float32)
            rOBlk = cute.make_rmem_tensor(16, Float32)
            if split_count > 1:
                if cutlass.const_expr(consumer_wg_idx == 0):
                    _fence_acq_rel_gpu()   # every WG-A thread: publish partial stores
                    cute.arch.barrier(barrier_id=2, number_of_threads=128)
                    if wg_thread_idx == 0:
                        prior_arrivals = cute.arch.atomic_add(
                            mCnt.iterator + cnt_index, Int32(1), sem="acq_rel", scope="gpu")
                        sRunningMax[(0, 0)] = Float32(1.0) if prior_arrivals == split_count - 1 else Float32(0.0)
                # both WGs rendezvous (second _HBAR phase): A-thread0's acquire and
                # the last-CTA flag happen-before every thread's read below
                cute.arch.barrier(barrier_id=_HBAR, number_of_threads=256)
                if sRunningMax[(0, 0)] > Float32(0.5):
                    log2e = Float32(_LOG2E)
                    # one (head, channel) element per thread per pass: a pass covers 128
                    # channels, the warpgroup's width, so pass p's channel is 128p + wg_thread_idx
                    # and only the head loop and the D/128 passes remain, with no div/mod.
                    q_idx = Int32(consumer_wg_idx)
                    while q_idx < heads_per_kv:
                        head = kv_head_idx * heads_per_kv + q_idx
                        # 16-wide load blocks: a dynamic while's backward branch stops
                        # ptxas hoisting loads across iterations, so every round costs a
                        # full L2 round-trip: with 4-wide rounds the merge is
                        # measurably slower at OUTER=32; 16-wide cuts the round count 4x and pass 2's
                        # LSE reloads hit L1 (same thread, same addresses as pass 1).
                        global_max = Float32(-1e30)
                        o = Int32(0)
                        while o < split_count:      # pass 1: max is associative -> 16 independent loads
                            for kk in cutlass.range_constexpr(16):
                                rLBlk[kk] = mLSE[(Int64(o + kk), Int64(batch_idx), head)] if o + kk < split_count else Float32(-1e30)
                            for kk in cutlass.range_constexpr(16):
                                global_max = cutlass.max(global_max, rLBlk[kk])
                            o += 16
                        # pass 2 and the store, once per 128 channels; pass 2 re-reads the LSE
                        # values, which pass 1 has just brought into L1
                        for p in cutlass.range_constexpr(self.head_dim // 128):
                            channel = wg_thread_idx if cutlass.const_expr(p == 0) else 128 * p + wg_thread_idx
                            for kk in cutlass.range_constexpr(4):
                                rAccP[kk] = Float32(0.0); rWsP[kk] = Float32(0.0)
                            o = Int32(0)
                            while o < split_count:      # pass 2: flat weighted sums, 4 parallel partials
                                for kk in cutlass.range_constexpr(16):
                                    inb = o + kk < split_count
                                    rLBlk[kk] = mLSE[(Int64(o + kk), Int64(batch_idx), head)] if inb else Float32(-1e30)
                                    rOBlk[kk] = mOp[(Int64(o + kk), Int64(batch_idx), head, channel)] if inb else Float32(0.0)
                                for kk in cutlass.range_constexpr(16):
                                    w = cute.arch.exp2((rLBlk[kk] - global_max) * log2e)
                                    rAccP[kk % 4] += w * rOBlk[kk]
                                    rWsP[kk % 4] += w
                                o += 16
                            acc = (rAccP[0] + rAccP[1]) + (rAccP[2] + rAccP[3])
                            weight_sum = (rWsP[0] + rWsP[1]) + (rWsP[2] + rWsP[3])
                            mOut[(Int64(batch_idx), head, channel)] = (acc / cutlass.max(weight_sum, Float32(1e-30))).to(cutlass.BFloat16)
                        q_idx += 2
                    if cutlass.const_expr(consumer_wg_idx == 0):
                        if wg_thread_idx == 0:
                            mCnt[cnt_index] = Int32(0)   # rearm for the next launch (kernel-boundary ordered)

    @cute.jit
    def _online_softmax(
        self,
        K_CH,
        K_TOK64,
        V_CH,
        consumer_wg_idx,
        lane,
        rKScale,
        rKScaleHi,
        rKZero,
        rMaxCorrection,
        rNewMax,
        rOldMax,
        rScoreAccum,
        rScoreAccum2,
        rVScale,
        sP,
        sQRowSum,
        sRunningMax,
        sWarpMax,
        tQkCoord,
        valid_token_count,
        warp_idx_in_wg,
        wg_thread_idx,
    ):
        """Online softmax in the log2 domain: mask, running max, exp2, and the rescale.

        Reads this
        tile's folded scores, applies the causal/length mask, updates the per-WG running
        max in `sRunningMax`, exponentiates in base 2 (the `sm2 = sm * log2e` convention,
        with the `mLSE * ln2` fixup at the end), and derives the per-thread correction
        factor the accumulators are rescaled by.
        """
        # The OLD max is read here, before the barrier below, because warp 0 stores the
        # new one after that barrier and nothing else orders the two. Read after it, a
        # warp that runs late loads warp 0's new max as its old one, so its accumulator
        # rows (output channels 16-63 of each half) skip the rescale while the LSE, which
        # comes from warp 0's rows, stays exact. The previous tile's store is ordered
        # before this read by the PV-prepare barrier. Reading at entry rather than just
        # before the barrier lets the max reduction hide the load; the measurements are
        # in docs/kernel/int_decode_pingpong_sm90.md ("Running-max race").
        for j in cutlass.range_constexpr(4):
            _, nj_m = tQkCoord[j]
            rOldMax[j] = sRunningMax[(nj_m, consumer_wg_idx)]
        rLog2Score = cute.make_rmem_tensor(4, Float32)
        for j in cutlass.range_constexpr(4):
            mj_, nj_ = tQkCoord[j]
            if cutlass.const_expr(self.fold and K_CH):
                # channel-wise K: sm2·s_d live in the Q' plane (gemm0 B operand) and the
                # whole bias sm2·Σ z_d·Q was hoisted to rKZero in prepare (post-RDY,
                # pre-early-FREE) — the fold is one FADD off registers.
                rLog2Score[j] = rScoreAccum[j] + rKZero[j] if mj_ < valid_token_count else Float32(-3.0e38)
            elif cutlass.const_expr(self.fold and K_TOK64):
                # per-half scales off the split accumulators; bias (2-term dot over the
                # half QSums) was pre-folded into rKZero in prepare -> 2 FFMA
                rLog2Score[j] = rScoreAccum[j] * rKScale[j] + (rScoreAccum2[j] * rKScaleHi[j] + rKZero[j]) if mj_ < valid_token_count else Float32(-3.0e38)
            elif cutlass.const_expr(self.fold):
                if cutlass.const_expr(self.bits == 2):
                    # sm2 lives in rKScale (prepare/IO pre-scale) and the full bias
                    # z'·QSum·sm2 was folded into rKZero in prepare: single FFMA off
                    # rScoreAccum — the whole score-dependent chain to exp2/max-reduce
                    # is this one op.
                    rLog2Score[j] = rScoreAccum[j] * rKScale[j] + rKZero[j] if mj_ < valid_token_count else Float32(-3.0e38)
                else:
                    # int4: sm2 lives in rKScale (prepare) and sQRowSum (IO pre-scale);
                    # z' is not pre-folded (bias overwrite is int2-only), so the fold
                    # keeps its rKZero·sQRowSum FFMA — but the trailing *sm2 leaves the
                    # score-dependent chain.
                    rLog2Score[j] = rScoreAccum[j] * rKScale[j] + rKZero[j] * sQRowSum[nj_] if mj_ < valid_token_count else Float32(-3.0e38)
            else:
                # softmax_scale_log2 prefolded into K dequant -> rScoreAccum is the score
                rLog2Score[j] = rScoreAccum[j] if mj_ < valid_token_count else Float32(-3.0e38)
        even_col_max = Float32(-3.0e38); odd_col_max = Float32(-3.0e38)
        for j in cutlass.range_constexpr(4):
            _, nj_x = tQkCoord[j]
            even_col_max = cute.arch.fmax(even_col_max, rLog2Score[j] if nj_x % 2 == 0 else Float32(-3.0e38))
            odd_col_max = cute.arch.fmax(odd_col_max, rLog2Score[j] if nj_x % 2 == 1 else Float32(-3.0e38))
        even_col_max = cute.arch.fmax(even_col_max, cute.arch.shuffle_sync_bfly(even_col_max, offset=4))
        even_col_max = cute.arch.fmax(even_col_max, cute.arch.shuffle_sync_bfly(even_col_max, offset=8))
        even_col_max = cute.arch.fmax(even_col_max, cute.arch.shuffle_sync_bfly(even_col_max, offset=16))
        odd_col_max = cute.arch.fmax(odd_col_max, cute.arch.shuffle_sync_bfly(odd_col_max, offset=4))
        odd_col_max = cute.arch.fmax(odd_col_max, cute.arch.shuffle_sync_bfly(odd_col_max, offset=8))
        odd_col_max = cute.arch.fmax(odd_col_max, cute.arch.shuffle_sync_bfly(odd_col_max, offset=16))
        if lane < 4:
            sWarpMax[(warp_idx_in_wg, 2 * lane, consumer_wg_idx)] = even_col_max
            sWarpMax[(warp_idx_in_wg, 2 * lane + 1, consumer_wg_idx)] = odd_col_max
        cute.arch.barrier(barrier_id=2 + consumer_wg_idx, number_of_threads=128)
        for j in cutlass.range_constexpr(4):
            _, nj_m = tQkCoord[j]
            tile_max = cute.arch.fmax(
                cute.arch.fmax(sWarpMax[(0, nj_m, consumer_wg_idx)], sWarpMax[(1, nj_m, consumer_wg_idx)]),
                cute.arch.fmax(sWarpMax[(2, nj_m, consumer_wg_idx)], sWarpMax[(3, nj_m, consumer_wg_idx)]))
            rNewMax[j] = cute.arch.fmax(rOldMax[j], tile_max)
            rMaxCorrection[j] = cute.arch.exp2(rOldMax[j] - rNewMax[j])   # log2 domain
        rScaledProb = cute.make_rmem_tensor(4, Float32)
        for j in cutlass.range_constexpr(4):
            p = cute.arch.exp2(rLog2Score[j] - rNewMax[j])
            if cutlass.const_expr(self.fold and not V_CH):
                rScaledProb[j] = p * rVScale[j]
            else:
                rScaledProb[j] = p          # non-fold / V-channel: P-tilde stays raw
        rProbBf16 = cute.make_rmem_tensor(4, BF16); rProbBf16.store(rScaledProb.load().to(BF16))
        for j in cutlass.range_constexpr(4):
            mj, nj = tQkCoord[j]
            sP[(nj, mj, consumer_wg_idx)] = rProbBf16[j]
        # owner stores of the per-WG running max: warp-0 lanes 0-3 own the column
        # pairs; every thread's rOldMax read precedes these through the barrier above.
        if wg_thread_idx < 4:
            for j in cutlass.range_constexpr(4):
                mj_o, nj_o = tQkCoord[j]
                sRunningMax[(nj_o, consumer_wg_idx)] = rNewMax[j]

    @cute.jit
    def _unpack_k_and_prepare(
        self,
        D,
        K_BASE,
        K_CH,
        K_STR,
        K_TOK64,
        Nkv,
        PG,
        V_BASE,
        V_CH,
        V_STR,
        consumer_state,
        consumer_wg_idx,
        k_code_bytes_per_thread,
        k_dequant_scale_log2,
        k_plane,
        lane,
        pipeline_k,
        pv_k_steps,
        qk_k_steps,
        rKScale,
        rKScaleHi,
        rKZero,
        rKZeroHi,
        rVScale,
        rVZero,
        sBias,
        sKCodeWords,
        sKvQuantParams,
        sLcFactors,
        sQ,
        sQPlane1,
        sQRowSum,
        softmax_scale_log2,
        tQkCoord,
        thr_qk,
        unpack_magic,
        wg_thread_idx,
    ):
        """Unpack K codes and build the QK gemm-0 A operand.

        The widest
        phase in the consumer: it reads this stage's packed K plane words, expands them
        with the register-resident unpack magic (one LOP3 per pair), and applies the
        prepare-phase affine fold so the wgmma A operand carries y = alpha*q + beta with
        the per-column s'/z' already folded in. Under channel-K it instead consumes the
        IO-built Q'-plane and the fold becomes an FADD.
        """
        # Three of the five OUTPUTS below are bound only under a `const_expr` guard,
        # so the `return` would raise UnboundLocalError on every other configuration.
        # Returning a value is a use like any other: the name has to exist on every
        # path. The guards, which an earlier version of this comment got wrong in both
        # cases: `tQkB1` is bound under `K_CH`, the channel-K
        # Q'-plane arm, and `sV2`/`zV2` are bound under NOT `self.fold`, the fold-off
        # dequant ablation. Naming the wrong guard here is how a later cleanup deletes
        # a pre-init that some configuration still needs.
        tQkB1 = sV2 = zV2 = None
        if cutlass.const_expr(self.group_channels is None):
            rKCodeWords = cute.make_rmem_tensor(k_code_bytes_per_thread // 4, Int32)
            sKCodeRow = sKCodeWords[(wg_thread_idx, None, consumer_state.index)]
            cute.autovec_copy(cute.make_tensor(sKCodeRow.iterator.align(16), sKCodeRow.layout), rKCodeWords)
        else:
            # The grouped body copies group 0's words only, here, where the whole-D body copies
            # its row, so their shared-memory latency still overlaps the scale reads; qk-mma
            # copies each later group after the previous group's wgmmas retired.
            rKCodeWords = self._load_code_group(sKCodeWords, k_code_bytes_per_thread, wg_thread_idx,
                                                0, consumer_state.index)
        if cutlass.const_expr(self.fold):
            for j in cutlass.range_constexpr(4):
                mj_s, _ = tQkCoord[j]
                if cutlass.const_expr(K_CH):
                    # hoist the plane's bias to registers: the softmax fold must not
                    # touch sBias after this tile's wgmma completes (early-FREE lets
                    # the builder overwrite the plane+bias as soon as we release it)
                    _, nj_b = tQkCoord[j]
                    rKZero[j] = sBias[(nj_b, k_plane)]
                if cutlass.const_expr(K_TOK64):
                    rKScale[j] = sKvQuantParams[(K_BASE + mj_s * 4, consumer_state.index)].to(Float32)
                    rKZero[j] = sKvQuantParams[(K_BASE + mj_s * 4 + 1, consumer_state.index)].to(Float32)
                    rKScaleHi[j] = sKvQuantParams[(K_BASE + mj_s * 4 + 2, consumer_state.index)].to(Float32)
                    rKZeroHi[j] = sKvQuantParams[(K_BASE + mj_s * 4 + 3, consumer_state.index)].to(Float32)
                elif cutlass.const_expr(not K_CH):
                    rKScale[j] = sKvQuantParams[(K_BASE + mj_s * K_STR, consumer_state.index)].to(Float32)
                    rKZero[j] = sKvQuantParams[(K_BASE + mj_s * K_STR + 1, consumer_state.index)].to(Float32)
                if cutlass.const_expr(not V_CH):
                    rVScale[j] = sKvQuantParams[(V_BASE + mj_s * V_STR, consumer_state.index)].to(Float32)
                    rVZero[j] = sKvQuantParams[(V_BASE + mj_s * V_STR + 1, consumer_state.index)].to(Float32)
            # NOTE: K stage NOT released yet — the V codes ride a separate pipeline but
            # this WG's V wait happens later; release after the V-phase copy like PV did
            # is wrong here (same stage index carries K for THIS tile) — release now:
            # codes in rKCodeWords, scales in rmem -> stage fully consumed.
            # Codes and scale block are both TMA-refilled, so every thread fences its
            # generic-proxy reads above, and the warp barrier orders all 32 fences
            # before lane 0's arrive (CROSS-PROXY ORDERING). Deferring this release to
            # the V release's existing fence instead measured about 10% slower at int4
            # (exploratory A/B, profile/h100_proxy_fence_20260924), whose two stages
            # leave each WG one K stage and no refill slack.
            # The grouped body still has later groups to read from this stage, so it releases
            # the stage after the last group's copy, in qk-mma, with the same three statements.
            if cutlass.const_expr(self.group_channels is None):
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_warp()
                pipeline_k.consumer_release(consumer_state)
            # affine prep in place: rKScale/rVScale -> s', rKZero -> z';
            # rVZero stays raw
            for j in cutlass.range_constexpr(4):
                if cutlass.const_expr(self.ALPHA_INV != 1.0):
                    if cutlass.const_expr(not K_CH):
                        rKScale[j] = rKScale[j] * Float32(self.ALPHA_INV)
                    if cutlass.const_expr(not V_CH):
                        rVScale[j] = rVScale[j] * Float32(self.ALPHA_INV)
                if cutlass.const_expr(K_CH):
                    continue
                if cutlass.const_expr(K_TOK64):
                    # per-half z' = z_g - beta*s_g, each consuming its RAW s'
                    rKZero[j] = rKZero[j] - Float32(self.BETA_FOLD / self.ALPHA_INV) * rKScale[j]
                    rKZeroHi[j] = rKZeroHi[j] - Float32(self.BETA_FOLD / self.ALPHA_INV) * rKScaleHi[j]
                    rKScale[j] = rKScale[j] * softmax_scale_log2
                    rKScaleHi[j] = rKScaleHi[j] * softmax_scale_log2
                    # bias pre-fold (BOTH widths): the tok-64 bias is a 2-term dot off
                    # the per-half sm2-scaled QSums — folding it here keeps the hot
                    # softmax fold at 2 FFMA (unlike token-128 int4, this changes only
                    # f32 association, not a bf16 cancellation; the 5e-3 gate arbitrates).
                    _, nj_a = tQkCoord[j]
                    rKZero[j] = rKZero[j] * sQRowSum[(nj_a, 0)] + rKZeroHi[j] * sQRowSum[(nj_a, 1)]
                    continue
                rKZero[j] = rKZero[j] - Float32(self.BETA_FOLD / self.ALPHA_INV) * rKScale[j]
                # sm2 fold (both widths): applied AFTER the z' fold (which must consume the
                # raw s').  K-side only — rVScale feeds the P-quant multiply and the
                # sLcFactors strip, stays raw.  Pairs with the sm2-pre-scaled sQRowSum
                # from the IO warpgroup.
                rKScale[j] = rKScale[j] * softmax_scale_log2
                if cutlass.const_expr(self.bits == 2):
                    # bias overwrite (int2 only): rKZero is dead after the softmax fold,
                    # and this column's sQRowSum index is a trace-time constant here —
                    # fold the score-independent bias z'·(QSum·sm2) in now, off the
                    # post-wgmma critical path.  The int2 hot fold collapses to a single
                    # FFMA off rScoreAccum.
                    _, nj_a = tQkCoord[j]
                    rKZero[j] = rKZero[j] * sQRowSum[nj_a]
            # sLcFactors rcp/zr strip (per-WG plane; its readers are this WG's own lc operand
            # loads in pv-mma).
            # The strip's physical order is `_lc_factor_layout`, and CuTe decomposes each
            # store's token subscript into it. Keep the subscript per fragment: fragments 0
            # and 2 hold tokens 8 apart, so one index shared by both would write both
            # fragments' factors to the same token.
            for j in cutlass.range_constexpr(4 if not V_CH else 0):
                mj_l, nj_l = tQkCoord[j]
                if nj_l == 0:
                    rcp_l = cute.arch.rcp_approx(cute.arch.fmax(rVScale[j], Float32(1e-30)))
                    zr_l = rVZero[j] * rcp_l - Float32(self.BETA_FOLD / self.ALPHA_INV)
                    rh_l = rcp_l.to(BF16); zh_l = zr_l.to(BF16)
                    sLcFactors[(0, mj_l, consumer_wg_idx)] = rh_l
                    sLcFactors[(1, mj_l, consumer_wg_idx)] = zh_l
                    sLcFactors[(2, mj_l, consumer_wg_idx)] = (rcp_l - rh_l.to(Float32)).to(BF16)
                    sLcFactors[(3, mj_l, consumer_wg_idx)] = (zr_l - zh_l.to(Float32)).to(BF16)
        else:
            # non-fold: dequant constants packed bf16x2 BEFORE the stage release
            # (scales ride the K ring).  K rows per A-word parity: a0/a2 = C row of
            # j=0, a1/a3 = C row of j=2 (same m16n8k16 fragment row convention the
            # repack targets).  sm2 prefolds into (s, z) so gemm0 emits log2-domain
            # scores directly.
            mK0, _ = tQkCoord[0]
            mK1, _ = tQkCoord[2]
            sK2 = cute.make_rmem_tensor(2, Int32); zK2 = cute.make_rmem_tensor(2, Int32)
            sr0 = sKvQuantParams[(mK0 * 4 + _KS, consumer_state.index)].to(Float32)
            sr1 = sKvQuantParams[(mK1 * 4 + _KS, consumer_state.index)].to(Float32)
            zk0 = sKvQuantParams[(mK0 * 4 + _KZ, consumer_state.index)].to(Float32)
            zk1 = sKvQuantParams[(mK1 * 4 + _KZ, consumer_state.index)].to(Float32)
            if cutlass.const_expr(self.int2_fast_mantissa_injection_active):
                # int2-only beta prefold (drops the per-word HSUB2): z'' = (z-4s)*sm2;
                # |z-4s| ~ |z| keeps bf16(z'') error in the ulp(v) class.  int4's
                # z-128s cancellation is the documented 1.25e-1 failure -> keeps sub, and
                # so does int2's 128 + q feed (`int2_fast_mantissa_injection=False`).
                zk0 = (zk0 - Float32(4.0) * sr0) * softmax_scale_log2
                zk1 = (zk1 - Float32(4.0) * sr1) * softmax_scale_log2
            else:
                zk0 = zk0 * softmax_scale_log2
                zk1 = zk1 * softmax_scale_log2
            sk0 = sr0 * k_dequant_scale_log2
            sk1 = sr1 * k_dequant_scale_log2
            sK2[0] = _cvt2_bf16x2(sk0, sk0); zK2[0] = _cvt2_bf16x2(zk0, zk0)
            sK2[1] = _cvt2_bf16x2(sk1, sk1); zK2[1] = _cvt2_bf16x2(zk1, zk1)
            # per-token V dequant pairs: A-word pair pr at k-step kt covers tokens
            # 16*kt + 8*pr + 2*tig + {0,1} (lo lane = even token)
            sV2 = cute.make_rmem_tensor(2 * pv_k_steps, Int32)
            zV2 = cute.make_rmem_tensor(2 * pv_k_steps, Int32)
            for kt in cutlass.range_constexpr(pv_k_steps):
                for pr in cutlass.range_constexpr(2):
                    t0 = 16 * kt + 8 * pr + 2 * (lane % 4)
                    sv0 = sKvQuantParams[(t0 * 4 + _VS, consumer_state.index)].to(Float32)
                    zv0 = sKvQuantParams[(t0 * 4 + _VZ, consumer_state.index)].to(Float32)
                    sv1 = sKvQuantParams[((t0 + 1) * 4 + _VS, consumer_state.index)].to(Float32)
                    zv1 = sKvQuantParams[((t0 + 1) * 4 + _VZ, consumer_state.index)].to(Float32)
                    if cutlass.const_expr(self.int2_fast_mantissa_injection_active):
                        zv0 = zv0 - Float32(4.0) * sv0   # beta prefold (see K note)
                        zv1 = zv1 - Float32(4.0) * sv1
                    if cutlass.const_expr(self.ALPHA_INV != 1.0):
                        sv0 = sv0 * Float32(self.ALPHA_INV)
                        sv1 = sv1 * Float32(self.ALPHA_INV)
                    sV2[2 * kt + pr] = _cvt2_bf16x2(sv1, sv0)
                    zV2[2 * kt + pr] = _cvt2_bf16x2(zv1, zv0)
            cute.arch.fence_proxy("async.shared", space="cta")   # as the fold release above
            cute.arch.sync_warp()
            pipeline_k.consumer_release(consumer_state)
        # The A operand covers the channels one wgmma group contracts, every channel of D on the
        # whole-D body; the B fragment is partitioned from the whole of sQ, so it has D/16
        # k-blocks either way (`sm90_utils.partition_fragment_ABC` sizes a register A from the
        # shape and partitions B from the tensor it is given).
        qk_channels = D if cutlass.const_expr(self.group_channels is None) else self.group_channels
        _, tQkA, tQkB = sm90_utils.partition_fragment_ABC(thr_qk, (Nkv, PG, qk_channels), None, sQ)
        if cutlass.const_expr(K_CH):
            _, _, tQkB1 = sm90_utils.partition_fragment_ABC(thr_qk, (Nkv, PG, D), None, sQPlane1)
        tQkAWords = cute.recast_tensor(tQkA, Int32)
        for kt in cutlass.range_constexpr(qk_k_steps):
            rQkAFrag = tQkAWords[None, None, kt]
            for p in cutlass.range_constexpr(4):
                if cutlass.const_expr(self.bits == 4):
                    yK = ((rKCodeWords[kt] >> (4 * p)) & Int32(MASK4)) | unpack_magic
                elif cutlass.const_expr(self.int2_fast_mantissa_injection_active):
                    t2 = (rKCodeWords[kt // 2] >> (2 * (4 * (kt % 2) + p))) << 5
                    yK = (t2 & Int32(MASK2L)) | unpack_magic
                else:
                    yK = ((rKCodeWords[kt // 2] >> (2 * (4 * (kt % 2) + p))) & Int32(MASK2)) | unpack_magic
                if cutlass.const_expr(self.fold):
                    rQkAFrag[p] = yK
                elif cutlass.const_expr(not self.int2_fast_mantissa_injection_active):
                    # real K values: (y - beta) exact (kills the bf16 z-128s
                    # cancellation), then q/alpha * (s*alpha_inv*sm2) + z*sm2
                    rQkAFrag[p] = _hfma2_bf16(_hsub2_bf16(yK, unpack_magic),
                                        sK2[p % 2], zK2[p % 2])
                else:
                    # int2 (4 + q feed): beta prefolded into zK2 -> single HFMA2 per word
                    rQkAFrag[p] = _hfma2_bf16(yK, sK2[p % 2], zK2[p % 2])

        # OUTPUTS. These five are built here and consumed by the LATER phases of the
        # same iteration (`qk-mma` reads the A words and both B descriptors,
        # `PV-prepare` reads the V scale/zero pair), so extraction has to hand them
        # back explicitly. Getting this wrong is not a compile error in the diff: the
        # caller simply raises `NameError: name 'tQkAWords' is not defined` at trace
        # time, because a name the span PRODUCES is invisible to a free-variable
        # analysis, which only ever looks at what the span CONSUMES.
        return tQkAWords, tQkB, tQkB1, sV2, zV2

    @cute.kernel
    def kernel(self, mQ, mKq, mVq, mScale, mScaleRaw, mPT, mSL, mOp, mOut, mLSE, mCnt,
               softmax_scale: Float32, output_scale: Float32, merge_head_limit_dyn: Int32,
               sQ_layout, sP_layout,
               tma_atom_k_codes, tma_atom_v_codes, tma_atom_quant_params):
        split_idx, kv_head_idx, batch_idx = cute.arch.block_idx()
        split_count, kv_head_count, _ = cute.arch.grid_dim()
        # 64-BIT INDEXING: every global offset that multiplies the request, the split or the
        # physical page is formed in 64 bits, by widening that coordinate where the tensor is
        # indexed (`Int64(batch_idx)`, `Int64(split_idx)`, `Int64(page)`), or where the query is
        # sliced to its request's rows, once, before the loops that read it. A static stride
        # times a 32-bit coordinate is a 32-bit product, and past 2^31 elements it wraps, so the
        # access lands outside its row. At HKV = 8 the channel-wise scale reads cross 2^31 at
        # page 524,288 with a 512-element scale block and inside page 699,050 with a 384-element
        # one; the query, output and partial rows cross it once batch * HQ * D does (batch 4096
        # at HQ = 4096, i.e. HKV = 512 at 8 query heads per kv head, and D = 128). Codes and
        # token-wise scales are TMA copies that take the page as a coordinate. The head, channel
        # and token-in-page terms are bounded by the tile and stay 32-bit.
        # tests/test_decode_wide_offsets_cpu.py holds every indexing site to this.
        cnt_index = Int64(batch_idx) * kv_head_count + kv_head_idx   # fused-combine split counter cell
        thread_idx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # ONE head count, and it is trace-time. This used to be a marshalled `Int32`
        # operand that an optional hint then overrode, so the same number could arrive
        # by two routes; now the builder, the epilogue and the fused combine all read
        # the same Python int and their compares and indexing constant-fold.
        heads_per_kv = self.heads_per_kv
        D, Nkv, PG, S = self.head_dim, self.page_size, self.q_head_tile, self.num_stages
        k_code_bytes_per_thread, v_code_bytes_per_thread = self.k_code_bytes_per_thread, self.v_code_bytes_per_thread    # repacked row bytes (per warpgroup thread)
        k_tma_cols, v_tma_cols = 128 * k_code_bytes_per_thread // 256, 128 * v_code_bytes_per_thread // 256   # TMA box columns: stage = (256, CH) bytes
        scale_element_bytes = self.scale_element_bytes                                   # scale element bytes (f32=4 / bf16=2)
        smem = cutlass.utils.SmemAllocator()
        mbar_k = smem.allocate_array(cutlass.Int64, num_elems=S * 2)
        mbar_v = smem.allocate_array(cutlass.Int64, num_elems=S * 2)
        # packed-code rings: a stage is one pool block verbatim, 128 * per bytes in the order
        # `code_pool_order` names; `_code_ring_words` is the consumer view.
        sKCodes = smem.allocate_tensor(cutlass.Uint8, cute.make_layout((128 * k_code_bytes_per_thread, S)),
                                       byte_alignment=128)
        sVCodes = smem.allocate_tensor(cutlass.Uint8, cute.make_layout((128 * v_code_bytes_per_thread, S)),
                                       byte_alignment=128)
        # interleaved paged scale ring: token-major [Nkv, 4, S] of sdtype (RAW s,z; folds at use).
        # 4 = (s_k, z_k, s_v, z_v) per token, contiguous in the innermost-of-stage `4` dim so one
        # TMA box (N, 4) lands densely.  K-fold reads sScale[token, 0:2]; V-fold reads [token, 2:4].
        RING_ELEMS = max(self.qoff["ring_tot"], 8)   # flat sectioned ring; legacy = 4*N (t*4+slot)
        sKvQuantParams = smem.allocate_tensor(self.scale_element_type, cute.make_ordered_layout((RING_ELEMS, S), order=(0, 1)),
                                      byte_alignment=16)
        # Under channel-wise K both Q' planes start at their swizzle's period, which the
        # builder's store addresses need (`_plane_store_split`); every other configuration
        # keeps 128 bytes and with it its shared-memory map.
        plane_alignment = _swizzle_period(_swizzle_of(sQ_layout)) if self.k_axis == "channel" else 128
        sQ = smem.allocate_tensor(BF16, sQ_layout.outer, byte_alignment=plane_alignment, swizzle=sQ_layout.inner)   # gemm0 B operand (k-channel: plane 0 = even groups)
        if cutlass.const_expr(self.k_axis == "token" and self.k_group == self.head_dim // 2):
            sQRowSum = smem.allocate_tensor(Float32, cute.make_ordered_layout((PG, 2), order=(0, 1)), byte_alignment=16)   # per-D-half QSums
        else:
            sQRowSum = smem.allocate_tensor(Float32, cute.make_layout(PG), byte_alignment=16)   # K-fold rank-1 term
        if cutlass.const_expr(self.k_axis == "channel"):
            # second Q' plane (odd groups) + per-plane bias vectors; funded by the K half of
            # the token-scale ring, which channel-wise K vacates (see qoff).
            sQPlane1 = smem.allocate_tensor(BF16, sQ_layout.outer, byte_alignment=plane_alignment, swizzle=sQ_layout.inner)
            sBias = smem.allocate_tensor(Float32, cute.make_ordered_layout((PG, 2), order=(0, 1)), byte_alignment=8)
        else:
            sQPlane1 = sQ
            sBias = sQRowSum      # unused alias keeps the helper signature uniform
        if cutlass.const_expr(self.v_axis == "channel"):
            sLGroup = smem.allocate_tensor(Float32, cute.make_ordered_layout((PG, 2), order=(0, 1)), byte_alignment=8)
        else:
            sLGroup = sQRowSum
        sP = smem.allocate_tensor(BF16, sP_layout.outer, byte_alignment=128, swizzle=sP_layout.inner)   # [PG, Nkv, 2] bf16; last dim = WG plane
        # per-warp column slots for the in-register softmax max reduction; last dim = wg plane.
        sWarpMax = smem.allocate_tensor(Float32, cute.make_ordered_layout((4, PG, 2), order=(1, 0, 2)), byte_alignment=16)
        # pingpong: per-WG running max + CTA-tail merge buffers (raw l/c per column per WG,
        # B's unnormalized O stash [half, dl, q], one 64-channel half per D/64)
        sRunningMax = smem.allocate_tensor(Float32, cute.make_ordered_layout((PG, 2), order=(1, 0)), byte_alignment=8)
        sMergeL = smem.allocate_tensor(Float32, cute.make_ordered_layout((PG, 2), order=(1, 0)), byte_alignment=8)
        sMergeC = smem.allocate_tensor(Float32, cute.make_ordered_layout((PG, 2), order=(1, 0)), byte_alignment=8)
        sMergeOut = smem.allocate_tensor(Float32, cute.make_ordered_layout((D // 64, 64, PG), order=(2, 1, 0)), byte_alignment=16)
        # l/c strip feeding each WG's own lc wgmma chain (last dim = wg plane):
        # 4 rows x 64 tokens bf16 per slot, DOUBLE-BF16 rows (probe fold_numerics lc_via_gemm:
        # the 2-row single-bf16 form FAILED — bf16(zr) re-breaks the feed's ±beta cancellation,
        # int4 1.25e-1; hi/lo split rows pass at the floor 1.700e-3/1.641e-3):
        #   row 0 = hi(rcp.approx(sV'_t))  row 2 = lo(...)   -> A m-rows 0 / 8  (l)
        #   row 1 = hi(zr_t)               row 3 = lo(...)   -> A m-rows 1 / 9  (c)
        # zr_t = zV_t*rcp − BETA_FOLD/ALPHA_INV, rcp on sV' AFTER ALPHA_INV (kernel-exact defs).
        # Physically each 16-token step is one 128-byte row in the lc operand's register order
        # (`_lc_factor_layout`), so its load is one conflict-free 16-byte load per step.
        lc_shape, lc_stride = _lc_factor_layout(Nkv)
        sLcFactors = smem.allocate_tensor(BF16, cute.make_layout(lc_shape, stride=lc_stride), byte_alignment=16)
        # REGISTER-RESIDENT UNPACK MAGIC for the unpack's (a & MASK) | MAGIC. Two 32-bit
        # immediates cannot fuse into one LOP3 (there is one immediate slot), so MAGIC is
        # loaded through smem to make it opaque to constant folding, which lets ptxas fuse
        # and(imm)+or(reg) into ONE LOP3 per pair. The inline-asm alternative halved the
        # magic LOP3s but caused heavy rematerialization and LDL/STL spills and was far slower.
        # Full record: git show 3f7f180:src/cutedsl_decode/_sm90/int_decode_pingpong.py (lines 1824-1831).
        sUnpackMagic = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=4)
        # Allocated LAST, and only where they are read, so that no buffer allocated before them
        # moves and a configuration that reads neither keeps its shared-memory map: the channel-K
        # builder's staged query words (`_build_q_layout`: 2 warps x D channels x QW 4-byte words,
        # QW <= 4, so at most 32 * D bytes, from five heads up) and the lc wgmma's ones-row constant
        # (`_lc_ones_layout`: a (ones, 0, ones, 0) row and a zeros row, 32 bytes).
        sBuildQ = sQRowSum        # unused alias keeps the helper signature uniform
        if cutlass.const_expr(self.k_axis == "channel"):
            build_q_shape, build_q_stride = _build_q_layout(self.heads_per_kv, D)
            sBuildQ = smem.allocate_tensor(Int32, cute.make_layout(build_q_shape, stride=build_q_stride), byte_alignment=16)
        sLcOnes = sQRowSum        # unused alias keeps the helper signature uniform
        if cutlass.const_expr(self.lc_ones_quad):
            lc_ones_shape, lc_ones_stride = _lc_ones_layout()
            sLcOnes = smem.allocate_tensor(Int32, cute.make_layout(lc_ones_shape, stride=lc_ones_stride), byte_alignment=16)

        # TMA pipelines (fp8-crib mechanism): producer = elect-one TMA issue (arrive_and_expect_tx is
        # elect_one inside the sync object), consumer = one warpgroup (lane 0 of each of its 4 warps
        # arrives on release). K stage barrier carries codes + the ONE interleaved paged scale block
        # (2 TMA ops, tx bytes accumulate: codes + N*4*sizeof(sdtype)); V stage carries codes only.
        producer_group = CooperativeGroup(Agent.Thread, 1)
        consumer_group = CooperativeGroup(Agent.Thread, 4)
        pipeline_k = PipelineTmaAsync.create(
            barrier_storage=mbar_k, num_stages=S, producer_group=producer_group, consumer_group=consumer_group,
            tx_count=128 * k_code_bytes_per_thread + self.qoff["ring_tot"] * scale_element_bytes, defer_sync=True)
        pipeline_v = PipelineTmaAsync.create(
            barrier_storage=mbar_v, num_stages=S, producer_group=producer_group, consumer_group=consumer_group,
            tx_count=128 * v_code_bytes_per_thread, defer_sync=True)
        if thread_idx == 0:
            sUnpackMagic[0] = Int32(MAGIC2L if self.int2_fast_mantissa_injection_active else MAGIC4)   # visibility via the barrier below
            if cutlass.const_expr(self.lc_ones_quad):
                # row 0 = (ones, 0, ones, 0) for warpgroup threads 0-3, row 1 = zeros for the
                # rest; bf16x2 (1.0, 1.0) is 0x3F803F80 (visibility via the barrier below)
                for e in cutlass.range_constexpr(4):
                    sLcOnes[(e, 0)] = Int32(0x3F803F80 if e % 2 == 0 else 0)
                    sLcOnes[(e, 1)] = Int32(0)
        cute.arch.barrier()

        # per-request tile count from seq_lens (trtllm-style ragged batching); clamped so a non-divisor OUTER
        # (or an empty last partition) can't read past this request's tiles
        seq_len = mSL[batch_idx]
        total_tiles = (seq_len + Nkv - 1) // Nkv
        tiles_per_cta = (total_tiles + split_count - 1) // split_count
        tile_count = cutlass.max(Int32(0), cutlass.min(tiles_per_cta, total_tiles - split_idx * tiles_per_cta))

        if warp_idx >= 8:                       # ===== IO warpgroup (warps 8-11, thread_idx 256-383) =====
            if cutlass.const_expr(self._min_blocks == 2):
                # pingpong: IO gives its registers to the merged consumers.  The split MUST
                # sum to an EXACT multiple of 384 within the 2-CTA pool (setmaxnregs.inc
                # stalls FOREVER if the CTA pool can't satisfy it — the r2b 24/72=56*384
                # exactness was load-bearing): 32*128 + 104*256 = 30720 = 80*384 <= 85*384.
                cute.arch.setmaxregister_decrease(32)
            io_thread_idx = thread_idx - 256
            with annotate.range("load-Q"):
                self._stage_q(
                    D,
                    PG,
                    batch_idx,
                    heads_per_kv,
                    io_thread_idx,
                    kv_head_idx,
                    mQ,
                    sBias,
                    sQ,
                    sQPlane1,
                    sQRowSum,
                    softmax_scale,
                )
            cute.arch.barrier(barrier_id=4, number_of_threads=384)   # pingpong qBar: IO + both consumers
            if warp_idx == 8:                   # K TMA producer: codes ring + the interleaved scale block
                self._tma_k_warp(
                    RING_ELEMS,
                    S,
                    batch_idx,
                    k_tma_cols,
                    kv_head_idx,
                    mKq,
                    mPT,
                    mScale,
                    pipeline_k,
                    sKCodes,
                    sKvQuantParams,
                    split_idx,
                    tile_count,
                    tiles_per_cta,
                    tma_atom_k_codes,
                    tma_atom_quant_params,
                )

            elif warp_idx == 9:                 # V TMA producer: codes ring only (scales ride the K pipeline)
                self._tma_v_warp(
                    S,
                    batch_idx,
                    kv_head_idx,
                    mPT,
                    mVq,
                    pipeline_v,
                    sVCodes,
                    split_idx,
                    tile_count,
                    tiles_per_cta,
                    tma_atom_v_codes,
                    v_tma_cols,
                )

            elif warp_idx >= 10:
              self._build_q_planes(
                  D,
                  Nkv,
                  batch_idx,
                  heads_per_kv,
                  kv_head_idx,
                  mPT,
                  mQ,
                  mScaleRaw,
                  sBias,
                  sBuildQ,
                  sQ,
                  sQPlane1,
                  sQ_layout,
                  softmax_scale,
                  split_idx,
                  thread_idx,
                  tile_count,
                  tiles_per_cta,
                  warp_idx,
              )

        elif warp_idx < 4:          # ===== PINGPONG consumer A (EVEN tiles) =====
            self._consumer_wg(0, thread_idx, seq_len, tile_count, split_idx, tiles_per_cta, split_count, kv_head_idx, batch_idx, heads_per_kv, merge_head_limit_dyn,
                              softmax_scale, output_scale, pipeline_k, pipeline_v, sUnpackMagic, sQ, sQRowSum, sKCodes, sVCodes,
                              sKvQuantParams, sP, sWarpMax, sRunningMax, sMergeL, sMergeC, sMergeOut, sLcFactors, sQPlane1, sBias, sLGroup, sLcOnes, mScaleRaw, mPT, mOp, mOut, mLSE, mCnt, cnt_index)
        elif warp_idx < 8:          # ===== PINGPONG consumer B (ODD tiles) =====
            self._consumer_wg(1, thread_idx, seq_len, tile_count, split_idx, tiles_per_cta, split_count, kv_head_idx, batch_idx, heads_per_kv, merge_head_limit_dyn,
                              softmax_scale, output_scale, pipeline_k, pipeline_v, sUnpackMagic, sQ, sQRowSum, sKCodes, sVCodes,
                              sKvQuantParams, sP, sWarpMax, sRunningMax, sMergeL, sMergeC, sMergeOut, sLcFactors, sQPlane1, sBias, sLGroup, sLcOnes, mScaleRaw, mPT, mOp, mOut, mLSE, mCnt, cnt_index)


    @cute.jit
    def __call__(self, mQ, mKq, mVq, mScale, mPT, mSL, mOp, mOut, mLSE, mCnt,
                 bmm1, bmm2, outer_splits, h_kv, batch, stream):
        # mCnt: int32 [B*HKV] split-arrival counters for the fused in-kernel combine.
        #      Allocate zeroed ONCE; the reducing CTA rearms its cell to 0 each launch.
        # SERVING-READINESS CONTRACT (paged-ONLY; serving_cache.ServingCache):
        # mQ: query tensor bf16 [B, HQ, D] — loaded by IO warps into sQ (swizzled, gemm0 B operand).
        # mKq/mVq: GLOBAL fragment-order code pools uint8 [NP, HKV, per/16, 128, 16] in the chunk16
        #      order `code_pool_order` names (repack_rs.py order="chunk16"; NO batch dim).  A
        #      row-major [NP, HKV, 128, per] pool is refused.  dim 0 = NP physical pages;
        #      mPT[(b, gt)] is the GLOBAL page id (the 64-token tile IS the page).
        # mScale: ONE interleaved PAGED scale pool, sdtype [NP, HKV, N, 4]; last dim = (s_k,z_k,s_v,
        #      z_v).  RAW token-wise quant params (x̂ = q·s + z); read through the SAME page id
        #      mPT[(b, gt)] as the codes.  sdtype f32 (default) or bf16; the fold upconverts to f32.
        #      The kernel owns the affine fold s' = ALPHA_INV·s, z' = z − BETA_FOLD·s and the sm mul.
        # mPT: int32 [B, maxPagesPerSeq] block table — logical (request, tile) -> GLOBAL page id.
        # mSL: int32 [B] per-request kv lengths (ragged batching; final tile masked in-kernel).
        # bmm1 = softmax scale (folded at use with the K fold), bmm2 = output scale.
        # The GQA ratio is NOT a launch argument: it is `heads_per_kv`, fixed at
        #      construction, and the constructor refuses a ratio above PG because the
        #      m64n8 wgmma only produces PG head-columns.  It used to arrive here as
        #      an operand, where nothing could check it and a larger ratio silently
        #      dropped the heads at and above PG.
        D, Nkv, PG = self.head_dim, self.page_size, self.q_head_tile
        sQ_layout = sm90_utils.make_smem_layout(BF16, LayoutEnum.ROW_MAJOR, (PG, D))       # gemm0 B operand (descriptor-compatible)
        sP_layout = sm90_utils.make_smem_layout(BF16, LayoutEnum.ROW_MAJOR, (PG, Nkv), 2)  # gemm1 B operand (bf16)

        # ---- TMA tensormaps: byte-identical regrouped gmem views of the GLOBAL pool ----
        # Codes: the per-(page,hk) block is ONE contiguous 128*per-byte run (NP pool, no batch
        # dim) in either code pool order, and the stage receives it verbatim; TMA's
        # 256-elem/dim box limit splits it into a (256, CH) box, tile dims leading + (HKV, NP)
        # coordinate dims trailing.  The copy's free coordinate is the GLOBAL page id.
        def _codes_view(t, per, name):
            # The shape AND the strides are checked, and the check RAISES (it runs in a plain
            # Python helper, where `python -O` cannot drop it and the DSL permits a raise): a
            # row-major pool has the same bytes per block, so only its shape tells it apart,
            # and the view below reads the pool as dense, so a transposed tensor of the right
            # shape would be read in the wrong order.
            require_code_pool(name, t.shape, self.code_pool_order, per, stride=t.stride)
            NPc, HKVc = t.shape[0], t.shape[1]
            ch = 128 * per // 256
            return cute.make_tensor(t.iterator, cute.make_layout(
                (256, ch, HKVc, NPc),
                stride=(1, 256, 256 * ch, 256 * ch * HKVc)))

        # Scales: interleaved PAGED pool [NP, HKV, N, 4] sdtype.  Per (page, hk) the N*4 block is
        # FULLY CONTIGUOUS, so the box is 1D of N*4 = 256 elems (>=16B inner box for both f32 and
        # bf16 — a 2D [N,4] box would have an 8B bf16 minor dim = illegal TMA).  Regroup to
        # [N*4, HKV, NP]; the copy's free coordinate (NP) is the GLOBAL page id (SAME as codes).
        # The 256-elem block is token-major (token n at offset n*4+s) -> lands densely in the
        # token-major sScale[token, s] = linear token*4+s smem ring.
        def _scale_view(t):
            # legacy pool [NP, HKV, N, 4] or sectioned flat pool [NP, HKV, block_tot]:
            # both flatten to a per-(hk, page) contiguous block; the TMA box covers the
            # RING sections (first ring_tot elems; channel sections are LDG-read).
            BT = self.qoff["block_tot"]
            if cutlass.const_expr(len(t.shape) == 4):
                NPc, HKVc, Nc, Q = t.shape
                assert Q == 4 and Nc == Nkv and Nc * 4 == BT, "legacy pool must be [NP, HKV, N, 4]"
                assert tuple(t.stride) == (HKVc * Nc * 4, Nc * 4, 4, 1), \
                    "scale pool must be C-contiguous (serving_cache contract)"
            else:
                NPc, HKVc, BTc = t.shape
                assert BTc == BT, f"sectioned pool last dim {BTc} != block_tot {BT} (quant_block_offsets)"
                assert tuple(t.stride) == (HKVc * BT, BT, 1), \
                    "scale pool must be C-contiguous (serving_cache contract)"
            rt = max(self.qoff["ring_tot"], 8)
            if cutlass.const_expr(rt > 256):
                # 2-D (128, rt/128) box: byte-identical regrouping of the first rt elems
                # (rt is a multiple of 128 by the ctor assert; a 1-D box caps at 256)
                return cute.make_tensor(t.iterator, cute.make_layout(
                    (128, rt // 128, HKVc, NPc), stride=(1, 128, BT, HKVc * BT)))
            return cute.make_tensor(t.iterator, cute.make_layout(
                (rt, HKVc, NPc), stride=(1, BT, HKVc * BT)))

        def _scale_raw_view(t):
            BT = self.qoff["block_tot"]
            NPc, HKVc = t.shape[0], t.shape[1]
            return cute.make_tensor(t.iterator, cute.make_layout(
                (NPc, HKVc, BT), stride=(HKVc * BT, BT, 1)))

        k_tma_cols, v_tma_cols = 128 * self.k_code_bytes_per_thread // 256, 128 * self.v_code_bytes_per_thread // 256
        op_bulk = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_k_codes, tma_k_codes = cpasync.make_tiled_tma_atom(
            op_bulk, _codes_view(mKq, self.k_code_bytes_per_thread, "mKq"),
            cute.make_layout((256, k_tma_cols), stride=(1, 256)), (256, k_tma_cols), 1)
        tma_atom_v_codes, tma_v_codes = cpasync.make_tiled_tma_atom(
            op_bulk, _codes_view(mVq, self.v_code_bytes_per_thread, "mVq"),
            cute.make_layout((256, v_tma_cols), stride=(1, 256)), (256, v_tma_cols), 1)
        # one TMA atom over the ring sections of the scale block: 1-D box up to 256
        # elems, else a (128, rt/128) 2-D box (same flat element order; the codes atoms
        # above are the in-file precedent for 2-D boxes).
        ring_box = max(self.qoff["ring_tot"], 8)
        if cutlass.const_expr(ring_box > 256):
            tma_atom_quant_params, tma_quant_params = cpasync.make_tiled_tma_atom(
                op_bulk, _scale_view(mScale),
                cute.make_layout((128, ring_box // 128), stride=(1, 128)), (128, ring_box // 128), 1)
        else:
            tma_atom_quant_params, tma_quant_params = cpasync.make_tiled_tma_atom(
                op_bulk, _scale_view(mScale),
                cute.make_layout(ring_box), (ring_box,), 1)

        # min 2 CTAs/SM, asserted in __init__; the 32/104 setmaxnregs split is compiled at 2.
        min_blocks = self._min_blocks
        # THE HEAD LIMIT FOR THE CTA-TAIL MERGE, and the ONE runtime route to the GQA
        # ratio in this kernel. It exists for a code-generation reason, not a
        # numerical one: folding that single compare against the constant costs the
        # uniform registers that hold the wgmma shared-memory descriptors, so ptxas
        # rebuilds the descriptor arithmetic inside both consumer warpgroup loops on
        # every tile (PR #65, 425/417 SASS per body against 336/335). See the
        # `heads_per_kv` section of docs/kernel/int_decode_pingpong_sm90.md.
        #
        # CLAMPED TO THE CONSTANT, so it can never admit a head the constant would
        # have masked. The constructor takes `heads_per_kv` directly and nothing
        # checks it against the tensors, so a kernel built with `heads_per_kv=4` and
        # launched with `mQ.shape[1]=64, h_kv=8` would otherwise get a limit of 8 and
        # write local heads 4..7 at the unchanged four-heads-per-group stride, over
        # the next group's outputs. That launch already contradicts itself and reads
        # the wrong Q rows, but it must not additionally corrupt a neighbour.
        #
        # Built here rather than taken as an argument because `__call__` already has
        # both operands, so no caller signature changes and no thread divides on the
        # device. Every other use of the ratio keeps the trace-time constant, which is
        # what specializes the Q'-plane builder. (2026-09-08)
        merge_head_limit_dyn = cutlass.min(Int32(mQ.shape[1]) // h_kv,
                                          Int32(self.heads_per_kv))
        self.kernel(mQ, tma_k_codes, tma_v_codes, tma_quant_params, _scale_raw_view(mScale), mPT, mSL, mOp, mOut, mLSE, mCnt,
                    bmm1, bmm2, merge_head_limit_dyn, sQ_layout, sP_layout,
                    tma_atom_k_codes, tma_atom_v_codes, tma_atom_quant_params).launch(
            grid=[outer_splits, h_kv, batch], block=[self.num_threads, 1, 1], stream=stream,
            min_blocks_per_mp=min_blocks)


def split_k_combine(op, lse, g):   # flash-decode LSE combine: op[OUTER,B,HQ,D], lse[OUTER,B,HQ] -> [B,HQ,D]
    M = lse.max(0, keepdim=True).values                 # [1,B,HQ]
    w = torch.exp(lse - M)                               # [OUTER,B,HQ]
    return (w.unsqueeze(-1) * op).sum(0) / w.sum(0).unsqueeze(-1).clamp_min(1e-30)
