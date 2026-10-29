# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SM100 quantized-KV decode with the affine dequant folded through the MMA
(`IntDecodeSm100`).

The kernel reads int4 or int2 packed KV codes and their per-token (or per-channel)
scale and zero-point streams. On the shipped path it never dequantizes an element:
a dedicated warpgroup deposits the codes into the MMA operand ring as e4m3 bytes
under a fixed affine feed ``y = alpha*q + beta``, and the scale and zero-point are
applied afterwards, in f32, to the MMA results:

* on the QK side, ``S = s_t*S_raw + (z_t - beta*s_t)*QSum_h``, where ``QSum_h`` is
  the row sum of the e4m3 Q the MMA actually sees;
* on the PV side, the V scale rides inside the P operand and the zero-point becomes
  a rank-1 correction ``c_h`` added in the epilogue.

Because the feed is exact (every code maps to an exactly representable e4m3 value),
the fold itself is exact algebra rather than an approximation: it adds no error to
the caller's own quantization of K and V. The kernel does introduce two
approximations of its own, and both are deliberate.

* P is requantized to e4m3 for the second MMA, as a hi row plus a residual lo row
  (see ``LO_SHIFT``), which is what the ``c_h`` correction and the hi/lo split are
  there to keep small.
* Under channel-wise K the query is scaled by each channel's K scale, one
  power-of-two lift per query-head row, and that scaled query Q' is rounded once to
  e4m3 for the first MMA, which keeps 3 mantissa bits. This one is not small. On the
  outlier accuracy fixture it moves the output 4.9e-2 to 1.5e-1 relative per (batch,
  KV head) away from the exact quantized reference, and at INT4, B = 1, KV = 8192 and
  K groups of 64 it raises the error against dense attention from the format's 0.081
  to 0.101. An accuracy gate therefore scores channel-wise K against a reference that
  models the rounding and reports its size beside every such row. Storing Q' as hi
  plus lo cuts that to 1.9e-3 to 4.2e-3, but fails a 1% no-slowdown budget in 206 of
  288 measured cases, with slowdowns up to 24%.

The ``fold_scale_zero=False`` ablation arm below does dequantize element-wise and is
less accurate still.

HOST CONTRACT
-------------

The kernel consumes packed pools, not dequantized tensors. ``QuantKVLayout``
describes the pool layout and ``QuantKVCache`` builds and owns the pools;
``_sm100/fold_runtime.py`` is the single compile and launch path.

* ``mKq``    uint8 [NP, HKV, 64, D*bits//8], token-major rows.
* ``mVq``    uint8 [NP, HKV, 64, D*bits//8] on the main feeds (INT4, and INT2
  on the E2M1 feed): stored exactly like ``mKq``. The INT2 E4M3 feed and the
  fold-off ablation keep channel-row-shaped [NP, HKV, D, 64*bits//8] pools.
* ``mScale`` depends on the quantization granularity, and there are two layouts:

  - LEGACY, which is ``k_quant == v_quant == ("token", head_dim)``: f32
    [NP, HKV, 64, 4], the interleaved per-token (s_k, z_k, s_v, z_v) block.
  - QUANT-FLEX, every other ``(k_quant, v_quant)`` pair: a rank-3 sectioned block
    [NP, HKV, block_tot] in the pool's ``sdtype`` (f32 or bf16, and bf16 is
    REQUIRED at K token-64 because halving the element is what funds the doubled K
    ring section). The section offsets come from
    ``quant_layout.quant_block_offsets``, and ``self.qoff["block_tot"]`` is the
    width the constructor resolved. ``__call__`` asserts both the dtype and
    ``block_tot``.

The within-word code order of both pools is MODE-KEYED: whether the store-time
permutation was applied depends on ``bits`` and the resolved
``dequant_int2_to_f4`` feed. Do not hardcode it. Ask ``k_pool_is_permuted`` for K
and ``v_pool_token_rows`` for V, keyed the same way. The pool builder asks the same
classmethods, so the two sides cannot disagree.
On the main feeds V is stored EXACTLY LIKE K: token rows, channels packed, K's
within-word order, so identical logical K and V codes give identical bytes
(``v_pool_token_rows``). A V pool in the channel-row layout has a different shape
and ``__call__`` refuses it.

The query descale multiplies the whole folded S, ``S = sm*(s_t*S_raw +
z'_t*QSum)*d_h``, so it is a per-row factor beside ``scale_s``. Two forms: a host
scalar folded into ``scale_s = sm_scale * d`` (``q_descale=None``, which compiles a
kernel with no per-row factor), or an fp32 ``q_descale`` tensor ``[B, HQ]`` the kernel
reads (any strides, so a per-tensor value can be a stride-0 expand), with ``scale_s``
left at the model's ``sm_scale``. ``q_descale`` must be positive, because the
column maximum is taken before it is applied. Pass ``scale_o = 1.0``. Do not
pre-shift z on the host either: the kernel owns ``s' = s / alpha`` and
``z' = z - beta * s'``.

The P-scale exponent ``lp = floor(log2(448 / m))``, where m is the largest s_v in
the pool, comes from ``mVMaxHist``, the cache's count table of page maxima
(``fold_runtime.v_max_hist_of``; ``QuantKVCache.v_max_hist``), and lp is taken from its
highest nonzero bin (``_pool_pscale_lp``). A one-item CTA's softmax warpgroup reads the table
while the first tile is being produced; under token-wise K its correction warpgroup reads it
too, except in the INT2 E2M1-feed split builds at four query heads per KV head, where, as
under channel-wise K, the softmax hands it to the correction through shared memory. A
persistent CTA reads it once, before the item loop, into shared memory. Channel-wise V carries no s_v in P and uses
the constant 8, and the fold-off ablation reads no exponent; neither takes the table, and both
take the 8 as the launch scalar ``pscale_lp`` (``fold_runtime.TABLE_FREE_PSCALE_LP``).

Zero-fill is the masking mechanism. An out-of-bounds page index reads ``s = z = 0``
through the same ``-1`` page-index pattern the code TMA uses, and the K-fold then
self-masks that page to ``S = 0``. Scale records at the positions of a final
partial page past the length enter ``0 * x`` products, so they must be values a
quantizer under the cache writer's input envelope could produce: the cache writer
clears them to ``s = z = 0`` when a run starts the page at position 0, and a pool
built on the host must hold such values at every position.

STRUCTURE
---------

``kernel`` is the flow: an amortized prologue, the persistent item loop, the
per-item pipeline and barrier construction, and a dispatch of one call per warp
role.

* ``__call__``              host launch (``call_graph`` for the graph split build; both
                            run ``_launch_decode``): shapes, TMA descriptors, smem/tmem sizing.
* ``_tma_qk_warp``          Q, K codes and the scale ring.
* ``_tma_vo_warp``          V codes, and the O store.
* ``_mma_kq_warp``          BMM1, S = K_operand @ Q.
* ``_mma_vp_warp``          BMM2, O = V_operand @ P.
* ``_softmax_warpgroup``    the K-fold on S, the online softmax, the hi/lo P
                            quantize and ``c_h``.
* ``_correction_warpgroup`` the Q'-plane builder, the O drain, and the V-fold
                            epilogue.
* ``_dequant_warpgroup``    code ring -> operand ring, with the byte maps.
* ``_tiled_mma_kq`` / ``_tiled_mma_vp``  the two MMA atoms, built ONCE each by
                            ``__call__`` and marshalled into ``kernel`` as
                            parameters.

The rings that connect them: the two code rings (TMA-QK warp -> the dequant K
stream, TMA-VO warp -> the dequant V stream), the two operand rings (the K stream ->
MMA-KQ warp, the V stream -> MMA-VP warp), the scale ring (TMA-QK warp -> softmax WG,
or -> both dequant streams under ``fold_scale_zero=False``), and the P and O rings
between the MMA warps and the softmax and correction warpgroups.

CROSS-PROXY ORDERING
--------------------

TMA copies and the MMA's shared-memory operand reads are async-proxy accesses, while
ordinary shared-memory loads and stores are generic-proxy ones, and neither an mbarrier
release/acquire nor a named barrier orders a same-address pair split across the two
(PTX ISA 9.4, "Proxies" and "Causality Order"). ``fence_view_async_shared``
(``fence.proxy.async.shared::cta``) is the bridge, and it acts only within the thread
that executes it, so every thread that touched the bytes fences, and the thread that
publishes them must be ordered after all of those fences:

* LDS of a TMA-filled stage, then a release to the TMA producer. The scale ring's
  readers, the softmax warpgroup or, under ``fold_scale_zero=False``, both dequant
  streams, fence, run a warp barrier and then release, because the stock
  ``PipelineTmaAsync`` release arrives from lane 0 of each warp only. Without the
  fence the softmax reads the next occupant of a scale stage. The code rings'
  readers, the dequant streams, release
  a code stage only behind the fence that already precedes an operand-ring commit:
  the first one after the stage's LDS, which may be the fence of the block before
  the one the stage feeds, since each stream loads one stage ahead. Every thread of
  the stream arrives on the stage's empty barrier itself, after its own fence, so no
  warp barrier and no separate code-release fence are needed. So that the INT2 FP4 K
  stream can free its stages mid-tile, it fences and commits after each operand block
  instead of once per tile (``merge_k_drain``).
* STS, then an async-proxy read. Every producing thread of the operand rings, the P
  ring and the channel-K Q' planes fences and then commits, so each fence precedes its
  own thread's arrival. Every correction thread fences the O tile before the
  ``sO_final`` named barrier, and the TMA-VO warp issues the store after it.
* The O store's read, then STS into the same bytes. The store reads ``sO``, which
  aliases K-ring stage 0, and its completion is a bulk async-group, not an mbarrier,
  so no fence or barrier orders the read; only a bulk-group wait in the issuing
  thread does (``cp.async.bulk.wait_group``, run here as ``.read 0`` so that it does
  not also wait for the global writes). Every lane of the TMA-VO warp commits right
  after the store. A non-persistent build waits there, before the CTA exits; a
  persistent build waits just before the next item's initialization barrier, after
  which the K deposit may write that stage, and once more after the item loop. The
  builds where ``o_direct_for`` holds have no O store to wait for: the correction
  warpgroup writes O from its registers straight to global memory, with plain stores,
  so nothing reads ``sO`` asynchronously. ``o_direct_for`` names them, from the
  configuration and the traced head counts: the split builds of token-wise V, except
  INT4 with token-wise K and eight-row query-head tiles, and four families of
  persistent INT2 E2M1-feed fold builds (token-wise K and V; channel-wise K in groups
  of 64 with token-wise V; channel-wise K and V in groups of 128 with eight-row
  query-head tiles; channel-wise K and V in groups of 64 with eight query heads per KV
  head). In every other build the O TMA descriptor exists, and the waits key on it.

SCOPE

The full set is enforced by asserts, and it is split across two places: the
constructor checks what it can see from the configuration, and ``__call__`` checks
what depends on the tensors it is handed. Read both before assuming a combination
works; what follows is the caller-relevant subset, not a transcript.

Checked by the constructor:

* ``page_size == 64``;
* ``sequence_tile % 128 == 0``;
* ``o_stages in (2, 3)``;
* ``q_head_tile >= 4``, because the P hi/lo store needs a 32-bit aligned e4m3
  vector;
* the quant-flex envelope, including the ``sdtype`` rule above, the
  ``sequence_tile % k_group`` and ``% v_group`` divisibility, and the head_dim
  restrictions each axis carries;
* the ablation arm's own scope.

Checked by ``__call__``:

* Q is Float8E4M3FN, and that dtype is what the MMA is built from. K and V are
  the uint8 CODE POOLS, not e4m3 tensors;
* ``threshold_scale_factor is None`` (BLASST is out of scope);
* ``2 * blk_tile_n <= mma_tile_n``, i.e. g <= 8, which is the hi/lo P row budget;
* ``mKq``/``mVq``/``mScale`` are all present, plus the scale-pool dtype and
  shape rules above;
* ``mVMaxHist`` is present exactly when the fold reads token-wise V, and uint32,
  and the ``pscale_lp`` launch scalar exactly when it is not;
  ``q_descale``, when present, is fp32. Their shapes (``[256]`` and ``[B, HQ]``) are
  what ``fold_runtime.compile_fold``'s fake tensors bind.

``flex_wired`` answers which non-legacy ``(k_quant, v_quant)`` pairs have a kernel
arm; the constructor raises ``NotImplementedError`` for a legal but unwired config.

DEBUG AND ABLATION MODES
------------------------

None of these is a shipping configuration.

* ``debug_s=True``           dumps tile 0's folded S, pre-mask and pre-scale.
* ``fold_scale_zero=False``  the classic-dequant ablation arm: the dequant WG
  writes real dequantized values and the whole fold is off. int2 only. Strictly
  less accurate than the fold, by construction. Each of the two concurrent K/V
  dequant streams reads its own tensor's (s, z) from the scale ring.

MODIFYING THIS KERNEL
---------------------

Two rules govern moving code into or between the warp-role methods, and both fail
in ways a diff does not show. A static role check enforces them after any extraction.

1. **A value the enclosing ``cutlass.range`` loop carries, and a callee mutates,
   fails IR verification.** The callee must RETURN the mutated value so the caller
   rebinds its own name from the return. The error is ``operand #N does not
   dominate this use`` at the loop's ``scf.yield``. Only the two MMA atoms are
   affected here; the pipeline participants are not carried, so mutating one inside
   a role is fine.
2. **A role must never REBUILD a value that reaches ``kernel`` as a marshalled
   parameter.** Rebuilding traces, runs, and is silently wrong: the marshalled
   object carries runtime fields loaded from param space, and a rebuilt one folds
   those fields to trace-time constants, so the runtime ACCUMULATE selection
   collapses and the first K tile of any item that reuses tmem is corrupted.
   Constructing in the callee also satisfies rule 1, which is what makes it a trap.

A name bound only inside a ``const_expr`` guard cannot be passed to a helper
unconditionally, because the argument is evaluated even when the guard's branch is
pruned; such names carry a ``= None`` pre-init.

``__call__``'s own docstring carries the rest of the launch contract: the Q, output,
LSE and split-K workspace tensors, which of them must be zero-initialized,
and the ``scale_s``/``scale_o`` convention.

Correctness gates: the configuration-matrix test, an end-to-end check with a
folded-S tap, and an output/PTX fingerprint pair for bit-identity across a
refactor.
"""

import math
import os
from typing import Literal, Type
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float4E2M1FN
from cutlass.cute.nvgpu import tcgen05, OperandMajorMode
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    NamedBarrier as nbar,
)

import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import (
    BFloat16,
    Float32,
    Int8,
    Int32,
    Int64,
    Uint8,
    Uint32,
    Optional,
    Union,
)

from fosca.kernel_contract import GQAPagedDecode, normalize_sdtype
from fosca.quant_layout import (
    quant_block_offsets,
    validate_quant_flex,
)
from fosca._sm100.fold_runtime import (
    PSCALE_FLOOR_BUCKET,
    VMAX_BINS,
    pscale_lp_of_bucket,
)
from fosca._sm100.int_fold_asm import (
    _cvt2_bf16x2,
    _e4m3x4_of_bf16x2,
    _fma2_bf16,
    _exit_if_negative,
    _l2_prefetch,
    _lift_exp,
    _prmt_r,
    _sub2_bf16,
    _umma_ss_f8f6f4,
    _tid_x_fresh,
    _opaque_i32,
)
from fosca._vendor.flash_attn.cute import mma_sm100_desc as sm100_desc
from fosca._sm100.gqa_decode import (
    # Kernel invariants
    mma_modes,
    mma_dice,
    warp_threads,
    warpgroup_warps,
    warpgroup_threads,
    max_reduction_iters,
    # Math helpers
    log2_e,
    exp2,
    warp_fmax,
    smem_fmax,
    # Kernel code
    GroupedQueryAttentionDecode as GqaDecode,
)

# Math helpers
warp_or = partial(cute.arch.warp_redux_sync, kind="or")  # skip predicate reduction


# (there is no _lp_of_runmax: the pscale is pool-wide, lp = floor(log2(448 /
# max s_v)), read from the cache's count table (`_pool_pscale_lp`); the hi/lo
# rows are certified at that exponent.)

# Debug helpers
debug_blasst = False
gmem_add = partial(cute.arch.atomic_add, sem="relaxed", scope="gpu")  # count skips


#: fold-off int2 bf16 feed (same primitive as the sm90 arm's MAGIC2L): the linear
#: magic yields y = 4 + q EXACTLY in bf16 — 0x4080 is 4.0 and injecting q at
#: mantissa bits 5-6 adds exactly q, since ulp(4.0) in bf16 is 4*2^-7 and q<<5 is
#: 32q. The magic base is then SUBTRACTED (`sub.rn.bf16x2`, exact by Sterbenz:
#: 4..7 minus 4) so the FMA evaluates the reference's own expression `q*s + z` on
#: natural-magnitude coefficients.
#:
#: NOT prefolded into `z' = z - 4*s`, which is one instruction cheaper. Prefolding
#: inflates |z| ~3 to |z'| ~11, so rounding the COEFFICIENTS to bf16 costs error
#: comparable to the e4m3 quantum itself rather than well below it, and an ablation
#: baseline must not be gratuitously inaccurate any more than it may be gratuitously
#: fast.
#:
#: This 4 is unrelated to ``BETA_FOLD[2]``: that 8 belongs to the FOLD path's
#: e4m3(8+q) feed, where 8..11 is the codes' exactly-representable window.
MAGIC2L = 0x40804080  # bf16x2 (4.0, 4.0) — the magic base, also the subtrahend
MASK2L_LO = 0x00000060  # low  lane's mantissa bits 5-6
MASK2L_HI = 0x00600000  # high lane's mantissa bits 5-6


# e4m3 byte encodings of the raw codes q = 0..15 (all exact in e4m3). Hardcoded so this
# module stays torch-free; asserted equal to torch's float8_e4m3fn bytes by
# tests/test_e4m3_code_table_cpu.py.
_E4M3_CODES = (0x00, 0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E,
               0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57)
_T32 = lambda b: b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)
LUT2C8 = _T32(_E4M3_CODES[8:12])  # int2 K table: e4m3(8+q), q in 0..3 = 0x50..0x53

# Every stage count this kernel exposes is timed against its default at a
# configuration the planner selects, and none reaches one percent: `p_stages=5`
# reads 0.3 percent faster than the shipped 4 and the default stays at 4
# deliberately. Time a stage-count change as two arms of one benchmark YAML,
# compared pairwise, before changing a default.
#
# ``sp_stages`` is not a knob at all: the block it sizes is allocated only when
# ``threshold_scale_factor is not None``, which no shipped configuration passes.
class IntDecodeSm100(GQAPagedDecode):
    # Fold feed constants. The dequant WG feeds e4m3(alpha*q + BETA_FOLD[bits])
    # raw-code bytes to the UMMA (int4: the raw code BYTE, alpha = 2**-9, see
    # ALPHA_INV_INT4; int2: 8+q via OR-magic or the 0x50..0x53 table, alpha = 1),
    # so the affine fold uses s' = s / alpha and z' = z - BETA_FOLD[bits] * s'.
    BETA_FOLD = {4: 0.0, 2: 8.0}
    # The int4 operand byte IS the raw 4-bit code. e4m3's subnormal band and its
    # first normal binade share one ulp of 2**-9, so byte patterns 0x00..0x0F
    # decode to q * 2**-9 exactly: the feed is alpha*q with alpha = 2**-9 and
    # beta = 0, and the fold rescales s' = ALPHA_INV_INT4 * s. Codes 1..7 are
    # subnormal operand bytes, which the tcgen05 kind::f8f6f4 datapath honours on
    # the A role (silicon-verified).
    #
    # The rescale is applied once per consumer, never twice: token-K at the
    # scale-quad copy, token-V inside the P operand paired with the in-kernel
    # lp - 9 shift, channel-V in the drain's s'_d (vch_s_mul), and
    # channel-K in the PUBLISHED LIFT rather than s'_d (kch_lift_mul, because the
    # po2 ladder clamp leaves no +9 headroom).
    #
    # Every factor is an exact f32 exponent shift, so on scales this feed is
    # DEFINED for, output is bit-identical to the e4m3(q) feed. Two bounds
    # come with it:
    #   * a scale at or beyond 2**119 overflows its 512x copy to inf (512 * 2**119
    #     is 2**128, past the largest f32). The serving cache writer guards it: it
    #     saturates every K and V element to [-2**64, 2**64] before it quantizes
    #     (`_append_core.INPUT_ENVELOPE`), so a scale's exact value is at most
    #     (2**64 + 2**64) / qmax = 2**65 / 15 for INT4; the f32 division and a bf16
    #     pool's rounding can carry the stored value slightly above that (bf16 to
    #     about 1.0034 times it), still below 2**62, whatever the input, NaN
    #     included (`fmax` drops a NaN). The packaged pool builders do not: they
    #     clamp s only from BELOW (at 1e-8) and take arbitrary bf16 inputs, so a
    #     pool built there must keep its scales below 2**119 itself, which no
    #     realistic test input comes near.
    #   * far below 1.0 the +1e-30 epsilon in c_h's rcp stops being negligible
    #     against 512*s rather than against s. The 1e-8 lower clamp, in the writer
    #     and in the pool builders, defends this one.
    ALPHA_INV_INT4 = 512.0
    # The int2-to-f4 feed (dequant_int2_to_f4): the K feed AND the V feed
    # become packed e2m1 nibbles read straight from SMEM by the f8f6f4 UMMA —
    # nibble == the 2-bit code, which decodes to q/2 EXACTLY (0, .5, 1, 1.5
    # are all e2m1 points). The fold absorbs the /2 into the per-token scales
    # (s' = 2*s_t for K, s'_v = 2*s_v riding inside the P operand for V, each
    # applied ONCE at the scale-quad register copy) and BOTH betas are 0
    # (z' = z_t, z'_v = z_v), an exact algebraic identity with the e4m3(8+q)/
    # BETA_FOLD[2]=8 feed. The launch-global pscale exponent shifts by -1
    # in-kernel so the quantized P hi/lo bytes stay bit-identical.
    BETA_FOLD_F4 = 0.0
    # P lo-row scale: P_lo = e4m3((w*pscale - float(P_hi)) * 2^LO_SHIFT). This is
    # the certified reference's dynamic ps2 = 2^floor(log2(448/max|res|)) pinned at
    # its residual ulp bound: hi spans up to 448 = 1.75*2^8, so
    # |w*pscale - hi| <= 2^(8-4) = 16 and floor(log2(448/16)) = 4, i.e. lo equals
    # the reference's lo row exactly whenever the tile's max residual is ulp-limited.
    # Every tile is: max P = 1 by online-softmax construction, so max w*pscale lands
    # in (224, 448]. A fixed shift avoids a second per-tile cross-token reduction
    # (the dynamic max|res| scan); worst case it gives up log2(max_res_bound/max_res)
    # lo-bits against the reference, and the certified headroom is ~6e-4 against a
    # 5e-3 gate.
    LO_SHIFT = 4

    @staticmethod
    def deposit_lane_offsets(t, ring_row, row_bytes, run_bytes, uniform_offsets):
        """The two per-lane byte offsets of an INT4 dequant-stream thread's deposit into
        a swizzled e4m3 operand stage in shared memory.

        The stage takes the swizzle whose span is `row_bytes`, the bytes one code page
        contributes to one row of the stage: 128 for INT4 K (K_SW128, a row is one
        token's channels) and the same 128 for INT4 V, which is stored exactly like K, so
        its MN-major stage (MN_SW128) has K's rows, bytes and swizzle. Stream thread `t`
        (0 to 63) owns the 16-byte code chunks `t + 64 q` of each code page, one per run
        `q`, at both head tiles; run `q` expands to `run_bytes` contiguous operand bytes
        at page row
        `t // cpr + q * 64 // cpr` and byte `run_bytes * (t % cpr)` of that row, where
        `cpr = row_bytes // run_bytes`. The swizzle XORs address bits [4, 4 + B) with
        bits [7, 7 + B) of the ABSOLUTE shared-memory address, with
        `16 << B == row_bytes`, so it repeats every `8 * row_bytes` bytes. The stage,
        page and run offsets are multiples of that period (`uniform_offsets` lists them,
        and each is checked), so the whole address is

            stage base + page offset + run offset + lane_offsets[m]

        for the 16-byte half `m` of the run, and only `lane_offsets` depends on the
        thread. `ring_row` is address bit 7 upward of the operand ring's start, which is
        128-byte aligned; only its low B bits matter.

        `t` and `ring_row` may be Python ints (the CPU lane-map test,
        `tests/test_sm100_dequant_banks_cpu.py`) or DSL Int32 values (the kernel, where
        the offsets are computed once per thread, outside the tile loop). Shifts and
        masks rather than `//` and `%` keep the arithmetic the same for both.
        """
        for name, n in (("row_bytes", row_bytes), ("run_bytes", run_bytes)):
            if type(n) is not int or n <= 0:
                raise ValueError(
                    f"deposit_lane_offsets: {name} must be a positive int, got {n!r}")
        if row_bytes not in (32, 64, 128):
            raise ValueError(
                f"deposit_lane_offsets: no K-major swizzle spans {row_bytes} bytes")
        if run_bytes % 16 or row_bytes % run_bytes:
            raise ValueError(
                f"deposit_lane_offsets: a {run_bytes}-byte run does not tile a "
                f"{row_bytes}-byte row in 16-byte halves")
        cpr = row_bytes // run_bytes
        if cpr & (cpr - 1):
            raise ValueError(
                f"deposit_lane_offsets: {cpr} runs per row is not a power of two")
        for off in uniform_offsets:
            if type(off) is not int or off % (8 * row_bytes):
                raise ValueError(
                    f"deposit_lane_offsets: offset {off!r} is not a multiple of the "
                    f"swizzle period {8 * row_bytes}, so it would change the swizzle key")
        line = (t >> (cpr.bit_length() - 1)) * row_bytes
        key = (ring_row + (line >> 7)) & (row_bytes // 16 - 1)
        col16 = (t & (cpr - 1)) * (run_bytes // 16)
        return [line + ((col16 + m) ^ key) * 16 for m in range(run_bytes // 16)]

    @classmethod
    def flex_wired(cls, k_quant, v_quant, head_dim=128):
        """Which non-legacy (k_quant, v_quant) configs have their kernel arm
        implemented. The ctor raises NotImplementedError for a legal-but-unwired
        config."""
        k_ok = k_quant in (("token", head_dim), ("token", head_dim // 2)) or (
            k_quant[0] == "channel" and k_quant[1] in (64, 128, 256)
        )
        v_ok = v_quant == ("token", head_dim) or (
            v_quant[0] == "channel" and v_quant[1] in (64, 128, 256)
        )
        legacy = (k_quant == ("token", head_dim)
                  and v_quant == ("token", head_dim))
        return k_ok and v_ok and not legacy

    @classmethod
    def resolve_dequant_int2_to_f4(cls, bits, dequant_int2_to_f4=None, *,
                                   fold_scale_zero=True):
        """The f4 feed is the int2 DEFAULT, and this is the single resolver that
        says so. `None` (kwarg absent) resolves to True for bits==2 unless the
        fold-off ablation (fold_scale_zero=False) is set; an explicit value
        always wins (explicit 1 with the fold off still trips the ctor assert).
        The pool-keying path calls this same resolver, so kernel mode and pool
        perm cannot disagree under defaults."""
        if dequant_int2_to_f4 is not None:
            return bool(int(dequant_int2_to_f4))
        return bool(bits == 2 and bool(int(fold_scale_zero)))

    @classmethod
    def k_pool_is_permuted(cls, bits, *, dequant_int2_to_f4=None,
                           k_quant=("token", 128),
                           fold_scale_zero=True):
        """Whether the K pool for this configuration carries the store permutation.

        THE LAYOUT IS A PROPERTY OF THE CONFIGURATION, NOT A CHOICE. There is no
        override and there is no ctor kwarg: the permuted order is what the f4 K
        deposit reads, so a caller who could ask for one independently of the
        other could only ask for a wrong answer.

        True for int4 at EVERY configuration, and for int2 with the f4 feed at
        either K axis. int4 has no feed term: the raw-code two-mask deposit is
        unconditional (no flag resolves it off, and `fold_scale_zero=False`
        is ctor-refused at this width), so the pool order is unconditional with
        it, and the store shuffle is the 8-slot form. The int2 False cases read
        the logical pack order, because the E4M3 feed gathers on nibble spacing,
        and int2's feed term matters: `resolve_dequant_int2_to_f4` is off under
        the fold-off arm, which then reads logical order.

        THERE IS NO AXIS TERM. Channel-wise K takes the two-mask deposit like
        everything else on this feed, at 3 ALU operations per 16 codes instead
        of 7. The two deposits emit byte-identical operand words for arbitrary
        codes with no scale involved, so the permutation is absorbed entirely
        between the pool and the operand stage and a channel-wise fold sees
        exactly the operands it saw before.

        Whoever writes the pool calls THIS. A permuted pool read by a gather, or
        a logical pool read by two masks, is silently wrong at every K channel
        and raises nothing, so one predicate on both sides is what makes the
        pairing unreachable rather than merely documented. The serving cache
        writer `QuantAppendSm100` writes the order `QuantKVLayout` records from
        this predicate, and `docs/K_STORE_PERM.md` is the specification a writer
        outside the library has to meet.

        Worth about 1% of decode latency at B >= 64 and nothing below, against
        57% less K construction ALU work, with bitwise-identical output."""
        if bits == 4:
            return True
        f4 = cls.resolve_dequant_int2_to_f4(
            bits, dequant_int2_to_f4,
            fold_scale_zero=fold_scale_zero)
        return bool(bits == 2 and f4)

    @classmethod
    def v_pool_token_rows(cls, bits, *, dequant_int2_to_f4=None,
                          fold_scale_zero=True):
        """Whether the V pool is stored exactly like the K pool.

        True means token rows with channels packed, in K's within-word order,
        so identical logical K and V codes give identical pool bytes, shape and
        strides. The V operand is then MN-major, read from a shared-memory image
        that equals the K operand's image with the roles of rows and columns
        swapped, so the V stream runs the K stream's deposit.

        True for INT4 at every configuration and for INT2 on the E2M1 feed: the
        two main feeds. The INT2 E4M3 feed and the fold-off ablation keep their
        token-major channel-row-shaped pool.

        Whoever writes a V pool asks THIS, as the K writer asks `k_pool_is_permuted`:
        the answer is the store order, and `QuantKVLayout` records it as
        `v_pool_permuted`, which the cache's shapes and the serving writer follow.
        Nothing in a pool's bytes says which layout it holds, so one predicate on
        both sides keeps writer and reader paired.
        """
        if bits == 4:
            return True
        return bool(bits == 2 and cls.resolve_dequant_int2_to_f4(
            bits, dequant_int2_to_f4, fold_scale_zero=fold_scale_zero))

    @classmethod
    def v_pool_rows(cls, page_size, head_dim, bits, *, dequant_int2_to_f4=None,
                    fold_scale_zero=True):
        """(rows, row_bytes) of one (page, kv-head) slab of the V pool."""
        if cls.v_pool_token_rows(bits, dequant_int2_to_f4=dequant_int2_to_f4,
                                 fold_scale_zero=fold_scale_zero):
            return page_size, head_dim * bits // 8
        return head_dim, page_size * bits // 8

    #: (sequence_tile, o_stages) for the quant configs where the shipped (256, 3)
    #: is measurably wrong. Everything absent from this table keeps (256, 3), the
    #: value every measured flex configuration uses.
    #:
    #: Keyed on the FULL (k_quant, v_quant) pair, not on v_axis, and that is
    #: load-bearing in three separate ways:
    #:   * V-ch256 CANNOT take 128 -- it fails the `sequence_tile % v_group`
    #:     assert below (128 % 256).
    #:   * tile 128 made V-ch128 21.2% WORSE, so channel-V alone is the wrong
    #:     predicate for the tile.
    #:   * K-ch128/V-ch128 is BOTH axes; o_stages feeds the channel-K Q'-plane
    #:     ring depth (`(o_stages + 2) * k_parts`) and the builder's kb_ahead
    #:     window, so o_stages=2 there is a different, unmeasured kernel.
    #: Deliberately bits-blind: at bits=4 the tile change also halves scale_stages,
    #: which moves kv_stages. int2 is measured; if int4 regresses this table grows
    #: a bits dimension.
    QUANT_TILING = {
        (("token", 128), ("channel", 64)): (128, 3),
        (("token", 128), ("channel", 128)): (256, 2),
    }

    @classmethod
    def resolve_tiling(cls, k_quant, v_quant, *,
                       sequence_tile=None, o_stages=None):
        """Per-quant-config ``(sequence_tile, o_stages)``.

        ``None`` (kwarg absent) resolves through ``QUANT_TILING``, defaulting to
        ``(256, 3)``; an explicit value ALWAYS wins, so the ``--seq-tile`` /
        ``--o-stages`` A/B arms keep forcing what they are told. Same
        resolver-classmethod shape as ``resolve_dequant_int2_to_f4`` and for the
        same reason: two entry points must not be able to disagree under
        defaults. Normalizes the pair itself (mirroring ``validate_quant_flex``)
        because the ctor resolves BEFORE it validates.
        """
        key = ((k_quant[0], int(k_quant[1])), (v_quant[0], int(v_quant[1])))
        st, ost = cls.QUANT_TILING.get(key, (256, 3))
        return (st if sequence_tile is None else int(sequence_tile),
                ost if o_stages is None else int(o_stages))

    def o_direct_for(self, grouped_heads, has_q_descale):
        """Whether the correction warpgroup stores O straight to global memory.

        ``grouped_heads`` is the number of query heads per KV head of the traced query
        tensor and ``has_q_descale`` whether the trace has a query-descale tensor; both are
        trace-time constants, so the answer is a constant of the compiled kernel. Where it
        holds, the correction warpgroup writes each element of O it holds from its
        registers to global memory with a plain store, and nothing is staged in ``sO``, no
        O TMA descriptor is built, and there is no O-store read to wait for. Every other
        build stages O in ``sO`` (K-ring stage 0) and the TMA-VO warp stores it.
        """
        # Where the correction warpgroup writes O from its registers straight to global
        # memory (`_correction_warpgroup`), with no staging in shared memory, no TMA store
        # and so no wait for that store's read: the split builds of token-wise V (a split's
        # partial output), and these persistent builds of the INT2 E2M1-feed fold: token-wise
        # K and V; channel-wise K in groups of 64 tokens with token-wise V; and channel-wise K
        # and V in groups of 128 tokens with eight-row query-head tiles; and, at eight query
        # heads per KV head, channel-wise K and V in groups of 64 tokens. Every other build
        # stages O in `sO` (K-ring stage 0) and the TMA-VO warp stores it.
        #
        # The direct store wins where the staging, the store and its two waits would
        # otherwise be paid once per one-tile item. It is not free everywhere: the INT4
        # split builds with token-wise K and eight-row query-head tiles (`q_head_tile` 8)
        # send the store's addresses to local memory across the correction's tile loop,
        # and the persistent channel-wise builds outside the scope above either gain local
        # memory or run slower. Only full tiles are admitted, because with padded rows the
        # store's row bound is a constant below the tile and the family goes to local
        # memory. The default tiling, no TMA mask and no query descale are required because
        # only those were measured.
        if not self.persistent:
            return bool(self.do_kernel_red and self.v_axis == "token"
                        and not (self.bits == 4 and self.k_axis == "token"
                                 and self.q_head_tile == 8))
        if not (self.bits == 2 and self.dequant_int2_to_f4 and self.fold_scale_zero):
            return False
        k, kg, v, vg = self.k_axis, self.k_group, self.v_axis, self.v_group
        return bool((k == "token" and v == "token")
                    or (k == "channel" and kg == 64 and v == "token")
                    or (k == "channel" and kg == 128 and v == "channel" and vg == 128
                        and self.q_head_tile == 8)
                    or (k == "channel" and kg == 64 and v == "channel" and vg == 64
                        and self.q_head_tile == 8 and grouped_heads == 8
                        and self._default_tiling and not self.tma_mask
                        and not has_q_descale))

    def __init__(
        self,
        page_size,
        head_dim,
        q_head_tile,
        sequence_tile=None,
        reduction_mode: Literal["kernel", "atomic", "none"] = "kernel",
        *,
        tma_mask=False,
        bits=4,
        code_stages=None,
        scale_stages=None,
        pt_stages=None,
        p_stages=None,
        o_stages=None,
        debug_s=False,
        fold_scale_zero=True,
        code_stages_k=None,
        k_op_stages=None,
        dequant_int2_to_f4=None,
        k_quant=("token", 128),
        v_quant=("token", 128),
        sdtype=None,
        persistent=True,
        persistent_sm_count=148,
    ):
        """
        Parameters
        ----------
        page_size
            Tokens per page. Must be 64 (the quant-pool contract: one scale
            block and one code block per (page, kv-head)).
        head_dim
            Head dimension.
        q_head_tile
            The number of Q head rows one threadblock packs. It is NOT the GQA
            ratio, and it does not bound the ratio: the Q tensor is packed with
            the grouped heads on the tiled mode, so the grid launches
            ``ceil_div(q_heads // kv_heads, q_head_tile)`` head tiles per KV head
            and a ratio larger than the tile is spread over several threadblocks
            rather than refused. It is also not quite the MMA N extent, which is
            ``max(16, q_head_tile)`` because an N-major 8-bit B operand in shared
            memory needs N to be a multiple of 16.
            ``IntPingPongDecodeSm90`` uses the same name for the same quantity;
            that kernel packs a fixed 8 and requires the ratio not to exceed it.
        sequence_tile
            KV tokens per threadblock per loop iteration.
        reduction_mode
            Split-K reduction algorithm:
              - ``"kernel"``: deterministic kernel reduction with partial result workspace.
              - ``"atomic"``: cluster reduction with atomic adds, no workspace.
              - ``"none"``: no split-K, flash decoding disabled.
        tma_mask
            Disable causal masking: out-of-bounds scores are masked to 0
            instead of -inf. Test-only, and the wrapper never builds it: the
            positions of a partial last page past the length then enter the
            column sum, so the softmax is not equivalent there, and neither is
            speculative verification. It may also produce subnormal
            exponentiations if non-oob scores are all negative (max goes to 0).
        bits
            Quantized KV code width (2 or 4). Selects the code-ring stage
            size and the dequant byte map.
        code_stages
            Code-ring depth. ``None`` picks 16 when the f4 feed resolves on
            (which is int2's default), and otherwise 8 per-page stages, i.e. 2
            operand stages of lookahead.
        scale_stages
            Scale-ring depth (1KB f32 [64,4] stages, one per page). ``None`` is 16
            when the f4 feed resolves on, and 2 tiles
            (``2 * sequence_tile/page_size``) otherwise. Must cover at least
            one tile, because the softmax WG's scale-quad prefetch waits a whole
            tile's stages at once (it copies the quads to registers and releases
            the stages before that tile's S-wait). Under ``fold_scale_zero=False``
            the two dequant streams consume the ring instead, and their bound is
            ``prefetch_iters`` tiles (``2 * sequence_tile/page_size`` at the
            shipped depths), asserted at trace time in ``_dequant_warpgroup``.
            The TMA-QK warp acquires each page's scale stage before the next
            page's K-code stage, so a full scale ring gates the CODE lookahead
            too.
        pt_stages
            smem page-table ring depth. Default 4, which is the FLOOR the
            protocol allows rather than a tuning choice:
            ``prefetch_iters = min(2, s_stages - 1)`` is 2 at the shipped
            depths and ``assert pt_stages > prefetch_iters + 1`` then demands 4,
            so the ring holds exactly one tile of margin over the lag TMA-VO
            runs at. The sPT publish rendezvous below is what keeps VO inside
            that margin. Depths up to 8 are nearly
            free in smem; only at 16 does the ring cost a KV stage.
        p_stages
            smem P (BMM2 B) ring depth. Default 4. Also caps the S depth through
            ``s_stages = min(max_s_stages, p_stages)``, and it is the binding
            side of that min at the shipped tiling, so raising it deepens two
            rings at once at the cost of smem the KV ring would otherwise get.
            5 measures slightly faster than 4 and the default stays at 4
            deliberately.
        o_stages
            tmem O (BMM2 C) ring depth (2 or 3). ``None`` resolves through
            ``resolve_tiling``, which is 3 at every quant config except
            (K token-128, V channel-128), where the measured pair is
            (sequence_tile 256, o_stages 2). Depth 3 moves the
            VP/correction coupling distance from tile s-2 to s-3, buying one
            extra tile of slack on the loop-carried
            softmax -> sM -> correction -> O-release -> MMA chain for one more
            ``mma_tile_n``-wide O + L tmem column set. At depth 3 the
            3-phase tL rings and the final barriers occupy named barriers
            7..15.
        debug_s
            Debug tap (the fold-off arm asserts it off): the softmax WG of
            CTA(0,...) dumps the
            FOLDED S of tile 0 (pre-max/exp2, f32,
            ``s_t*S_raw + z'_t*QSum_h``, i.e. before the uniform ``scale_s``
            factor) to the ``mDbgS`` gmem tensor [sequence_tile, blk_tile_n].
            Driver: the end-to-end check's folded-S tap.
        fold_scale_zero
            Ablation arm; default True is the shipped affine fold. False selects
            classic dequant: the dequant WG writes the REAL dequantized values
            ``e4m3(s_t*q + z_t)`` into the operand ring and the whole scale/zero
            fold is off (``fold_k = fold_p = False``, so no K-fold FMAs, no
            QSum, no P hi/lo rows, no pscale, no ``c_h``, baseline single-row P
            and baseline epilogue). Correct numerics, but a strictly less
            accurate kernel than the shipped fold, because the operand rounds
            through bf16 and then e4m3 and P requantizes to e4m3. The dequant is
            deliberately ELEMENT-WISE (one packed bf16 FMA per two elements),
            because an ablation baseline has to pay what dequantization costs.
            Host contract unchanged (``scale_s = sm_scale/q_scale``,
            ``scale_o = 1``). int2 with ``dequant_int2_to_f4=0`` and legacy
            token-128/token-128 quant only; an absent ``dequant_int2_to_f4``
            resolves to 0 here so the pool perm follows. Each of the dequant
            warpgroup's two concurrent K/V streams waits the tile's scale
            stages at the top of every tile, reads only its own tensor's
            (s, z) for the token rows it maps, fences the generic-proxy reads
            against the TMA producer's async-proxy refill, and releases; the
            K stream holds two token rows per page (its two flat chunks) and
            the V stream one token column.
        code_stages_k
            The K code ring's depth; the V code ring takes ``code_stages``
            minus it. ``None`` splits ``code_stages`` in half. Each ring must
            hold a whole tile of its tensor's code stages and be an even tile
            multiple.
        k_op_stages
            The K operand ring's depth; the V operand ring takes ``kv_stages``
            minus it. ``None`` resolves to ``min(2, kv_stages // 2)`` at trace
            time, where ``kv_stages`` resolves from the shared-memory budget,
            which is the silicon-swept int2 split. ``fold_runtime.BITS_TUNING``
            sets 3 at ``bits == 4``, the measured INT4 best. At least 2 per
            ring; the upper bound is asserted at trace time.
        dequant_int2_to_f4
            int2 only: feed BOTH GEMM A operands as PACKED e2m1 read directly
            from SMEM, with no ``tcgen05.cp`` and no TMEM staging. ``None``
            (kwarg absent) resolves to True for ``bits == 2`` unless the
            fold-off ablation (``fold_scale_zero=False``) is set; see
            ``resolve_dequant_int2_to_f4``, the single resolver the pool build
            also uses. An explicit 0 selects the token-major flag-off mode.

            POOL CONTRACT: a pool is valid for exactly ONE kernel mode. Kernels
            built with this flag REQUIRE ``mVq`` stored exactly like ``mKq``
            (token rows, K's within-word order); INT4 uses the same layout. INT2
            with the flag off uses token-major bytes in a channel-row-shaped
            pool. The two shapes differ, and ``__call__`` refuses the wrong one.
            Callers must key the pool build off the same resolver the ctor uses.

            The K feed (BMM1): the dequant WG's int2 K map becomes a 2->4-bit nibble
            deposit into the canonical b4x16_p64 panel image (16B chunks: 8 data
            plus 8 pad bytes per 16 elements, the same byte footprint as the e4m3
            stage), and ``tiled_mma_kq`` is built with
            ``a_dtype = Float4E2M1FN`` (K-major, unchanged).

            The V feed (BMM2): V is stored exactly like K, so the V stream runs the
            K stream's partition and deposits the same packed_p64 panel image
            with the roles of rows and K swapped (chunk of token k, channel
            panel i at i*16*mma_tile_k + k*16). The MMA reads it as an MN-major
            E2M1 operand; the DSL (4.5.2 and the pinned 4.8.0) cannot legalize
            that shared-memory descriptor, so `_mma_vp_warp` builds both descriptors and issues
            the instruction through inline PTX (a silicon probe adjudicated the
            image). B (P) stays e4m3 MN-major, and
            the P store path and the quantized P hi/lo BYTES are bit-identical
            to flag-off, because the in-kernel ``lp - 1`` shift
            compensates the doubled s_v.

            Fold constants: both feeds are q/2 exact, so ``s' = 2*s_t`` and
            ``s'_v = 2*s_v`` (one FMUL each at the scale-quad copy) and both
            betas are ``BETA_FOLD_F4 = 0.0``. Asserts ``bits == 2``.
        persistent
            Static round-robin persistent CTAs (default on): launch
            ``min(h_kv * batch, persistent_sm_count)`` CTAs, each looping over
            items ``bid_z, bid_z + grid_z, ...`` so the per-item prologue
            amortizes (launch ramp, tmem alloc/dealloc and descriptor prefetch
            are microsecond-class per item). Incompatible configs silently
            downgrade to the per-item grid, so callers may force the kwarg
            globally.
            The WHOLE per-item flow INCLUDING the mbarrier and pipeline init
            section re-runs per item, separated by one CTA-wide boundary sync, so
            every ring and mbar re-inits from phase 0 while quiesced and all
            per-kernel ring protocols hold verbatim per item. Only tmem
            alloc/dealloc, TMA descriptor prefetch and
            ``griddepcontrol_launch_dependents`` hoist out of the loop.
            Scope (ctor-asserted): ``reduction_mode='none'`` and no
            ``debug_s``. A seqlen-0 item
            takes the exit-early arm per item, with the tmem dealloc guarded.

        Notes
        -----
        BLASST (``threshold_scale_factor``) is out of scope for this kernel: the
        skip-predicate ring would need a third consumer (the dequant WG) and the
        code and operand rings would need skip-aware advancing.
        ``__call__`` and ``can_implement`` assert it is None.

        """
        # Resolve the quant-config-dependent tiling BEFORE anything reads either
        # name. Both LOCALS are rebound, not just the self.* copies: the
        # scale_stages default, the split code-ring arithmetic and the
        # `% v_group` / `% k_group` legality asserts all read the LOCAL
        # `sequence_tile`, and `self._o_stages` is assigned from the LOCAL
        # `o_stages`. Setting only self.sequence_tile would leave those asserts
        # validating 256 while the kernel runs 128, an assert that guards nothing.
        # `k_quant`/`v_quant` are still the RAW ctor args here
        # (validate_quant_flex has not run yet), so the resolver normalizes them.
        sequence_tile, o_stages = self.resolve_tiling(
            k_quant, v_quant,
            sequence_tile=sequence_tile, o_stages=o_stages)
        # Whether this build runs its quantization pair's default tiling, the only tiling the
        # per-warpgroup register budgets were measured at (`rebalance_regs` in `kernel`).
        self._default_tiling = (sequence_tile, o_stages) == self.resolve_tiling(k_quant, v_quant)
        self.head_dim = head_dim
        self.q_head_tile = q_head_tile
        self.page_size = page_size
        self.sequence_tile = sequence_tile
        self.do_kernel_red = reduction_mode == "kernel"
        self.do_atomic_red = reduction_mode == "atomic"
        self.do_none_red = reduction_mode == "none" or reduction_mode is None
        self.tma_mask = tma_mask
        self.bits = bits
        # Persistent scope resolution: requested persistence silently
        # downgrades on incompatible configs (split-K reductions and the
        # debug tap) so callers may force the kwarg globally — the pattern
        # the correctness gates rely on (splits-4 cells keep the per-item
        # grid).
        self.persistent = bool(int(persistent)) and (
            (reduction_mode == "none" or reduction_mode is None)
            and not debug_s
        )
        self.persistent_sm_count = int(persistent_sm_count)
        # o_stages=4 is unavailable: it required the `fused_correction` epilogue,
        # because the non-fused tL rings would need 8 named-barrier ids.
        assert o_stages in (2, 3), "o_stages must be 2 or 3"
        self._o_stages = o_stages
        self.debug_s = debug_s
        # The K operand ring's depth; None resolves at trace time, where
        # kv_stages is known. The K code ring's depth is resolved with the
        # code-ring split further down.
        self.k_op_stages = None if k_op_stages is None else int(k_op_stages)
        # The int2 K feed is packed e2m1 straight from SMEM. An ABSENT
        # kwarg resolves via the classmethod (bits==2 => on unless an incompatible
        # knob is set); an explicit 0 is the bit-identical flag-off state.
        # `fold_scale_zero=False` selects the classic-dequant ablation arm; see the
        # ctor docstring. `nofold` in `kernel` is `not self.fold_scale_zero`.
        self.fold_scale_zero = bool(int(fold_scale_zero))
        self.dequant_int2_to_f4 = self.resolve_dequant_int2_to_f4(
            bits, dequant_int2_to_f4,
            fold_scale_zero=self.fold_scale_zero)
        # V POOL ORDER, DERIVED the same way: when true, V is stored exactly
        # like K and the V operand is MN-major (see `v_pool_token_rows`).
        self.v_token_rows = self.v_pool_token_rows(
            bits, dequant_int2_to_f4=self.dequant_int2_to_f4,
            fold_scale_zero=self.fold_scale_zero)
        # K POOL ORDER, DERIVED. When this is true the pool's 2-bit codes carry
        # the store-time permutation, and the K deposit reads two masks instead of
        # the two-part gather. There is no kwarg for it: `k_pool_is_permuted`
        # states why, and whoever builds the pool asks the same classmethod, so
        # the two sides cannot disagree about an absent argument.
        #
        # It holds for EITHER K axis. The deposit reproduces the logical-order
        # panel image byte for byte, so a channel-wise fold sees the same operands
        # either way. Channel-wise K therefore reads the two-mask deposit at 3 ALU
        # operations per 16 codes rather than 7.
        #
        # This is the object's answer about its own pool contract, not a
        # trace-time selector: `kernel` does not read it. It is kept because a
        # caller holding the object can ask.
        self.k_pool_permuted = self.k_pool_is_permuted(
            bits, dequant_int2_to_f4=dequant_int2_to_f4,
            k_quant=k_quant,
            fold_scale_zero=self.fold_scale_zero)
        if not self.fold_scale_zero:
            # Scope, deliberately as narrow as the sm90 arm's (its ctor takes
            # the legacy config only): every restriction here is a place the
            # fold-off arm has NO implementation, not a place it is untuned.
            assert bits == 2, (
                "fold_scale_zero=False is wired for int2 only (bits == 2): the "
                "element-wise bf16 FMA itself is width-agnostic, but the code-"
                "to-bf16 magic and the two byte maps are int2's"
            )
            assert not self.dequant_int2_to_f4, (
                "fold_scale_zero=False needs the e4m3 operand feed: the f4 "
                "(e2m1) nibble feed cannot carry dequantized values, and its "
                "V pool perm differs (pass dequant_int2_to_f4=0)"
            )
            # (the legacy token-128/token-128 quant restriction — one (s, z)
            # pair per token per tensor is what a per-thread dequant table is
            # built from — rides the flex knob-exclusion list below)
            assert not debug_s, (
                "the folded-S tap dumps s_t*S_raw + z'_t*QSum, which does not "
                "exist with the fold off (S_raw is already dequantized)"
            )
        if self.dequant_int2_to_f4:
            assert bits == 2, (
                "dequant_int2_to_f4 is the int2 K-feed lever (bits == 2)"
            )
        if code_stages is None:
            code_stages = 16 if self.dequant_int2_to_f4 else 8
        self.code_stages = code_stages
        # Scale ring: f32 [64, 4] per page = 1KB per stage. The softmax WG's
        # scale-quad prefetch waits ONE TILE's worth (pages_s = sequence_tile/64
        # stages) at once, copies the quads to registers and releases the stages
        # before that tile's S-wait, so the ring must hold a tile; at 2*pages_s
        # the TMA-QK warp can issue tile t+1's scales while the softmax has not
        # yet prefetched them (issuance blocks on a full ring). Since the TMA warp
        # acquires each page's scale stage before the next page's K-code stage, a
        # full scale ring gates the CODE lookahead too. Constructor knob; default
        # 2 tiles. The fold-off streams' own, larger bound is asserted at trace
        # time in `_dequant_warpgroup`.
        if scale_stages is None:
            scale_stages = (16 if self.dequant_int2_to_f4
                            else 2 * (sequence_tile // page_size))
        self.scale_stages = scale_stages
        # The page-table and P ring depths are knobs so that a claim about them can be
        # measured instead of argued; both were literals inside `__call__`, reachable
        # from no caller.
        #
        # `pt_stages` (smem page table): 4 is the MINIMUM the protocol allows, not a
        # preference. `prefetch_iters = min(2, s_stages - 1)` is 2 at the shipped depths
        # and `assert pt_stages > prefetch_iters + 1` then demands at least 4, so the ring
        # carries exactly one tile of margin over the lag TMA-VO runs at. A stage costs
        # `pages_s` int32, which is 16 bytes at sequence_tile=256/page_size=64, so a deeper
        # ring is close to free in smem; whether it is worth anything is a question about
        # how far VO can slip, and the note at the sPT publish rendezvous records what
        # happened the last time it slipped past the margin.
        #
        # `p_stages` (smem P, the BMM2 B operand) also sets the S depth, through
        # `s_stages = min(max_s_stages, p_stages)`. It is the BINDING side of that min at
        # the shipped tiling, which is why reclaiming tmem columns deepens nothing while
        # `p_stages` caps `s_stages`. So raising it deepens two rings at once, at the cost
        # of smem the KV ring would otherwise get, and `kv_stages` in the
        # `FOLD_TRACE_TILING` line is where that cost shows up.
        self._pt_stages = 4 if pt_stages is None else int(pt_stages)
        self._p_stages = 4 if p_stages is None else int(p_stages)
        # THREE IS THE ONLY FLOOR CONSTRUCTION CAN PROVE, and the effective floor is
        # usually 4. The kernel asserts `pt_stages > prefetch_iters + 1` with
        # `prefetch_iters = min(2, s_stages - 1)`, and `s_stages = min(max_s_stages,
        # p_stages)` is not known until trace time, so all that follows here is
        # `s_stages >= 2`, hence `prefetch_iters >= 1`, hence `pt_stages >= 3`. At the
        # shipped depths `s_stages` is 4 and the tracer demands 4; at `p_stages == 2` it
        # demands only 3. There is deliberately NO upper bound: a page-table ring competes
        # for smem with the KV ring, which is the residual of the budget, so an
        # over-deep ring is caught by the `kv_stages >= 1` assert further down with the
        # arithmetic in hand rather than by a guess here.
        assert self._pt_stages >= 3, (
            f"pt_stages must be at least 3; got {self._pt_stages}. The kernel asserts "
            f"`pt_stages > prefetch_iters + 1` and `prefetch_iters` is at least 1 for any "
            f"legal s_stages, so a shallower page-table ring has no margin over the lag "
            f"TMA-VO runs at. Whether 3 is enough depends on s_stages, which is resolved "
            f"at trace time, and the tracer is what rejects it there.")
        assert self._p_stages >= 2, (
            f"p_stages must be at least 2 for a ring; got {self._p_stages}")
        # Four warpgroups: mma/tma, softmax, correction and dequant.
        self.threads_per_cta = 4 * warpgroup_threads

        # The four shape arguments both kernels take are checked in ONE place, so the
        # two constructors cannot drift apart on them. This guard runs FIRST, BEFORE
        # THE SHARED SHAPE CHECK, because `_validate_shape` asserts and `python -O`
        # drops asserts. THE SM100 CONSTRAINT IS 128-TILING, NOT D==128:
        # `tiles_dm` and `tiles_dk` are ceil(D/128) and every tile is partitioned and
        # stored whole, so a D that is not a multiple of 128 runs past the last full tile.
        # The legacy token-D path is tiled for D=256, but whether D=256 works is
        # untested; the wrapper refuses it, on the ground that nothing measures it.
        if head_dim <= 0 or head_dim % 128 != 0:
            # NOT an assert: python -O would drop this guard.
            raise ValueError(
                f"this kernel tiles the head dimension at 128 (mma_tile_m == mma_tile_k "
                f"== 128) and masks no partial tile, so head_dim must be a positive "
                f"multiple of 128; got {head_dim}")
        self._validate_shape(page_size, head_dim, q_head_tile, bits)
        # THIS KERNEL NEEDS AT LEAST FOUR Q HEAD ROWS, and this refuses the smaller
        # tiles here rather than letting the DSL fail at trace time. The P hi/lo
        # quantize stores `blk_tile_n` e4m3 values as one vector, and
        # `nvgpu.cvt_fptrunc` requires the result be 32-bit aligned: at 8 bits an
        # element, that is 4 elements. `q_head_tile=2` therefore dies inside the
        # tracer with "'nvgpu.cvt_fptrunc' op result #0 must be 32-bits aligned",
        # an ICE naming a line in the softmax warpgroup rather than the
        # configuration that caused it.
        #
        # `blk_tile_n_of` maps BOTH GQA ratio 1 and ratio 2 to `q_head_tile=2`, so
        # neither compiles today. The refusal is unconditional, and it is a
        # retained restriction rather than a requirement of every surviving
        # path: the fold-off arm runs with `fold_p == False` and takes the
        # single-row P store, so the hi/lo justification below scopes to the
        # fold-enabled configurations. The bound stays unconditional because
        # nothing needs `q_head_tile < 4` and no arm is tested at it.
        assert q_head_tile >= 4, (
            f"q_head_tile >= 4 is required: the P hi/lo quantize truncates "
            f"{q_head_tile} e4m3 values as one vector and nvgpu.cvt_fptrunc "
            f"requires a 32-bit-aligned result, which takes 4. GQA ratios 1 and "
            f"2 both resolve to q_head_tile=2 through blk_tile_n_of and are not "
            f"supported by this path.")
        assert sequence_tile > 0 and sequence_tile % 128 == 0
        assert self.do_kernel_red ^ self.do_atomic_red ^ self.do_none_red
        # Quant-granularity flex, at parity with the SM90 IntPingPongDecodeSm90:
        # per-tensor (axis, group) configs. K: token {D, D/2} or channel
        # {64, 128, 256}; V: token {D} or channel {64, 128, 256} (token-64 V
        # deferred, same as SM90). The per-(page, hk) scale block is the sectioned
        # layout of quant_layout.quant_block_offsets ([K ring | V ring | K ch |
        # V ch]); the legacy ("token", D) x 2 config keeps the interleaved [64, 4]
        # f32 mScale contract and compiles byte-identical. `sdtype` ("f32"/"bf16")
        # is the scale-pool element type: bf16 is REQUIRED by K token-64 (it funds
        # the doubled K ring section) and disallowed elsewhere (SM90 rule).
        self.k_quant, self.v_quant = validate_quant_flex(
            k_quant, v_quant, page_size, head_dim)
        self.k_axis, self.k_group = self.k_quant
        self.v_axis, self.v_group = self.v_quant
        # Where the softmax warpgroup reads the count table of page maxima with two 16-byte
        # loads per lane (`_vmax_hist_loads`) instead of eight 4-byte ones: the split builds
        # of the INT2 E2M1-feed fold with token-wise K and V, whose one or two tiles per CTA
        # leave the table's loads and reduction on the critical path. The cache allocates
        # the table 16-byte aligned (`compile_fold`). The correction warpgroup, which in
        # these builds reads the table itself except at four query heads per KV head
        # (`lp_tail`) and holds the exponent through its tile loop, keeps the 4-byte loads:
        # with the vector form that held exponent goes to local memory at 64/8 heads.
        # Persistent builds read the table once per CTA.
        self.vmax_hist_vec = (not self.persistent and self.do_kernel_red and self.bits == 2
                              and self.dequant_int2_to_f4 and self.fold_scale_zero
                              and self.k_axis == "token" and self.v_axis == "token")
        self.qoff = quant_block_offsets(
            self.k_quant, self.v_quant, N=page_size, D=head_dim)
        if sdtype is None:
            sdtype = "bf16" if (self.k_axis == "token"
                                and self.k_group == head_dim // 2) else "f32"
        # Either spelling reaches here; the canonical NAME is what this kernel stores
        # and compares against below.
        sdtype = normalize_sdtype(sdtype)
        if self.k_axis == "token" and self.k_group == head_dim // 2:
            assert sdtype == "bf16", (
                "K token-64 requires sdtype='bf16' (the doubled K ring "
                "section is funded by the halved scale bytes — SM90 rule)")
        else:
            assert sdtype == "f32", (
                "sdtype='bf16' is only legal with K token-64; every other "
                "config keeps the f32 scale pool")
        self.scale_sdtype = sdtype
        self.scale_dtype = BFloat16 if sdtype == "bf16" else Float32
        assert self.qoff["block_tot"] % 128 == 0, self.qoff  # TMA box shape
        if not self.qoff["legacy"]:
            for knob, name in (
                (debug_s, "debug_s"),
                (not self.fold_scale_zero, "fold_scale_zero=False"),
            ):
                assert not knob, (
                    f"quant-flex configs support the default topology only; "
                    f"{name} requires the legacy (token,{head_dim})x2 config")
            if not self.flex_wired(self.k_quant, self.v_quant, head_dim):
                raise NotImplementedError(
                    "SM100 quant-flex: this config's kernel arm is not "
                    "wired yet (stage b: channel-V, c: channel-K 128/256, "
                    "d: K token-64, e: K channel-64 — see flex_wired)")
            if self.v_axis == "channel":
                assert sequence_tile % self.v_group == 0, (
                    "channel-V groups must tile the sequence tile")
                # the per-part fresh-write walks a single dm accumulator
                # chain (contiguous part ranges in the VP mma_k walk)
                assert head_dim == 128, (
                    "channel-V requires head_dim == 128 (tiles_dm == 1)")
            if self.k_axis == "token" and self.k_group == head_dim // 2:
                # the dual-S mma_k half mapping assumes one 128-wide D
                assert head_dim == 128, "K token-64 requires head_dim == 128"
            if self.k_axis == "channel":
                assert sequence_tile % self.k_group == 0, (
                    "channel-K groups must tile the sequence tile")
                # the Q'-plane builder partition assumes one 128-wide D
                assert head_dim == 128, (
                    "channel-K requires head_dim == 128 (tiles_dk == 1)")
        assert self.code_stages >= 2
        if self.k_op_stages is not None:
            assert self.k_op_stages >= 2, (
                "each split operand ring needs at least 2 stages; the upper "
                "bound is asserted at trace time once kv_stages resolves"
            )
        # SPLIT CODE RINGS, and they must stay split. The code FIFO is not one
        # ring shared by two producer warps and two consumer streams under a
        # positional-ownership contract; it is TWO independent rings (K: TMA-QK
        # -> dequant K stream; V: TMA-VO -> dequant V stream) carved from the
        # same smem block. Per ring, every stage is produced by one warp and
        # consumed and released by one stream, so no cross-stream walk exists
        # and no walker can be lapped by a stage its own releases do not gate.
        # The shared-ring form this replaced deadlocked: one stage of the first
        # V group stayed produced-once-never-released with the two V-stream
        # sibling walkers desynced across a ring revolution, which is an
        # ownership misclassification the skip-walk tokens could not close.
        codes_per_tile_k = (
            (sequence_tile // 128) * math.ceil(head_dim / 128) * 2
        )
        codes_per_tile_v = codes_per_tile_k
        self.code_stages_k = (
            int(code_stages_k) if code_stages_k is not None
            else self.code_stages // 2
        )
        self.code_stages_v = self.code_stages - self.code_stages_k
        for ring, per_tile, name in (
            (self.code_stages_k, codes_per_tile_k, "K"),
            (self.code_stages_v, codes_per_tile_v, "V"),
        ):
            assert ring >= per_tile and ring % per_tile == 0, (
                f"split {name} code ring needs >= one tile of {name} code "
                f"stages AND an even tile multiple (per_tile={per_tile} at "
                f"sequence_tile={sequence_tile}, head_dim={head_dim}); got "
                f"{ring} (code_stages={self.code_stages} split "
                f"{self.code_stages_k}/{self.code_stages_v}; pass "
                f"code_stages_k to move the split point)"
            )
        assert self.scale_stages >= sequence_tile // page_size, (
            "scale ring must hold at least one tile's pages (the softmax "
            "scale-quad prefetch waits a whole tile's stages at once)"
        )

    def can_implement(
        self, kv_splits, qo_shape, kv_shape, qkv_dtype, o_dtype, threshold_scale_factor
    ):
        GqaDecode.can_implement(self, kv_splits, qo_shape, kv_shape, qkv_dtype, o_dtype)

        if threshold_scale_factor is not None:
            raise ValueError(
                "BLASST (threshold_scale_factor) is not supported by "
                "IntDecodeSm100 (out of scope for v1)"
            )
        if qkv_dtype is not cutlass.Float8E4M3FN:
            raise ValueError(
                f"IntDecodeSm100 is fp8-mode-only (Float8E4M3FN), got {qkv_dtype}"
            )

    ##############################
    # Decode Kernel launch
    ##############################
    @cute.jit
    def __call__(
        self,
        kv_splits: Int32,
        seqlens: Union[cute.Tensor, Int32],
        table_offsets: Optional[cute.Tensor],
        page_table: cute.Tensor,
        q_bshd: cute.Tensor,
        o_bshd: cute.Tensor,
        l_bsh: Optional[cute.Tensor],
        o_partial_bshd: Optional[cute.Tensor],
        l_partial_bsh: Optional[cute.Tensor],
        m_partial_bsh: Optional[cute.Tensor],
        scale_s: Float32,
        scale_o: Float32,
        threshold_scale_factor: Optional[Float32],
        stream: cuda.CUstream,
        enable_pdl: bool = True,
        mKq: Optional[cute.Tensor] = None,
        mVq: Optional[cute.Tensor] = None,
        mScale: Optional[cute.Tensor] = None,
        mDbgS: Optional[cute.Tensor] = None,
        mVMaxHist: Optional[cute.Tensor] = None,
        q_descale: Optional[cute.Tensor] = None,
        pscale_lp: Optional[Float32] = None,
    ):
        """Every build but the graph split build: the argument list `launch_fold` passes, with
        no graph argument, so that an eager build's compiled function takes exactly the
        arguments it took before graph mode existed. The parameters are `_launch_decode`'s."""
        self._launch_decode(
            kv_splits, seqlens, table_offsets, page_table, q_bshd, o_bshd, l_bsh,
            o_partial_bshd, l_partial_bsh, m_partial_bsh, scale_s, scale_o,
            threshold_scale_factor, stream, enable_pdl, mKq, mVq, mScale, mDbgS, mVMaxHist,
            q_descale, pscale_lp,
            None, None, None)

    @cute.jit
    def call_graph(
        self,
        kv_splits: Int32,
        seqlens: Union[cute.Tensor, Int32],
        table_offsets: Optional[cute.Tensor],
        page_table: cute.Tensor,
        q_bshd: cute.Tensor,
        o_bshd: cute.Tensor,
        l_bsh: Optional[cute.Tensor],
        o_partial_bshd: Optional[cute.Tensor],
        l_partial_bsh: Optional[cute.Tensor],
        m_partial_bsh: Optional[cute.Tensor],
        scale_s: Float32,
        scale_o: Float32,
        threshold_scale_factor: Optional[Float32],
        stream: cuda.CUstream,
        enable_pdl: bool = True,
        mKq: Optional[cute.Tensor] = None,
        mVq: Optional[cute.Tensor] = None,
        mScale: Optional[cute.Tensor] = None,
        mDbgS: Optional[cute.Tensor] = None,
        mVMaxHist: Optional[cute.Tensor] = None,
        q_descale: Optional[cute.Tensor] = None,
        pscale_lp: Optional[Float32] = None,
        mWork: Optional[cute.Tensor] = None,
        max_pages: Optional[Int32] = None,
        mSplitCounts: Optional[cute.Tensor] = None,
    ):
        """The graph split build (`compile_fold(graph_split=True)`, launched by
        `launch_fold_graph`): `__call__`'s arguments and the work list's three. The parameters
        are `_launch_decode`'s."""
        self._launch_decode(
            kv_splits, seqlens, table_offsets, page_table, q_bshd, o_bshd, l_bsh,
            o_partial_bshd, l_partial_bsh, m_partial_bsh, scale_s, scale_o,
            threshold_scale_factor, stream, enable_pdl, mKq, mVq, mScale, mDbgS, mVMaxHist,
            q_descale, pscale_lp,
            mWork, max_pages, mSplitCounts)

    @cute.jit
    def _launch_decode(
        self,
        kv_splits: Int32,
        seqlens: Union[cute.Tensor, Int32],
        table_offsets: Optional[cute.Tensor],
        page_table: cute.Tensor,
        q_bshd: cute.Tensor,
        o_bshd: cute.Tensor,
        l_bsh: Optional[cute.Tensor],
        o_partial_bshd: Optional[cute.Tensor],
        l_partial_bsh: Optional[cute.Tensor],
        m_partial_bsh: Optional[cute.Tensor],
        scale_s: Float32,
        scale_o: Float32,
        threshold_scale_factor: Optional[Float32],
        stream: cuda.CUstream,
        enable_pdl: bool,
        mKq: Optional[cute.Tensor],
        mVq: Optional[cute.Tensor],
        mScale: Optional[cute.Tensor],
        mDbgS: Optional[cute.Tensor],
        mVMaxHist: Optional[cute.Tensor],
        q_descale: Optional[cute.Tensor],
        pscale_lp: Optional[Float32],
        mWork: Optional[cute.Tensor],
        max_pages: Optional[Int32],
        mSplitCounts: Optional[cute.Tensor],
    ):
        """
        Parameters
        ----------
        kv_splits
            Threadblocks per sequence (flash decoding). The graph split build (``mWork``)
            ignores it: each CTA's split count comes from its work entry.
        seqlens
            Per-batch sequence lengths. The graph split build reads its lengths from the
            work list instead, and takes this tensor only for the shared-memory map of the
            build it shares code with.
        table_offsets
            Starting offset into ``page_table`` for each batch. None in the graph split
            build, which computes ``b * max_pages``.
        page_table
            Logical → virtual page index mapping.
        q_bshd
            Q tensor in BSHD logical view (strides can be BHSD)
        o_bshd
            Output tensor. Must be zero initialized for atomic reduction.
        l_bsh
            Log-sum-exp output (Float32 [B, s_q, HQ], natural log), or None. An unsplit
            build's reduction warp stores it (`reduction_none`); a split build passes it to the
            reduction kernel only, so its decode kernel is the one built without it. The atomic
            reduction's store is base 2 and unpredicated, and `compile_fold` refuses an LSE
            there.
        o_partial_bshd
            Partial O per kv split (kernel-red workspace).
        l_partial_bsh
            Partial ``colsum_p`` per kv split (kernel-red workspace).
        m_partial_bsh
            Partial ``colmax_s`` per kv split (kernel-red workspace). The reduction
            kernel takes the final ``colmax_s`` as their maximum.
        scale_s
            Softmax scale.
        scale_o
            Output scale, applied in the reduction epilogue.
        threshold_scale_factor
            BLASST per-batch skip-softmax threshold scale factor. The kernel
            divides this by each batch's KV seqlen to obtain the effective
            per-request threshold. ``None`` disables BLASST.
        stream
            CUDA stream to launch on.
        enable_pdl
            Programmatic Dependent Launch. Runtime-dynamic — no recompile on
            toggle.
        mKq, mVq, mScale
            Quantized KV pools (the pool contract: V stored exactly like K for
            INT4 or INT2 with ``dequant_int2_to_f4=1``, otherwise in the
            channel-row order. K is packed in logical order, EXCEPT that INT4 and
            INT2 with the f4 feed at either K axis require the store permutation
            applied to the codes first: ask ``k_pool_is_permuted``, which is the
            same classmethod this class answers, because a pool written in
            one order and read in the other is wrong at every K channel and
            raises nothing):
            ``mKq`` uint8 [NP,HKV,64,D*bits//8]; ``mVq`` uint8 the same shape
            and order as ``mKq`` on INT4 and INT2 with ``dequant_int2_to_f4=1``
            (``v_pool_token_rows``), else [NP,HKV,D,64*bits//8]; the slab shape
            is asserted. ``mScale`` has TWO layouts and this asserts
            whichever the configuration resolved: legacy
            (``k_quant == v_quant == ("token", head_dim)``) is f32
            [NP,HKV,64,4], the interleaved per-token block; every other
            granularity is the rank-3 sectioned block [NP,HKV,block_tot] in the
            pool's ``sdtype``, whose width is ``self.qoff["block_tot"]`` and
            whose section offsets come from ``quant_layout.quant_block_offsets``.
            See the module docstring's HOST CONTRACT.
            Scale convention: the query descale d (q ~ q_fp8 * d; it is
            ``1/qs_Q`` for the host's MULTIPLY-scale query quantization)
            multiplies the WHOLE folded S
            (S = sm*(s_t*S_raw + z'_t*QSum_e4m3)*d). Without ``q_descale`` the
            host folds a per-tensor d into ``scale_s = sm_scale * d``; with it,
            ``scale_s`` is ``sm_scale`` and the kernel multiplies each row by
            its own d_h. Either way the factor is applied in log2 space at the
            exp2 exactly like the baseline; only the per-token coefficients
            (s_t, z'_t) are applied as raw f32 FMAs on S before the max-reduce.
            The per-token K/V affines come from ``mScale``; pass
            ``scale_o = 1.0`` (the V dequant is folded in-kernel).
        mVMaxHist
            uint32 [256] in global memory: bin b counts the cache's page
            maxima of s_v whose ``fold_runtime.vmax_bucket`` is b. Required
            exactly when the fold reads token-wise V
            (``fold_runtime.reads_v_max_hist``), None otherwise. The kernel's
            P-scale exponent is ``pscale_lp_of_bucket`` of its highest nonzero
            bin (``_pool_pscale_lp``); a table whose top bin is above the
            pool's largest s_v only lowers the exponent, and one below it lets
            products saturate in e4m3.
        q_descale
            Optional fp32 [B, HQ] query descale, any strides (a stride-0
            expand for a per-tensor value). None compiles the kernel without
            the per-row factor.
        pscale_lp
            f32 launch scalar, the P-scale exponent of a build that reads no
            table (channel-wise V, and the fold-off ablation): always
            ``fold_runtime.TABLE_FREE_PSCALE_LP``. Required exactly when
            ``mVMaxHist`` is not, None otherwise. Why the constant is a launch
            argument rather than written into the kernel: see the P-scale
            exponent in ``kernel``.
        mDbgS
            ``debug_s=True`` output: f32 [sequence_tile, blk_tile_n];
            row t / col n = folded S of tile 0 (split 0, CTA coords 0) for
            token t, packed head n, BEFORE the scale_s/log2e factor and the
            causal mask: ``s_t*S_raw[n,t] + (z_t - BETA_FOLD*s_t)*QSum[n]``.
        mWork, max_pages, mSplitCounts
            The graph split build (CUDA-graph mode), given together or not at all.
            ``mWork`` is int32 [W, 4], 16-byte aligned, one entry per CTA of a fixed
            one-dimensional grid of W CTAs: ``(item, split, k, seq_len)``, where item is
            ``b * HKV + h`` and k the request's split count, or item -1 for a slot past the
            end of the list. ``max_pages`` is the row stride of the engine's page table
            ``[B, max_pages]``, which ``page_table`` holds flat. ``mSplitCounts`` is int32
            [B], each request's k, which the reduction reads. The partial tensors are
            ``[S, B, 1, HQ, D]`` with S the most chunks one request can get, and each piece
            writes slot ``split``. The host writes the list before the graph replays, so
            the kernels read it before their dependent-launch waits. Without these the
            build is today's, unchanged.
        """
        ##############################
        # TiledMma creation
        ##############################
        mma_dtype = q_bshd.dtype
        acc_dtype = Float32
        assert mKq is not None and mVq is not None and mScale is not None
        assert mKq.dtype is Uint8 and mVq.dtype is Uint8
        # The V pool's slab shape says which layout it carries, so a pool of
        # the other layout is refused here rather than read as the wrong codes.
        # The two shapes cannot collide at D=128 (token rows 64 x D*bits/8,
        # channel rows D x 64*bits/8).
        v_slab = self.v_pool_rows(self.page_size, self.head_dim, self.bits,
                                  dequant_int2_to_f4=self.dequant_int2_to_f4,
                                  fold_scale_zero=self.fold_scale_zero)
        assert tuple(mVq.shape[2:]) == v_slab, (
            f"V pool slab {tuple(mVq.shape[2:])} != {v_slab}: this configuration "
            f"stores V {'exactly like K (token rows)' if self.v_token_rows else 'in channel rows'}; "
            "pools written in the other layout must be rebuilt")
        if cutlass.const_expr(self.qoff["legacy"]):
            assert mScale.dtype is Float32
        else:
            # flex: flat sectioned block [NP, HKV, block_tot] in sdtype
            assert mScale.dtype is self.scale_dtype, (
                f"flex mScale must be {self.scale_sdtype}"
            )
            assert len(mScale.shape) == 3 and (
                mScale.shape[-1] == self.qoff["block_tot"]
            ), (mScale.shape, self.qoff)
        assert mDbgS is not None or not self.debug_s
        # The table exists exactly where s_v rides inside P: the fold with token-wise V.
        if cutlass.const_expr(self.fold_scale_zero and self.v_axis == "token"):
            assert mVMaxHist is not None and mVMaxHist.dtype is Uint32, (
                "the fold with token-wise V needs mVMaxHist, uint32 [256]")
            assert pscale_lp is None, "a build that reads mVMaxHist takes no pscale_lp"
        else:
            assert mVMaxHist is None, (
                "channel-wise V and the fold-off ablation take no mVMaxHist")
            assert pscale_lp is not None, (
                "channel-wise V and the fold-off ablation take pscale_lp")
        assert q_descale is None or q_descale.dtype is Float32
        # THE GRAPH SPLIT BUILD is the split build with its coordinates taken from a work
        # list (`mWork`): a fixed grid of W CTAs that CUDA-graph replay can launch for any
        # plan. The three graph arguments come together.
        graph_split = mWork is not None
        assert (max_pages is not None) == graph_split, (
            "mWork and max_pages come together (the graph split build)")
        assert (mSplitCounts is not None) == graph_split, (
            "mWork and mSplitCounts come together (the graph split build)")
        if cutlass.const_expr(graph_split):
            assert self.do_kernel_red and not self.persistent, (
                "the graph split build is a split build (reduction_mode='kernel')")
            assert mWork.dtype is Int32 and mSplitCounts.dtype is Int32
            assert isinstance(seqlens, cute.Tensor) and table_offsets is None, (
                "the graph split build takes seqlens as a tensor (for the varlen "
                "shared-memory map) and computes its table offsets, so table_offsets is None")
            # One CTA covers a KV head's whole query-head group: the work item names a
            # (request, KV head) pair and nothing else.
            assert q_bshd.shape[2] // mKq.shape[1] <= self.q_head_tile, (
                "the graph split build needs one query-head tile per KV head")
        # fp8-fold scope (see ctor docstring)
        assert mma_dtype is cutlass.Float8E4M3FN, "IntDecodeSm100 is fp8-mode-only"
        assert threshold_scale_factor is None, "BLASST out of scope for v1"

        # Block tile sets the granularity at which threadblocks consume work
        blk_tile_s = self.sequence_tile
        blk_tile_h = self.q_head_tile
        # The query-token tile is PINNED at 1. The runtime `prediction` axis is
        # untouched: the Q tensor still carries (grouped_heads, prediction) on its
        # tiled mode, the causal mask still walks the last `prediction` tokens, and
        # a multi-token request would tile one token per threadblock. What a knob
        # here would buy is PACKING several query tokens into one MMA N tile, an
        # optimization for a feature neither shipping kernel has.
        blk_tile_p = 1
        blk_tile_d = self.head_dim
        blk_tile_shpd = (blk_tile_s, blk_tile_h, blk_tile_p, blk_tile_d)

        # MMA tile sets the granularity at which TMAs + MMAs are staged
        # NOTE: mma_tile_k is computed from the CONTAINER width 8 —
        # under dequant_int2_to_f4 the K elements are e2m1 but each still
        # occupies one byte of the ring stage (8 data + 8 pad bytes per 16
        # nibbles), so the byte/stage accounting is dtype-role-independent.
        mma_tile_m = 128
        mma_tile_k = 128 * 8 // mma_dtype.width
        # N-major 8b B in smem requires N multiple of 16
        min_mma_tile_n = 16 if mma_dtype.width == 8 else 8
        blk_tile_n = blk_tile_h * blk_tile_p  # linearized tiler
        mma_tile_n = max(min_mma_tile_n, blk_tile_n)
        mma_tile_mnk = (mma_tile_m, mma_tile_n, mma_tile_k)
        # P hi/lo row budget: hi rows n in [0, g), lo rows n in [g, 2g) of
        # the P stage / O accumulator N axis. g <= 8 rides the baseline's
        # mma_tile_n = 16 for free; g = 16 would need mma_tile_n = 32, which the
        # accounting supports but nothing wires.
        assert 2 * blk_tile_n <= mma_tile_n, (
            f"P hi/lo needs 2*blk_tile_n <= mma_tile_n "
            f"({2 * blk_tile_n} > {mma_tile_n}); g > 8 unsupported in v1"
        )

        # MMA tiles per block tile
        tiles_sm = blk_tile_s // mma_tile_m
        tiles_dm = math.ceil(blk_tile_d / mma_tile_m)
        tiles_dk = math.ceil(blk_tile_d / mma_tile_k)
        pages_s = blk_tile_s // self.page_size
        assert blk_tile_s % mma_tile_m == 0
        assert mma_tile_n % blk_tile_n == 0
        if cutlass.const_expr(self.k_axis == "channel"):
            # The KQ-MMA and softmax arms branch on k_parts against tiles_sm, and
            # only these three shapes are wired. Asserted rather than assumed
            # because `sequence_tile` is a constructor knob, and an unwired ratio
            # would otherwise fall through to the k_parts==1 arm at a silently
            # wrong plane rate.
            _kp = max(blk_tile_s // self.k_group, 1)
            assert _kp in (1, tiles_sm, 2 * tiles_sm), (
                f"channel-K: k_parts={_kp} at sequence_tile={blk_tile_s}, "
                f"k_group={self.k_group}, tiles_sm={tiles_sm} is not one of the "
                f"wired plane rates (1 / tiles_sm / 2*tiles_sm)")

        # GEMM1: (S_K, H_R, D, (H_K, B))
        # With dequant_int2_to_f4, A is packed e2m1 read directly from SMEM
        # (silicon-exact, the packed_p64 panel image); B stays e4m3.
        tiled_mma_kq = self._tiled_mma_kq(mma_dtype, acc_dtype, mma_tile_mnk)

        # GEMM2: (D, H_R, S_K, (H_K, B)). A = V is MN-major (channel-
        # contiguous) in every configuration; on the f4 feed it is packed e2m1
        # issued with hand-built descriptors (see `_mma_vp_warp`). B (P) is
        # e4m3 MN-major.
        tiled_mma_vp = self._tiled_mma_vp(mma_dtype, acc_dtype, mma_tile_mnk)

        ##############################
        # Calculate stage counts
        ##############################
        # Fixed stage counts
        self.pt_stages = pt_stages = self._pt_stages  # smem page table buffer
        # sp_stages is still a literal, and deliberately: the block it sizes is allocated
        # only when `threshold_scale_factor is not None`, which no shipped configuration
        # passes (BLASST is out of scope), so it contributes zero smem and there is nothing
        # to tune. A knob nothing can exercise would be worse than the constant.
        self.sp_stages = sp_stages = 4  # smem skip predicates
        self.p_stages = p_stages = self._p_stages  # smem P (BMM2 B)
        self.o_stages = o_stages = self._o_stages  # tmem O (BMM2 C) ring

        # Calculate tmem alloc
        tmem_capacity_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        # K token-64 splits the D contraction into two S accumulator chains;
        # K channel-64 runs two per-group chains per M-tile — both need the
        # second S accumulator set
        k_tok64_c = (self.k_axis == "token"
                     and self.k_group == self.head_dim // 2)
        # The straddle predicate is "a channel group is narrower than an M-tile",
        # i.e. k_parts > tiles_sm. Do not write it `sequence_tile // k_group > 2`:
        # that is the same thing ONLY at the shipped sequence_tile=256, where
        # tiles_sm == 2. At sequence_tile=128 with Gt=64 that form goes False while
        # the MMA path's own predicate (`k_parts == 2*tiles_sm`) stays True, so the
        # KQ gemm would write a tCtS2 this accounting never allocated. The
        # structural form is the one that holds at every tiling.
        k_ch64_c = (self.k_axis == "channel"
                    and self.k_group < mma_tile_m)
        dual_s_c = k_tok64_c or k_ch64_c
        tmem_s_stage_cols = tiles_sm * mma_tile_n * (2 if dual_s_c else 1)
        # per-thread colsum: the tL tmem channel carries it to the correction
        # warpgroup, except under channel-V, whose [colsum | l] transport is a
        # pre-reduced smem ring instead.
        v_chan_c = self.v_axis == "channel"
        v_parts_c = max(
            (self.sequence_tile // self.v_group) if v_chan_c else 1, 1
        )
        o_ring_c = o_stages * v_parts_c if v_chan_c else o_stages
        # `o_ring` is STRUCTURALLY PINNED to o_stages * v_parts; it is not a slack
        # knob. `softmax_phase` advances mod `o_stages` and selects both the
        # `colsum_p` register slot and `o_slot`, while the O pipeline has `o_ring`
        # stages, and the two coincide only when v_parts divides the walk.
        # Deepening o_ring alone by 1 corrupts every launch. By contrast lc_stages
        # IS slack: +2 changed nothing.
        tmem_alloc_cols = (
            0 if v_chan_c
            else mma_tile_n * o_stages
        )
        tmem_alloc_cols += tiles_dm * mma_tile_n * o_ring_c  # O (virtual ring)
        max_s_stages = (tmem_capacity_cols - tmem_alloc_cols) // tmem_s_stage_cols
        self.s_stages = s_stages = min(max_s_stages, p_stages)

        tmem_alloc_cols += tmem_s_stage_cols * s_stages  # S
        tmem_alloc_cols = 2 ** math.ceil(math.log2(tmem_alloc_cols))  # po2
        self.tmem_alloc_cols = tmem_alloc_cols
        assert tmem_alloc_cols <= tmem_capacity_cols

        # Calculate smem alloc
        smem_alloc_bits = 0
        mbarrier_bits = Int64.width
        pipe_stage_bits = mbarrier_bits * 2  # producer + consumer
        mk_stage_bits = mma_tile_m * mma_tile_k * mma_dtype.width
        nk_stage_bits = mma_tile_n * mma_tile_k * mma_dtype.width
        mn_stage_bits = mma_tile_m * mma_tile_n * mma_dtype.width
        # tmem ptr
        smem_alloc_bits += Int32.width
        # seqlen, table offset
        is_varlen = isinstance(seqlens, cute.Tensor)
        smem_alloc_bits += (Int32.width * 2) if is_varlen else 0
        # page table
        smem_alloc_bits += pt_stages * pages_s * Int32.width
        # skip predicates
        if cutlass.const_expr(threshold_scale_factor is not None):
            smem_alloc_bits += sp_stages * (Int32.width + pipe_stage_bits)
        # colmax + colsum
        smem_alloc_bits += blk_tile_n * acc_dtype.width
        smem_alloc_bits += blk_tile_n * warpgroup_warps * acc_dtype.width
        # fold stats: c_h per-warp reduction slots (the sVmax cell is not
        # needed — the pscale is launch-global). Charged unconditionally.
        # The device allocates `sC` under `fold_p`, which is
        # `self.fold_scale_zero`, so the fold-off arm is charged for a ring it
        # never allocates. That over-charge is deliberate: narrowing it is the
        # one edit here that could move `kv_stages`, and over-charging can only
        # cost a ring stage, never overflow shared memory.
        smem_alloc_bits += (
            blk_tile_n * warpgroup_warps
        ) * acc_dtype.width
        if cutlass.const_expr(self.do_atomic_red):
            smem_alloc_bits += max_reduction_iters * blk_tile_n * acc_dtype.width * 2
            smem_alloc_bits += max_reduction_iters * mbarrier_bits * 2
        # Q, S, P, O
        smem_alloc_bits += tiles_dk * nk_stage_bits + mbarrier_bits  # 1 mbar for Q
        smem_alloc_bits += s_stages * pipe_stage_bits  # s in tmem
        smem_alloc_bits += p_stages * (tiles_sm * mn_stage_bits + pipe_stage_bits)
        smem_alloc_bits += o_ring_c * pipe_stage_bits  # o in tmem (virtual ring)
        # Code ring: one PAGE of packed codes per stage
        # (64 x mma_tile_k codes = 4KB int4 / 2KB int2).
        code_stages = self.code_stages
        code_stage_bits = self.page_size * mma_tile_k * self.bits
        smem_alloc_bits += code_stages * (code_stage_bits + pipe_stage_bits)
        # Scale ring: the RING sections of the per-page scale block (legacy:
        # the interleaved [page_size, 4] f32 quad = 1KB; flex: ring_tot
        # sdtype elements — channel sections bypass the ring, read per group
        # by gmem LDG in the consuming WG).
        scale_ring_elems = self.qoff["ring_tot"]
        scale_stage_bits = scale_ring_elems * self.scale_dtype.width
        smem_alloc_bits += self.scale_stages * (scale_stage_bits + pipe_stage_bits)
        if cutlass.const_expr(self.v_axis == "channel"):
            # channel-V: pre-reduced [colsum | l] smem ring
            # ([nL, 4 warps] f32 per virtual slot) + its pipeline mbars
            nL_c = blk_tile_n * 2
            smem_alloc_bits += (o_ring_c + v_parts_c) * (
                nL_c * warpgroup_warps * acc_dtype.width + pipe_stage_bits
            )
        if cutlass.const_expr(self.k_axis == "channel"):
            # channel-K: Q'-plane ring (B-operand-sized e4m3 stages) + bias
            # ring ([lift[g] | bias[g]] f32 entries). There are no amax slots and
            # no round mbarrier: the lift is per row, so no cross-warp rendezvous
            # exists.
            k_parts_c = max(self.sequence_tile // self.k_group, 1)
            qp_st_c = (self._o_stages + 2) * k_parts_c
            qp_stages_bits = qp_st_c * (
                mma_tile_n * mma_tile_k * mma_dtype.width + pipe_stage_bits
            )
            qp_stages_bits += qp_st_c * (
                2 * blk_tile_n * acc_dtype.width + pipe_stage_bits
            )
            smem_alloc_bits += qp_stages_bits
        alignment_bits = 1024 - (smem_alloc_bits % 1024)
        # K, V
        smem_capacity_bits = utils.get_smem_capacity_in_bytes("sm_100") * 8
        remaining_bits = smem_capacity_bits - smem_alloc_bits - alignment_bits
        kv_stages = remaining_bits // mk_stage_bits
        kv_stages -= 1 if kv_stages * pipe_stage_bits > alignment_bits else 0
        assert kv_stages >= 1, (
            f"SMEM budget exhausted: code_stages={self.code_stages} "
            f"(x{code_stage_bits // 8}B) + scale ring leave "
            f"kv_stages={kv_stages}; shrink the code/scale rings"
        )
        if cutlass.const_expr(os.environ.get("FOLD_TRACE_TILING")):
            # Trace-time report of the EFFECTIVE tiling, printed from inside the
            # kernel rather than from whichever caller thinks it configured it.
            #
            # This exists because a tiling A/B can silently compare a kernel with
            # itself, when an override lands at one constructor site and the
            # benchmark goes through another. The effective value reports itself.
            #
            # `kv_stages` is the reason this sits AFTER the smem budget rather
            # than beside the other stage counts: it is the RESIDUAL of that
            # budget, so it is where any ring that grows (e.g. `qp_stages`
            # doubling with `k_parts`) actually shows up as a cost.
            _qp_st = ((self._o_stages + 2) * max(blk_tile_s // self.k_group, 1)
                      if self.k_axis == "channel" else 0)
            print(
                f"[FOLD_TRACE_TILING] sequence_tile={self.sequence_tile} "
                f"o_stages={self._o_stages} "
                f"mma_tile_mnk={mma_tile_mnk} tiles_sm={tiles_sm} "
                f"pages_s={pages_s} "
                f"k_quant=({self.k_axis},{self.k_group}) "
                f"v_quant=({self.v_axis},{self.v_group}) "
                f"k_parts={max(self.sequence_tile // self.k_group, 1) if self.k_axis == 'channel' else 1} "
                f"v_parts={v_parts_c} o_ring={o_ring_c} "
                f"qp_stages={_qp_st} "
                f"s_stages={s_stages} code_stages={self.code_stages} "
                f"scale_stages={self.scale_stages} "
                f"pt_stages={self.pt_stages} p_stages={self.p_stages} "
                f"kv_stages={kv_stages} "
                f"tmem_cols={tmem_alloc_cols} "
                f"smem_kB={smem_alloc_bits // 8192}",
                flush=True,
            )

        ##############################
        # TMA creation
        ##############################
        h_k = mKq.shape[1]
        o_bshd_ = o_partial_bshd if self.do_kernel_red else o_bshd

        # Reorder and group modes for GEMM
        # ((h_g, s_q), d, (h_k, b))
        mQ_nkl = GqaDecode.gemm_view(GqaDecode.gqa_pack(q_bshd, h_k), True)
        # (d, (h_g, s_q), (h_k, b_partial))
        mO_mnl = GqaDecode.gemm_view(GqaDecode.gqa_pack(o_bshd_, h_k), False)

        # Pool views (rows, row_bytes, (h_k, page_count)) — the pool tensors
        # are [NP, HKV, rows, row_bytes]; reorder modes like gemm_view does for BSHD.
        def pool_view(t):
            layout = cute.group_modes(cute.select(t.layout, (2, 3, 1, 0)), 2, 4)
            return cute.make_tensor(t.iterator, layout)

        # ((MMA_N, MMA_K), #MMA_N, #MMA_K, q_stages)
        smem_layout_q = sm100_utils.make_smem_layout_b(
            tiled_mma_kq, mma_tile_mnk, mma_dtype, tiles_dk
        )

        # ((MMA_M, MMA_K), #MMA_M, #MMA_K, kv_stages)
        # The K ring keeps its BYTE footprint and e4m3/SW128 allocation
        # layout under dequant_int2_to_f4 (16B chunks = 8 data + 8 pad bytes
        # per 16 f4 elements == 1B per element, exactly the e4m3 stage), so
        # the smem layout is always built from an e4m3 K-major atom — a
        # layout-only twin when tiled_mma_kq carries the f4 A dtype. The
        # MMA-side f4 packed_p64 view is built in decode() over the same
        # ring bytes (swizzle-free recast).
        tiled_mma_kq_smem = tiled_mma_kq
        if cutlass.const_expr(self.dequant_int2_to_f4):
            tiled_mma_kq_smem = sm100_utils.make_trivial_tiled_mma(
                mma_dtype,
                mma_dtype,
                OperandMajorMode.K,
                OperandMajorMode.K,
                acc_dtype,
                tcgen05.CtaGroup.ONE,
                mma_tile_mnk[:2],
                tcgen05.OperandSource.SMEM,
            )
        smem_layout_k_mma = sm100_utils.make_smem_layout_a(
            tiled_mma_kq_smem, mma_tile_mnk, mma_dtype, kv_stages
        )
        # The same layout-only twin trick applies to the V side: the ring alloc
        # keeps the e4m3 MN_SW128 stage layout (byte footprint identical:
        # the f4 stage is 16B chunks = 8 data + 8 pad bytes per 16 nibbles
        # == 1 B/elem). The f4 V stage itself is the swizzle-free MN-major
        # packed_p64 panel image the dequant V stream writes through a byte
        # view, and `_mma_vp_warp` describes it with hand-built descriptors.
        tiled_mma_vp_smem = tiled_mma_vp
        if cutlass.const_expr(self.dequant_int2_to_f4):
            tiled_mma_vp_smem = sm100_utils.make_trivial_tiled_mma(
                mma_dtype,
                mma_dtype,
                OperandMajorMode.MN,
                OperandMajorMode.MN,
                acc_dtype,
                tcgen05.CtaGroup.ONE,
                mma_tile_mnk[:2],
            )
        smem_layout_v_mma = sm100_utils.make_smem_layout_a(
            tiled_mma_vp_smem, mma_tile_mnk, mma_dtype, kv_stages
        )

        # (MMA_TILE_M, MMA_TILE_K, kv_stages)
        smem_layout_k_mk = cute.composition(
            smem_layout_k_mma,
            cute.make_layout((mma_tile_m, mma_tile_k, kv_stages)),
        )

        # ((PAGE, MMA_TILE_K), #PAGE_M, 1, kv_stages)
        smem_layout_k_tma = cute.tiled_divide(
            smem_layout_k_mk, (self.page_size, mma_tile_k)
        )
        # (TMA, #PAGE_M, kv_stages)
        smem_layout_k_tma = cute.select(smem_layout_k_tma, [0, 1, 3])

        # Code ring stage layouts.
        smem_layout_code_k_tma = smem_layout_code_v_tma = None
        smem_layout_code_k_box = smem_layout_code_v_box = None
        # One PAGE of packed codes per stage, plain row-major
        # uint8 (no swizzle), 16B-aligned rows. A K stage is one
        # full-width (64 tokens, row_bytes) box, so the dequant K
        # stream partitions it into linear contiguous chunks. A V stage is
        # the same box where V is stored like K (`v_token_rows`); the INT2
        # E4M3 feed and the fold-off arm copy their channel-row-shaped pool's
        # flat token-major page stream (the channel-row order) with a
        # (128, 64*bits/8) box instead. Both stage kinds have the same byte
        # total (code_stage_bytes_).
        code_k_box_bytes = mma_tile_k * self.bits // 8
        code_v_row_bytes = self.page_size * self.bits // 8
        code_stage_bytes_ = self.page_size * mma_tile_k * self.bits // 8
        # TMA box layouts (compact row-major within a box). A token-row V pool
        # is stored exactly like K, so its code stage is K's box too.
        smem_layout_code_k_box = cute.make_layout(
            (self.page_size, code_k_box_bytes), stride=(code_k_box_bytes, 1)
        )
        if cutlass.const_expr(self.v_token_rows):
            smem_layout_code_v_box = smem_layout_code_k_box
        else:
            smem_layout_code_v_box = cute.make_layout(
                (mma_tile_m, code_v_row_bytes), stride=(code_v_row_bytes, 1)
            )
        # ((box), #boxes, stages) views for tma_partition. The code rings
        # are split: the K view spans stages [0, code_stages_k); the V
        # view is REBASED past them by the caller (byte offset
        # code_stages_k * stage_bytes into the same flat sCode block) and
        # spans code_stages_v stages.
        smem_layout_code_k_tma = cute.make_layout(
            ((self.page_size, code_k_box_bytes), 1, self.code_stages_k),
            stride=(
                (code_k_box_bytes, 1),
                self.page_size * code_k_box_bytes,
                code_stage_bytes_,
            ),
        )
        if cutlass.const_expr(self.v_token_rows):
            smem_layout_code_v_tma = cute.make_layout(
                ((self.page_size, code_k_box_bytes), 1, self.code_stages_v),
                stride=(
                    (code_k_box_bytes, 1),
                    self.page_size * code_k_box_bytes,
                    code_stage_bytes_,
                ),
            )
        else:
            smem_layout_code_v_tma = cute.make_layout(
                ((mma_tile_m, code_v_row_bytes), 1, self.code_stages_v),
                stride=((code_v_row_bytes, 1), code_stage_bytes_,
                        code_stage_bytes_),
            )

        o_smem_dtype = mO_mnl.dtype
        smem_layout_atom_o = tcgen05.make_smem_layout_atom(
            tcgen05.mma.SmemLayoutAtomKind.MN_SW128, o_smem_dtype
        )
        smem_layout_o = cute.tile_to_shape(
            smem_layout_atom_o, (max(blk_tile_d, mma_tile_m), mma_tile_n), order=(1, 0)
        )
        smem_layout_o = cute.flat_divide(smem_layout_o, (mma_tile_m, mma_tile_n))

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_store_op = (
            cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp()
            if self.do_atomic_red
            else cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        )

        # Construct multimode gmem tiler
        tma_tile_n = (blk_tile_h, mma_tile_n // blk_tile_h)
        tma_tile_mnk = (mma_tile_m, tma_tile_n, mma_tile_k)
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ_nkl,
            cute.select(smem_layout_q, mma_modes),
            tma_tile_mnk,
            tiled_mma_kq,
        )
        tma_atom_kc = tma_tensor_kc = tma_atom_vc = tma_tensor_vc = None
        tma_atom_sc = tma_tensor_sc = None
        mSC_flat = None
        # Packed-code TMA atoms: K box (64, mma_tile_k*bits//8), one
        # full-width box per page. The V box is the same on the main feeds
        # (V stored like K); the channel-row-shaped pools of the INT2 E4M3
        # feed and the fold-off arm use a (128, 64*bits//8) box per page.
        mKC_mkl = pool_view(mKq)   # (64, D*bits//8, (h_k, NP))
        # (64, D*bits//8, (h_k, NP)) for a token-row V pool, else
        # (D, 64*bits//8, (h_k, NP))
        mVC_mkl = pool_view(mVq)
        tma_atom_kc, tma_tensor_kc = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            mKC_mkl,
            smem_layout_code_k_box,
            (self.page_size, code_k_box_bytes),
        )
        tma_atom_vc, tma_tensor_vc = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            mVC_mkl,
            smem_layout_code_v_box,
            ((self.page_size, code_k_box_bytes) if self.v_token_rows
             else (mma_tile_m, self.page_size * self.bits // 8)),
        )
        # Scale-block TMA atom: one (64, 4) f32 box per (page, kv-head)
        # (16B inner box — the SM90 interleaved-scale-block TMA shape).
        # Issued by tma_qk alongside K (same -1 OOB page-index zero-fill
        # pattern, making the K-fold self-masking on OOB pages), consumed
        # by the softmax WG (K-fold + P-quant of tile i share the stage).
        if cutlass.const_expr(self.qoff["legacy"]):
            mSC_mkl = pool_view(mScale)  # (64, 4, (h_k, NP))
            smem_layout_scale_box = cute.make_layout(
                (self.page_size, 4), stride=(4, 1)
            )
            tma_atom_sc, tma_tensor_sc = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_op, mSC_mkl, smem_layout_scale_box,
                (self.page_size, 4)
            )
        else:
            # Flex: the block is FLAT [NP, HKV, block_tot]; the TMA box
            # covers only the RING prefix as (128, ring_tot/128) columns
            # of 128 contiguous elements (>= 16B inner box at bf16).
            # Channel sections ride the SAME per-page block but are read
            # by gmem LDG per group (mSC_flat below), not the ring.
            ring_cols = scale_ring_elems // 128
            mSC_flat = cute.make_tensor(
                mScale.iterator,
                cute.group_modes(
                    cute.select(mScale.layout, (2, 1, 0)), 1, 3
                ),
            )  # (block_tot, (h_k, NP)) — gmem LDG view (channel sections)
            mSC_mkl = cute.flat_divide(
                mSC_flat, (128,)
            )  # (128, block_cols, (h_k, NP))
            if cutlass.const_expr(scale_ring_elems > 0):
                smem_layout_scale_box = cute.make_layout(
                    (128, ring_cols), stride=(1, 128)
                )
                tma_atom_sc, tma_tensor_sc = (
                    cute.nvgpu.cpasync.make_tiled_tma_atom(
                        tma_load_op, mSC_mkl, smem_layout_scale_box,
                        (128, ring_cols)
                    )
                )
        # Where `o_direct_for` holds, the correction warpgroup stores O through the
        # plain-stride view `mO_mnl` (passed as `mOraw`), and no O TMA descriptor is built.
        # The four places that differ between the two kinds of build key on which of the
        # two exists: the TMA-VO warp's O store, the correction's global views, and the
        # per-item and exit waits for the store's read.
        tma_atom_o, tma_tensor_o, mOraw = None, None, None
        if cutlass.const_expr(self.o_direct_for(q_bshd.shape[2] // h_k,
                                                q_descale is not None)):
            mOraw = mO_mnl
        else:
            tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_store_op,
                mO_mnl,
                cute.select(smem_layout_o, mode=[0, 1]),
                tma_tile_mnk[:2],
            )

        # GEMM view for LSE output
        # ((h_g, s_q), (h_k, b)). NOT handed to a split build's decode kernel, which never
        # stores it: `launch_reduction` takes `l_bsh` itself, so a split build with an LSE
        # launches the same decode kernel as one without.
        mL_nl = (None if l_bsh is None or self.do_kernel_red
                 else GqaDecode.gemm_view_bsh(l_bsh, h_k))
        assert l_bsh is None or l_bsh.dtype == acc_dtype

        # GEMM views for workspace tensors
        mM_partial_nl = mL_partial_nl = None
        if cutlass.const_expr(self.do_kernel_red):
            assert (
                m_partial_bsh.dtype
                == l_partial_bsh.dtype
                == o_partial_bshd.dtype
                == acc_dtype
            )

            # ((h_g, s_q), (h_k, b), kv_splits)
            mM_partial_nl = GqaDecode.gemm_view_bsh(m_partial_bsh, h_k)
            mL_partial_nl = GqaDecode.gemm_view_bsh(l_partial_bsh, h_k)

        ##############################
        # Launch kernel(s)
        ##############################
        scale_s_log2_e = scale_s * log2_e

        # BLASST threshold is normalized per-CTA inside `kernel` using the
        # CTA's batch seqlen. Precompute log2 of the host-side scale factor
        # here so the kernel just subtracts log2(seqlen) per CTA.
        enable_blasst = threshold_scale_factor is not None
        log2_threshold_scale_factor = (
            Float32(cute.math.log2(threshold_scale_factor)) if enable_blasst else None
        )

        n_tiles = cute.ceil_div(mQ_nkl.shape[0], (blk_tile_h, blk_tile_p))
        grid_y = cute.size(n_tiles)
        grid_z = cute.size(mQ_nkl.shape[2])  # l tiles
        grid_x = 1 if self.do_none_red else kv_splits
        if cutlass.const_expr(self.persistent):
            # Static round-robin persistence: one resident CTA per SM
            # (216KB smem => 1 CTA/SM) loops items bid_z, bid_z+grid_z, ...
            # (grid_y > 1 stays correct — the item loop strides z only, so
            # each (y, z-chain) pair is still covered by exactly one CTA.)
            grid_z = cutlass.min(grid_z, Int32(self.persistent_sm_count))
        grid = (grid_x, grid_y, grid_z)
        if cutlass.const_expr(graph_split):
            # One CTA per work-list slot, on z, where the split build puts its items: the
            # grid a CUDA graph captures is the same for every plan.
            grid = (1, 1, mWork.shape[0])
        cluster_x = kv_splits if self.do_atomic_red else 1

        self.kernel(
            # MMA
            blk_tile_shpd,
            mma_tile_mnk,
            tiled_mma_kq,
            tiled_mma_vp,
            mma_dtype,
            o_smem_dtype,
            # Page Table
            seqlens.iterator if is_varlen else seqlens,
            table_offsets.iterator if is_varlen and not graph_split else None,
            page_table.iterator,
            # K / V operand ring smem layouts
            smem_layout_k_mma,
            smem_layout_k_tma,
            smem_layout_v_mma,
            # Code ring
            smem_layout_code_k_tma,
            smem_layout_code_v_tma,
            # Packed-code TMA
            tma_atom_kc,
            tma_tensor_kc,
            tma_atom_vc,
            tma_tensor_vc,
            tma_atom_sc,
            tma_tensor_sc,
            mSC_flat,
            # Debug folded-S tap (debug_s mode)
            mDbgS,
            # Q
            smem_layout_q,
            tma_atom_q,
            tma_tensor_q,
            # Plain-stride Q view (the QSum reads; the TMA tensor above is a
            # coordinate tensor and cannot be load()ed)
            mQ_nkl,
            # O
            smem_layout_o,
            tma_atom_o,
            tma_tensor_o,
            mOraw,
            # Rest
            mL_nl,
            mL_partial_nl,
            mM_partial_nl,
            scale_s_log2_e,
            scale_o,
            pscale_lp,
            mVMaxHist,
            q_descale,
            log2_threshold_scale_factor,
            mWork,
            max_pages,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[cluster_x, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=enable_pdl,
        )

        if cutlass.const_expr(self.do_kernel_red):
            GqaDecode.launch_reduction(
                self.head_dim,
                o_bshd,
                l_bsh,
                o_partial_bshd,
                l_partial_bsh,
                m_partial_bsh,
                scale_o,
                stream,
                enable_pdl,
                mSplitCounts,
            )

    @cute.jit
    def _tiled_mma_kq(self, mma_dtype, acc_dtype, mma_tile_mnk):
        """The BMM1 (S = K_operand @ Q) MMA atom. HOST-SIDE ONLY -- see `_tiled_mma_vp`.

        `_launch_decode` is the ONLY caller. `_mma_kq_warp` must NOT call this: it takes
        the marshalled atom as a parameter, mutates that, and returns it.

        With `dequant_int2_to_f4`, A is packed e2m1 read directly from SMEM
        (silicon-exact, the packed_p64 panel image); B stays e4m3.
        """
        a_dtype_kq = Float4E2M1FN if self.dequant_int2_to_f4 else mma_dtype
        return sm100_utils.make_trivial_tiled_mma(
            a_dtype_kq,
            mma_dtype,
            OperandMajorMode.K,  # K
            OperandMajorMode.K,  # Q
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
            tcgen05.OperandSource.SMEM,
        )

    @cute.jit
    def _tiled_mma_vp(self, mma_dtype, acc_dtype, mma_tile_mnk):
        """The BMM2 (O = V @ P) MMA atom. HOST-SIDE ONLY: `_launch_decode` is the only caller.

        DO NOT CALL THIS FROM A WARP ROLE, and do not construct an atom inside the
        kernel by any other route. `_launch_decode` builds both atoms and they reach
        `kernel` as marshalled parameters, so their accumulate field is a RUNTIME
        value the kernel loads out of param space. An atom built inside the kernel
        has that field as a trace-time Python value instead, so the
        `ACCUMULATE, False` that makes the first K tile overwrite tmem and the
        `ACCUMULATE, True` that makes the rest accumulate fold to ONE constant, the
        first tile accumulates into the previous item's leftover tmem, and the kernel
        returns nondeterministic garbage in a narrow set of shapes.

        A static role check fails any role that rebuilds a marshalled parameter.

        The DSL constraint that made in-callee construction look correct is real but
        has a different fix: a value the enclosing `cutlass.range` item loop carries
        and a callee mutates cannot be left in the callee, because the op defining it
        then sits in a child region and IR verification fails with "operand #N does
        not dominate this use" at the loop's `scf.yield`. The fix is for the callee to
        RETURN the mutated value so the caller rebinds its own name, which is what
        `_mma_kq_warp` and `_mma_vp_warp` do.

        A = V is MN-major (channel-contiguous) in every configuration. On the
        main feeds V is stored exactly like K, and its shared-memory image is
        the K operand's image with rows and K swapped: SW128 E4M3 at INT4, the
        packed_p64 E2M1 panel image on the int2 f4 feed. With
        `dequant_int2_to_f4` this atom supplies the accumulator fragment, the P
        layout and the instruction fields, but `_mma_vp_warp` issues the MMA
        through inline PTX with hand-built descriptors, because the DSL (4.5.2
        and the pinned 4.8.0) does not legalize an MN-major sub-byte
        shared-memory descriptor (a silicon probe reproduces the refusal and
        adjudicates the hand-built route). B (P) is e4m3 MN-major.
        """
        a_dtype_vp = Float4E2M1FN if self.dequant_int2_to_f4 else mma_dtype
        return sm100_utils.make_trivial_tiled_mma(
            a_dtype_vp,
            mma_dtype,
            OperandMajorMode.MN,  # V (channel-contiguous in every configuration)
            OperandMajorMode.MN,  # P
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
        )

    @cute.jit
    def _mma_kq_warp(
        self,
        enable_blasst,
        iters_s,
        k_channel,
        k_consumer,
        k_parts,
        k_stages,
        k_tok64,
        kv_stage_bytes,
        mma_order_kq_nbar,
        mma_order_vp_nbar,
        mma_tile_k,
        mma_tile_m,
        prefetch_iters,
        q_load_mbar,
        qp_consumer,
        sQP,
        s_producer,
        sp_consumer,
        tAsK,
        tBsQ,
        tCtS,
        tCtS2,
        thrblk_mma_kq,
        tiled_mma_kq,
        tiles_dk,
        tiles_sm,
    ):
        """The BMM1 (S = K_operand @ Q) MMA warp.

        It owns its own tile loop and walks the k/kv/qp/s/sp rings. It does NOT
        own its MMA atom: `tiled_mma_kq` arrives as a parameter, this role mutates
        that one and returns it, and building one here instead is the bug
        `_tiled_mma_vp`'s docstring records. The participant objects likewise come
        in as parameters, because `kernel` builds them once per item.
        """
        # Read straight off `self` instead of taking 21 parameters that were
        # only ever `X = self.X` in the caller. Same expression, same object,
        # and the body below is untouched -- these are the names it uses.
        f4_feed = self.dequant_int2_to_f4

        # `tiled_mma_kq` is the CALLER'S atom, marshalled into `kernel` as a
        # parameter. This role mutates that one and returns it; it must never build
        # its own. `_tiled_mma_vp`'s docstring has the full reason and the bug.
        # Setup mma descriptors.
        if cutlass.const_expr(f4_feed):
            # A = packed e2m1 straight from SMEM (silicon-exact
            # packed_p64). The ring stage holds the
            # canonical K-major b4x16_p64 panel image: panel p (2048B) =
            # the 16B chunks of all 128 token rows covering dims
            # [16p, 16p+16), row r's chunk at byte r*16; nibbles
            # LSB-first in bytes 0-7 of each chunk, bytes 8-15 pad.
            # Fragment layout in f4 nibble-slot units over a
            # SWIZZLE-FREE recast of the same ring bytes (recast_ptr
            # without a swizzle argument strips the SW128 the e4m3
            # alloc view carries): ((M, (16, 2)), 1, #MMA_K, stages) —
            # 16 elems/chunk stride 1, 2 panels per K=32 atom stride
            # 4096 slots (2048B), rows stride 32 slots (16B), atoms
            # stride 8192 slots, stages stride 2*kv_stage_bytes slots.
            # make_umma_smem_desc derives LBO/SBO correctly on this
            # path (probe finding 3 — no manual desc patch needed).
            # Split operand rings: the K ring is the FIRST k_stages
            # stages of the physical ring — same base, same strides,
            # shallower stage mode; kq_ring walks the K pipeline only.
            sKf4 = cute.make_tensor(
                cute.recast_ptr(tAsK.iterator, dtype=Float4E2M1FN),
                cute.make_layout(
                    ((mma_tile_m, (16, 2)), 1, mma_tile_k // 32, k_stages),
                    stride=(
                        (32, (1, 4096)),
                        0,
                        8192,
                        2 * kv_stage_bytes,
                    ),
                ),
            )
            tAsK_desc = thrblk_mma_kq.make_fragment_A(sKf4)
            kq_ring = k_consumer
        else:
            # e4m3 split operand rings: the K ring is the FIRST k_stages
            # physical stages, so ring-local indices coincide with
            # physical stage indices in the full-extent e4m3 fragment and
            # the view needs no rebase.
            tAsK_desc = thrblk_mma_kq.make_fragment_A(tAsK)
            kq_ring = k_consumer
        tBsQ_desc = thrblk_mma_kq.make_fragment_B(tBsQ)
        if cutlass.const_expr(k_channel):
            # channel-K: B comes from the Q'-plane ring instead of sQ
            tBsQp_desc = thrblk_mma_kq.make_fragment_B(sQP)

        # Wait for Q
        cute.arch.mbarrier_wait(q_load_mbar, phase=0)

        # Sequence loop
        for s in cutlass.range(iters_s):
            s_token = s_producer.try_acquire()

            mma_order_vp_nbar.arrive_and_wait()
            k_token = kq_ring.try_wait()

            s_handle = s_producer.acquire_and_advance(s_token)
            # Q'-plane acquire, three mutually exclusive shapes:
            #   k_parts == 1            -> one plane per SEQUENCE tile
            #   k_parts == tiles_sm > 1 -> one plane per M-tile
            #   k_parts == 2*tiles_sm   -> two planes per M-tile (straddle)
            # The `k_parts > 1` is load-bearing, not defensive: at
            # sequence_tile == mma_tile_m (tiles_sm == 1) with Gt == 128,
            # `k_parts == 1` and `k_parts == tiles_sm` are BOTH true, and
            # without it this warp would acquire and release the plane
            # ring twice per tile against one produced stage. It is a
            # no-op at the shipped sequence_tile=256, where tiles_sm == 2.
            if cutlass.const_expr(k_channel and k_parts == 1):
                qp_handle = qp_consumer.wait_and_advance()
            for sm in cutlass.range_constexpr(tiles_sm):
                tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, False)
                if cutlass.const_expr(
                    k_channel and k_parts == tiles_sm and k_parts > 1
                ):
                    qp_handle = qp_consumer.wait_and_advance()
                for dk in cutlass.range_constexpr(tiles_dk):
                    k_handle = kq_ring.wait_and_advance(k_token)
                    is_last_iter = sm == tiles_sm - 1 and dk == tiles_dk - 1
                    if is_last_iter:
                        mma_order_kq_nbar.arrive()
                    else:
                        k_token = kq_ring.try_wait()

                    if cutlass.const_expr(
                        k_channel and k_parts == 2 * tiles_sm
                    ):
                        # K channel-64: two groups per M-tile — run
                        # the full-D chain TWICE over the SAME K
                        # stage, one Q' plane each, into tCtS/tCtS2
                        # (the softmax selects per token page-half).
                        mmas_k2 = cute.size(tAsK.shape[2])
                        for gsub in cutlass.range_constexpr(2):
                            qp_h2 = qp_consumer.wait_and_advance()
                            tiled_mma_kq.set(
                                tcgen05.Field.ACCUMULATE, False
                            )
                            for mma_k in cutlass.range_constexpr(
                                mmas_k2
                            ):
                                if cutlass.const_expr(gsub == 0):
                                    cS_g = tCtS[
                                        mma_dice
                                        + (sm, s_handle.index)
                                    ]
                                else:
                                    cS_g = tCtS2[
                                        mma_dice
                                        + (sm, s_handle.index)
                                    ]
                                cute.gemm(
                                    tiled_mma_kq,
                                    cS_g,
                                    tAsK_desc[
                                        None, None, mma_k,
                                        k_handle.index,
                                    ],
                                    tBsQp_desc[
                                        None, None, mma_k,
                                        qp_h2.index,
                                    ],
                                    cS_g,
                                )
                                if cutlass.const_expr(mma_k == 0):
                                    tiled_mma_kq.set(
                                        tcgen05.Field.ACCUMULATE,
                                        True,
                                    )
                            qp_h2.release()
                        k_handle.release()
                        continue

                    mmas_k = cute.size(tAsK.shape[2])
                    for mma_k in cutlass.range_constexpr(mmas_k):
                        if cutlass.const_expr(k_channel):
                            b_desc = tBsQp_desc[
                                None, None, mma_k, qp_handle.index
                            ]
                        else:
                            b_desc = tBsQ_desc[None, None, mma_k, dk]
                        if cutlass.const_expr(k_tok64):
                            # dual D-half chains: lo (mma_k <
                            # mmas_k/2) -> tCtS, hi -> tCtS2, each
                            # chain fresh-started
                            if cutlass.const_expr(
                                mma_k == mmas_k // 2
                            ):
                                tiled_mma_kq.set(
                                    tcgen05.Field.ACCUMULATE, False
                                )
                            if cutlass.const_expr(
                                mma_k < mmas_k // 2
                            ):
                                cS_t = tCtS[
                                    mma_dice + (sm, s_handle.index)
                                ]
                            else:
                                cS_t = tCtS2[
                                    mma_dice + (sm, s_handle.index)
                                ]
                        else:
                            cS_t = tCtS[mma_dice + (sm, s_handle.index)]
                        cute.gemm(
                            tiled_mma_kq,
                            cS_t,
                            tAsK_desc[None, None, mma_k, k_handle.index],
                            b_desc,
                            cS_t,
                        )
                        if dk == 0 and mma_k == 0:
                            tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, True)
                        if cutlass.const_expr(k_tok64):
                            if cutlass.const_expr(mma_k == mmas_k // 2):
                                tiled_mma_kq.set(
                                    tcgen05.Field.ACCUMULATE, True
                                )
                    k_handle.release()
                if cutlass.const_expr(
                    k_channel and k_parts == tiles_sm and k_parts > 1
                ):
                    qp_handle.release()
            if cutlass.const_expr(k_channel and k_parts == 1):
                qp_handle.release()
            s_handle.commit()

            # Skip-predicate ring (BLASST). The K ring holds no V stages,
            # so this warp has nothing to walk past once MMA-VP starts;
            # the predicate is consumed so the sp ring stays balanced.
            if s >= prefetch_iters:
                if cutlass.const_expr(enable_blasst):
                    sp_handle = sp_consumer.wait_and_advance()
                    sp_handle.release()

        # Tail loop
        for s in cutlass.range_constexpr(prefetch_iters):
            mma_order_vp_nbar.arrive_and_wait()
            mma_order_kq_nbar.arrive()
            if cutlass.const_expr(enable_blasst):
                if s < min(prefetch_iters, iters_s):
                    sp_handle = sp_consumer.wait_and_advance()
                    sp_handle.release()

        # Hand the mutated atom back: the accumulate field this role toggles
        # is the caller's runtime state, and the caller's item loop carries it.
        return tiled_mma_kq

    @cute.jit
    def _close_empty_item(
        self,
        _item,
        role,
        mma_order_vp_nbar,
        tma_order_v_nbar,
        sM_consumer_nbar,
        sL_final_nbar,
        sM_final_nbar,
        tL_consumer_nbar,
        v_channel,
        with_phase,
    ):
        """What one role of a persistent CTA does for a work item with no tile.

        The named barriers carry their counts from item to item, so an empty item must
        leave each one as a full item does (the PERSISTENT ITEM CLOSURE note in
        `kernel`). `role` is a trace-time string naming the calling role:

        * "correction": meet `reduction_none`'s final barriers as a full item's
          correction does, `sL_final` then `sM_final` (the caller has already arrived on
          `sM_final` once), because the reduction warp runs all three for every item;
        * every role, when this is the CTA's first item: the one-time arrivals the role
          makes on item 0 and every later item relies on (MMA-VP's `mma_order_vp`,
          TMA-VO's `tma_order_v`, the correction's `sM_consumer` and every
          `tL_consumer` phase), which the role's own method never runs for an empty item.
        """
        if cutlass.const_expr(role == "correction"):
            sL_final_nbar.arrive_and_wait()
            sM_final_nbar.arrive_and_wait()
        if _item == 0:
            if cutlass.const_expr(role == "mma_vp"):
                mma_order_vp_nbar.arrive()
            elif cutlass.const_expr(role == "tma_vo"):
                tma_order_v_nbar.arrive()
            else:
                sM_consumer_nbar.arrive()
                if cutlass.const_expr(not v_channel):
                    for phase in cutlass.range_constexpr(self.o_stages):
                        with_phase(tL_consumer_nbar, phase).arrive()

    @cute.jit
    def _graph_work_entry(self, mWork, slot):
        """`(item, split, k, seq_len)` of the graph split build's work-list slot `slot`, each the
        same in every thread of the CTA: one 16-byte load per thread, and a `make_warp_uniform`
        on each of the three coordinates. A slot past the end of the list (item -1) exits the
        CTA here, before anything else runs (`_exit_if_negative`). These are work coordinates
        that stand where the split build reads its block indices, not warp-role indices, and
        the graph split build is never persistent (`_launch_decode` asserts it)."""
        rWork = cute.make_rmem_tensor(4, Int32)
        cute.copy(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=128),
            cute.make_tensor(mWork[slot, None].iterator.align(16), cute.make_layout(4)),
            rWork)
        item = cute.arch.make_warp_uniform(rWork[0])
        _exit_if_negative(item)
        return (item, cute.arch.make_warp_uniform(rWork[1]),
                cute.arch.make_warp_uniform(rWork[2]), rWork[3])

    @cute.jit
    def _vmax_hist_loads(self, mVMaxHist, lane_idx, vec=False):
        """Issue the loads of the cache's count table for `_pool_pscale_lp`, one warp.

        Load i of lane l reads bin 32 i + l, so each of the eight loads is one coalesced
        128-byte line, and all eight issue before any is used. With `vec` (the softmax
        warpgroup where `vmax_hist_vec` holds), each lane instead issues two 16-byte
        loads, of bins 4 l to 4 l + 3 and
        128 + 4 l to 128 + 4 l + 3, so that the warp reads the 1 KiB table in two
        instructions; word i of lane l is then bin 4 l + i % 4 + 128 (i // 4). The
        caller has run `griddepcontrol.wait` (PDL: the cache writer, which keeps the
        table, may be the previous grid), and can do other work before it takes the
        exponent, so that the loads' latency overlaps that work.
        """
        words = []
        if cutlass.const_expr(vec):
            for j in cutlass.range_constexpr(VMAX_BINS // (4 * warp_threads)):
                r4 = cute.make_rmem_tensor(4, Uint32)
                g4 = cute.make_tensor(
                    (mVMaxHist.iterator + 4 * lane_idx + j * 4 * warp_threads).align(16),
                    cute.make_layout(4))
                cute.autovec_copy(g4, r4)
                for e in cutlass.range_constexpr(4):
                    words.append(r4[e])
        else:
            for i in cutlass.range_constexpr(VMAX_BINS // 32):
                words.append(mVMaxHist[32 * i + lane_idx])
        return words

    @cute.jit
    def _pool_pscale_lp(self, words, lane_idx, lp_shift, vec=False):
        """The pool's P-scale exponent from the table words `_vmax_hist_loads` issued.

        Bin b of `mVMaxHist` counts the page maxima whose `fold_runtime.vmax_bucket` is
        b, so the highest nonzero bin is the bin of the largest s_v in the pool, and
        lp = `pscale_lp_of_bucket` of it, which is floor(log2(448 / max s_v)),
        taken no lower than at 1e-30 (`PSCALE_FLOOR_BUCKET`), the same floor the
        reference applies as max(max s_v, 1e-30), so an empty table gives 108. The
        warp takes the maximum,
        so every lane returns the same lp plus the feed's re-anchor `lp_shift`, an f32.
        `vec` must be the value the words were loaded with.
        """
        top = Int32(PSCALE_FLOOR_BUCKET)
        for i in cutlass.range_constexpr(VMAX_BINS // 32):
            if words[i] != Uint32(0):
                if cutlass.const_expr(vec):
                    top = cutlass.max(top, 4 * lane_idx + i % 4 + (i // 4) * 4 * warp_threads)
                else:
                    top = cutlass.max(top, 32 * i + lane_idx)
        top = cute.arch.warp_redux_sync(top, kind="max")
        return Float32(pscale_lp_of_bucket(top) + lp_shift)

    @cute.jit
    def _mma_vp_warp(
        self,
        _item,
        enable_blasst,
        iters_s,
        k_stages,
        kv_stage_bytes,
        mma_order_kq_nbar,
        mma_order_vp_nbar,
        mma_tile_k,
        mma_tile_m,
        o_producer,
        p_consumer,
        prefetch_iters,
        sSP_i32,
        sp_consumer,
        tAsK,
        tAsV,
        tBsP_nk,
        tCtO,
        thrblk_mma_vp,
        tiled_mma_vp,
        tiles_dm,
        tiles_sk,
        tmem_ptr,
        v_channel,
        v_consumer,
        v_parts,
        v_stages,
    ):
        """The BMM2 (O = V_operand @ P) MMA warp.

        It owns its own tile loop and walks the kv/v/p/sp/o rings; the participant
        objects come in as parameters because `kernel` builds them once per item.
        """
        # Read straight off `self` instead of taking 21 parameters that were
        # only ever `X = self.X` in the caller. Same expression, same object,
        # and the body below is untouched -- these are the names it uses.
        f4_feed = self.dequant_int2_to_f4
        o_stages = self.o_stages
        persistent = self.persistent
        tmem_alloc_cols = self.tmem_alloc_cols

        # Setup mma descriptors. `tiled_mma_vp` is the caller's atom, marshalled
        # in as a kernel parameter, and this role MUST mutate that one for the
        # reason `_mma_kq_warp` records: an atom built inside the kernel folds
        # its accumulate field to a trace-time constant, and the runtime
        # first-tile selection below then stops selecting. The mutated atom goes
        # back to the caller through the return below.
        tiled_mma_vp.set(tcgen05.Field.ACCUMULATE, True)
        # On the f4 feed the V pool is stored exactly like K (`v_token_rows`
        # is unconditional there), so V is an MN-major E2M1 operand.
        vp_hand = f4_feed
        if cutlass.const_expr(vp_hand):
            # Token-row V pool: A = V^T as packed e2m1, MN-MAJOR, read from the
            # image the V stream deposits with the K stream's partition: chunk
            # (token k, channel panel i) of a stage at byte i*16*mma_tile_k +
            # k*16, 16 channel nibbles LSB-first in bytes 0-7, bytes 8-15 pad.
            # That is the MN-major no-swizzle canonical layout (core matrix = 8
            # tokens x one 16B chunk), so LBO = 8 tokens = 128 B and SBO = one
            # channel panel = 16*mma_tile_k B. The DSL (4.5.2 and the pinned
            # 4.8.0) cannot legalize an MN-major sub-byte descriptor
            # (a silicon probe reproduces the refusal), so the descriptors are built
            # here from the vendored CUTLASS encoders and the instruction is issued
            # through inline PTX. That probe adjudicated this image, both stride
            # fields and the transpose bit on silicon, with negative controls for
            # each.
            vp_a_desc = sm100_desc.make_smem_desc_base(
                cute.make_layout((mma_tile_m // 16, mma_tile_k),
                                 stride=(mma_tile_k, 1)),
                cute.make_swizzle(0, 4, 3),
                sm100_desc.Major.MN,
            )
            vp_a_hi = vp_a_desc >> 32
            vp_b_desc = sm100_desc.smem_desc_base_from_tensor(
                tBsP_nk, sm100_desc.Major.MN)
            vp_b_lo_c, vp_b_hi = vp_b_desc & 0xFFFFFFFF, vp_b_desc >> 32
            # The vendored encoder has no E2M1 entry: encode the atom's own
            # fields with the B type in the A slot, then set A's format field
            # (bits 7-9) to E2M1.
            vp_op = tiled_mma_vp.op
            assert vp_op.a_dtype is Float4E2M1FN and (
                vp_op.a_major_mode == OperandMajorMode.MN
                and vp_op.b_major_mode == OperandMajorMode.MN), (
                "hand-issued VP expects A = MN-major E2M1, B = MN-major P")
            vp_idesc = sm100_desc.make_instr_desc(
                vp_op.b_dtype, vp_op.b_dtype, vp_op.acc_dtype,
                vp_op.shape_mnk[0], vp_op.shape_mnk[1],
                sm100_desc.Major.MN, sm100_desc.Major.MN)
            vp_idesc = (vp_idesc & ~(0x7 << 7)) | (
                int(sm100_desc.MXF8F6F4Format.E2M1) << 7)
            # the V ring is the last v_stages stages of the operand ring
            vp_a_start = Int32(vp_a_desc & 0xFFFFFFFF) | (
                sm100_desc.make_smem_desc_start_addr(
                    cute.recast_ptr(tAsK.iterator, dtype=Uint8)
                    + k_stages * kv_stage_bytes))
            # K-step of one instruction (32 tokens) and one stage, in the
            # descriptor's 16-byte units
            vp_a_kstep = (32 * 16) >> 4
            vp_a_stage = kv_stage_bytes >> 4
            vp_ring = v_consumer
        else:
            # e4m3 split operand rings: the V ring is the LAST v_stages
            # physical stages. The fragment stays full-extent over the
            # e4m3 view, and the gemm below adds the constexpr k_stages
            # offset to the V-ring-local index, which is the same physical
            # stage the f4 arm reaches by rebasing its own view.
            # V is MN-major here (MN_SW128 at INT4 and on the INT2 E4M3
            # feed), so the descriptor is the DSL's for that layout.
            tAsV_desc = thrblk_mma_vp.make_fragment_A(tAsV)
            vp_ring = v_consumer
        # The V ring's first physical stage within the e4m3 A fragment: the
        # fragment is full-extent, so the V-ring-local index is offset past
        # the K ring.
        v_stage_off = k_stages
        tBsP_desc = thrblk_mma_vp.make_fragment_B(tBsP_nk)
        if cutlass.const_expr(vp_hand):
            # One MMA instruction K-step along P, from the fragment's own layout
            # (the same crd2idx walk the vendored gemm_ptx takes over sB).
            assert mma_tile_k // 32 == cute.size(tBsP_nk.shape[2]), (
                "hand-issued VP expects one 32-token instruction per P k-mode")
            # The P descriptor's start, in the descriptor's 16-byte units, is the
            # stage-0 word plus trace-time steps: one per P stage and one per
            # (k step, sk chunk). A tile then forms its stage's word with one
            # multiply-add on the stage index after the P wait, the form the atom
            # path's descriptor takes, and each instruction adds one constant.
            # Deriving the start from a slice's address instead puts a shift, a mask
            # and an OR on the issue path after the P wait.
            # The start field is 14 bits wide and every shared-memory address fits
            # it, so the adds never carry into the stride fields above it.
            vp_b_kofs = []
            for _sk in cutlass.range_constexpr(cute.size(tBsP_nk.shape[3])):
                for _k in cutlass.range_constexpr(cute.size(tBsP_nk.shape[2])):
                    _ofs = cute.crd2idx((0, 0, _k, _sk, 0), tBsP_nk.layout)
                    assert isinstance(_ofs, int) and _ofs % 16 == 0, _ofs
                    vp_b_kofs.append(_ofs >> 4)
            vp_b_nk = cute.size(tBsP_nk.shape[2])
            _pst = cute.crd2idx((0, 0, 0, 0, 1), tBsP_nk.layout)
            assert isinstance(_pst, int) and _pst % 16 == 0, _pst
            vp_b_pstep = _pst >> 4
            vp_b_base = Int32(vp_b_lo_c) | sm100_desc.make_smem_desc_start_addr(
                tBsP_nk[None, None, 0, 0, 0].iterator)

        # Prefetch loop.
        # One-shot prime on item 0 only (same dangling-arrive
        # roll-over argument as the TMA-VO prime).
        if cutlass.const_expr(persistent):
            if _item == 0:
                mma_order_vp_nbar.arrive()
        else:
            mma_order_vp_nbar.arrive()
        for s in cutlass.range_constexpr(prefetch_iters):
            mma_order_kq_nbar.arrive_and_wait()
            mma_order_vp_nbar.arrive()

        # Sequence loop. The V ring holds no K stages, so there is no
        # position to keep against MMA-KQ; only the ordering nbar ping-pong
        # couples the two MMA warps. The hand-issued tile body is kept one tile
        # per iteration, as the atom path's is: LLVM unrolled it over two tiles
        # in some channel-V builds.
        vp_unroll = 1 if vp_hand else -1
        for s in cutlass.range(iters_s, unroll=vp_unroll):
            keep_tile = not enable_blasst
            if cutlass.const_expr(enable_blasst):
                sp_handle = sp_consumer.wait_and_advance()
                keep_tile = sSP_i32[sp_handle.index] != 0
                sp_handle.release()
                if not keep_tile:
                    mma_order_kq_nbar.arrive_and_wait()
                    mma_order_vp_nbar.arrive()

            if keep_tile:
                p_token = p_consumer.try_wait()
                o_token = o_producer.try_acquire()

                mma_order_kq_nbar.arrive_and_wait()
                v_token = vp_ring.try_wait()

                p_handle = p_consumer.wait_and_advance(p_token)
                o_handle = o_producer.acquire_and_advance(o_token)
                if cutlass.const_expr(vp_hand):
                    vp_b_stage = vp_b_base + p_handle.index * vp_b_pstep
                # channel-V: FRESH-write each group PART's own O slot
                # (the correction WG DRAINS every slot into a register
                # accumulator applying that group's per-channel scale,
                # so a slot must hold one part's PV product alone).
                # Parts map to sk chunks (Gt=128) and to mma_k
                # half-ranges within a chunk (Gt=64); part boundaries
                # commit the running slot and acquire the next one —
                # the KQ restart idiom per part. v_parts == 1 (or
                # legacy) keeps the single per-tile slot.
                for sk in cutlass.range_constexpr(tiles_sk):
                    for dm in cutlass.range_constexpr(tiles_dm):
                        v_handle = vp_ring.wait_and_advance(v_token)
                        is_last_iter = sk == tiles_sk - 1 and dm == tiles_dm - 1
                        if is_last_iter:
                            mma_order_vp_nbar.arrive()
                        else:
                            v_token = vp_ring.try_wait()

                        mmas_k = cute.size(tAsV.shape[2])
                        # mma_k steps per part within this sk chunk
                        psk = v_parts // tiles_sk
                        pks = mmas_k // psk if psk >= 1 else mmas_k
                        for mma_k in cutlass.range_constexpr(mmas_k):
                            if cutlass.const_expr(v_channel):
                                part_start = (
                                    (mma_k % pks == 0)
                                    if psk >= 1
                                    else (sk == 0 and mma_k == 0)
                                )
                                if cutlass.const_expr(part_start):
                                    if cutlass.const_expr(
                                        not (sk == 0 and mma_k == 0)
                                    ):
                                        o_handle.commit()
                                        o_handle = (
                                            o_producer.acquire_and_advance()
                                        )
                                    tiled_mma_vp.set(
                                        tcgen05.Field.ACCUMULATE, False
                                    )
                            if cutlass.const_expr(vp_hand):
                                # Same accumulate rule as the atom path: the
                                # atom carries True except where a channel-V
                                # part starts (set False around the gemm).
                                vp_acc = 0 if (v_channel and part_start) else 1
                                vp_a_lo = vp_a_start + (
                                    v_handle.index * vp_a_stage
                                    + mma_k * vp_a_kstep)
                                vp_b_lo = vp_b_stage + vp_b_kofs[
                                    sk * vp_b_nk + mma_k]
                                vp_d = tCtO[mma_dice + (dm, o_handle.index)
                                            ].iterator.toint()
                                with cute.arch.elect_one():
                                    _umma_ss_f8f6f4(
                                        vp_d,
                                        vp_a_lo, vp_a_hi, vp_b_lo, vp_b_hi,
                                        vp_idesc, vp_acc)
                            else:
                                # V-ring-local index into the full-extent
                                # fragment: offset past the K ring
                                a_desc_vp = tAsV_desc[
                                    None, None, mma_k,
                                    v_stage_off + v_handle.index,
                                ]
                                cute.gemm(
                                    tiled_mma_vp,
                                    tCtO[mma_dice + (dm, o_handle.index)],
                                    a_desc_vp,
                                    tBsP_desc[None, None, mma_k, sk,
                                              p_handle.index],
                                    tCtO[mma_dice + (dm, o_handle.index)],
                                )
                            if cutlass.const_expr(v_channel):
                                if cutlass.const_expr(part_start):
                                    tiled_mma_vp.set(
                                        tcgen05.Field.ACCUMULATE, True
                                    )
                        v_handle.release()
                p_handle.release()
                o_handle.commit()

        # Wait for signal to dealloc tmem, then dealloc.
        # The correction tail reads o_stages buffers — commit empties
        # for the missing tiles when iters_s < o_stages. (channel-V:
        # the drain tail is per-REAL-virtual-tile with dynamic guards
        # — no empty commits exist and none are padded.)
        if cutlass.const_expr(not v_channel):
            for _ in cutlass.range(o_stages - iters_s):
                o_producer.commit()
                o_producer.advance()
        o_producer.tail()
        # In a persistent build tmem stays allocated across items; the
        # after-loop tail owns the single relinquish + dealloc.
        if cutlass.const_expr(not persistent):
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        # Hand the mutated atom back: the accumulate field this role toggles
        # is the caller's runtime state, and the caller's item loop carries it.
        return tiled_mma_vp

    @cute.jit
    def _tma_vo_warp(
        self,
        _item,
        batches,
        blk_tile_d,
        blk_tile_hp,
        codev_producer,
        coord_b,
        coord_hk,
        coord_hp,
        enable_blasst,
        iters_s,
        kv_split_idx,
        mO,
        mVC,
        mcast_coord,
        mcast_layout,
        mma_tile_k,
        mma_tile_m,
        mma_tile_n,
        pages_k,
        prefetch_iters,
        sCode,
        sO_final_nbar,
        sO_mma,
        sPT,
        sSP_i32,
        smem_layout_code_v_tma,
        sp_consumer,
        tiles_dm,
        tiles_sk,
        tma_atom_o,
        tma_atom_vc,
        tma_order_k_nbar,
        tma_order_v_nbar,
    ):
        """The TMA warp for V codes and the O store.

        It issues the V code-ring TMA loads for its own tile loop and, at the end of
        the item, copies the assembled O tile out to gmem and commits the copy as a
        bulk group, which it waits on at once in a non-persistent build and `kernel`
        waits on before the next item in a persistent one. Where `o_direct_for` holds, the
        correction warpgroup stores O itself and this warp stores nothing.
        """
        # Read straight off `self` instead of taking 21 parameters that were
        # only ever `X = self.X` in the caller. Same expression, same object,
        # and the body below is untouched -- these are the names it uses.
        bits = self.bits
        do_kernel_red = self.do_kernel_red
        page_size = self.page_size
        persistent = self.persistent
        pt_stages = self.pt_stages

        # Slice and partition V (TMA targets the CODE ring; the dequant WG
        # maps code stages into the operand ring)
        # Packed V codes: one box per page (and head-dim tile) = one whole
        # code stage per TMA op: K's (64, mma_tile_k*bits//8) box where V is
        # stored like K, else (mma_tile_m, 64*bits//8) over the channel-row
        # pool. Both move the same code-stage bytes. The V code ring's stages live
        # PAST the K ring's in the same flat sCode block, so the view is
        # rebased by code_stages_k stages (the layout already carries the
        # code_stages_v extent).
        code_v_row_bytes = page_size * bits // 8
        code_stage_bytes_v = page_size * mma_tile_k * bits // 8
        sVC = cute.make_tensor(
            sCode.iterator + self.code_stages_k * code_stage_bytes_v,
            smem_layout_code_v_tma,
        )  # ((BOX_ROWS, BOX_BYTES), 1, code_stages_v)
        if cutlass.const_expr(self.v_token_rows):
            # Token-row V: the K box, one 128-channel slab of a page per
            # stage; head-dim tile dm selects the slab's byte columns.
            gVC = cute.local_tile(
                mVC,
                (page_size, mma_tile_k * bits // 8),
                coord=(0, None, (coord_hk, None)),
            )  # (PAGE, BOX_BYTES, #TILE_DM, #PAGE_S)
        else:
            gVC = cute.local_tile(
                mVC,
                (mma_tile_m, code_v_row_bytes),
                coord=(None, 0, (coord_hk, None)),
            )  # (MMA_TILE_M, ROW_BYTES, #TILE_DM, #PAGE_S)
        sV_tma, gV_tma = cute.nvgpu.cpasync.tma_partition(
            tma_atom_vc,
            mcast_coord,
            mcast_layout,
            smem_tensor=sVC,
            gmem_tensor=cute.group_modes(gVC, 0, 2),
        )  # (TMA, Rest...)

        # Prefetch K loop. This one-shot prime fires
        # on item 0 only — the protocol's +1 arrive asymmetry leaves
        # exactly one dangling VO arrive at each item's end, which
        # rolls over as the next item's prime (re-priming per item
        # completes a spurious generation and shifts the K/V
        # interleave pairing off by one -> ring credit deadlock).
        if cutlass.const_expr(persistent):
            if _item == 0:
                tma_order_v_nbar.arrive()
        else:
            tma_order_v_nbar.arrive()
        for s in cutlass.range_constexpr(prefetch_iters):
            # The V code ring is this warp's alone, so there is no shared
            # position to maintain against TMA-QK; only the ordering nbar
            # ping-pong couples the two TMA warps.
            tma_order_k_nbar.arrive_and_wait()
            tma_order_v_nbar.arrive()

        # Sequence loop
        pt_index = 0
        for s in cutlass.range(iters_s):
            # Load skip predicate
            keep_tile = not enable_blasst
            if cutlass.const_expr(enable_blasst):
                sp_handle = sp_consumer.wait_and_advance()
                keep_tile = sSP_i32[sp_handle.index] != 0
                sp_handle.release()
                if not keep_tile:
                    tma_order_k_nbar.arrive_and_wait()
                    tma_order_v_nbar.arrive()

            if keep_tile:
                # Load page indices
                tma_order_k_nbar.arrive_and_wait()
                rPT = sPT[None, pt_index].load().reshape((pages_k, tiles_sk))

                # Load V
                codev_token = codev_producer.try_acquire()
                # Per-page code stages: acquire per (sk, dm, pk),
                # fill with 1 page box.
                for sk in cutlass.range_constexpr(tiles_sk):
                    for dm in cutlass.range_constexpr(tiles_dm):
                        for pk in cutlass.range_constexpr(pages_k):
                            code_handle = codev_producer.acquire_and_advance(
                                codev_token
                            )
                            is_last_iter = (
                                sk == tiles_sk - 1
                                and dm == tiles_dm - 1
                                and pk == pages_k - 1
                            )
                            if is_last_iter:
                                tma_order_v_nbar.arrive()
                            else:
                                codev_token = codev_producer.try_acquire()

                            virtual_page_idx = rPT[pk, sk]
                            cute.copy(
                                tma_atom_vc,
                                gV_tma[None, dm, virtual_page_idx],
                                sV_tma[None, 0, code_handle.index],
                                tma_bar_ptr=code_handle.barrier,
                            )
            pt_index = 0 if pt_index == (pt_stages - 1) else pt_index + 1

        if cutlass.const_expr(tma_atom_o is not None):
            # Slice and partition O
            # (TILE_D, TILE_H)
            coord_b_partial = (
                kv_split_idx * batches + coord_b if do_kernel_red else coord_b
            )
            gO = cute.local_tile(
                mO,
                tiler=(blk_tile_d, blk_tile_hp),
                coord=(0, coord_hp, (coord_hk, coord_b_partial)),
            )
            # (MMA_TILE_M, MMA_TILE_N, #TILE_DM, #TILE_HN=1)
            gO_mma = cute.flat_divide(gO, (mma_tile_m, mma_tile_n))
            # (TMA, #TILE_DM, #TILE_HN)
            sO_tma, gO_tma = cute.nvgpu.cpasync.tma_partition(
                tma_atom_o,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(sO_mma, 0, 2),
                gmem_tensor=cute.group_modes(gO_mma, 0, 2),
            )

            # Store O to gmem
            sO_final_nbar.arrive_and_wait()
            cute.copy(tma_atom_o, sO_tma, gO_tma)
            # Commit the store as a bulk group and wait for it to read sO. sO is the
            # first bytes of K-ring stage 0, which the next item's K deposit
            # overwrites, and the store reads it through the async proxy: the CTA
            # barriers ahead order only the store's issue, never its reads, so only
            # a bulk-group wait in the issuing thread says the read is over (PTX
            # ISA, cp.async.bulk.wait_group). `.read` waits for the shared-memory
            # reads alone, not for the global writes. A non-persistent build waits
            # here, before the CTA exits. A persistent build waits instead just
            # before the next item's initialization barrier and after the item loop
            # (`kernel`), so that the read overlaps the item boundary: waiting here
            # measured up to 1% slower at short lengths with many items per CTA.
            #
            # Every lane commits and waits. A bulk group belongs to the thread that
            # committed it, and in the PTX every lane of this warp executes the
            # plain store: the DSL 4.5.2 PTX carries no election for it (it does
            # elect the reducing store of `reduction="atomic"`), although ptxas
            # then issues one uniform store per box for the whole warp (one UTMASTG
            # in SASS, and L2 writes of about one O tile per item, not 32).
            # Do not wrap the store in elect_one: DSL 4.8.0 documents
            # that it elects TMA stores itself and warns that an outer elect_one
            # deadlocks, while an all-lane commit and wait is right either way.
            cute.arch.cp_async_bulk_commit_group()
            if cutlass.const_expr(not persistent):
                cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def _tma_qk_warp(
        self,
        blk_tile_d,
        blk_tile_hp,
        codek_producer,
        coord_hb,
        coord_hk,
        coord_hp,
        cpasync_atom,
        enable_blasst,
        iters_s,
        kv_split_idx,
        kv_splits,
        lane_idx,
        mKC,
        mQ,
        mSC,
        mcast_coord,
        mcast_layout,
        mma_tile_k,
        mma_tile_n,
        page_count,
        page_table_iter,
        pages_m,
        pages_s,
        prefetch_iters,
        q_load_mbar,
        qflex,
        sCode,
        sPT,
        sScale,
        scale_producer,
        scale_ring_elems,
        scale_ring_live,
        smem_layout_code_k_tma,
        sp_consumer,
        tBsQ,
        table_offset,
        thrblk_mma_kq,
        tiles_dk,
        tiles_sm,
        tma_atom_kc,
        tma_atom_q,
        tma_atom_sc,
        tma_order_k_nbar,
        tma_order_v_nbar,
    ):
        """The TMA warp for Q, K codes and the scale ring.

        It stages Q once, then issues the per-tile K code-ring and scale-ring TMA
        loads for its own tile loop. The scale box rides the same rPT-sourced -1 out-of-bounds page index as
        the code TMA, so an OOB page reads s = z = 0 and the K-fold self-masks it.
        """
        # Read straight off `self` instead of taking 21 parameters that were
        # only ever `X = self.X` in the caller. Same expression, same object,
        # and the body below is untouched -- these are the names it uses.
        bits = self.bits
        page_size = self.page_size
        pt_stages = self.pt_stages
        scale_stages = self.scale_stages

        # Slice and partition Q
        # (TILE_H, TILE_D)
        gQ = cute.local_tile(
            mQ,
            tiler=(blk_tile_hp, blk_tile_d),
            coord=(coord_hp, 0, coord_hb),
        )
        # (MMA_TILE_N, MMA_TILE_K, #TILE_DK)
        gQ_mma = cute.local_tile(gQ, (mma_tile_n, mma_tile_k), coord=(0, None))
        # (MMA, #MMA_N, #MMA_K, #TILE_DK)
        tBgQ = thrblk_mma_kq.partition_B(gQ_mma)
        # (TMA, #TILE_DK)
        tBsQ_tma, tBgQ_tma = cute.nvgpu.cpasync.tma_partition(
            tma_atom_q,
            mcast_coord,
            mcast_layout,
            smem_tensor=cute.group_modes(tBsQ, 0, 3),
            gmem_tensor=cute.group_modes(tBgQ, 0, 3),
        )

        # Slice and partition K (TMA targets the CODE ring; the dequant WG
        # maps code stages into the operand ring)
        # Packed K codes: one full-width (64, mma_tile_k*bits//8) box per
        # page = one code stage per TMA op, the same shape family as V.
        code_k_box_bytes = mma_tile_k * bits // 8
        sKC = cute.make_tensor(
            sCode.iterator, smem_layout_code_k_tma
        )  # ((PAGE, BOX_BYTES), #BOXES, code_stages)
        gKC = cute.local_tile(
            mKC,
            (page_size, code_k_box_bytes),
            coord=(0, None, (coord_hk, None)),
        )  # (PAGE, BOX_BYTES, #BOXES(tiles_dk), #PAGE_S)
        sK_tma, gK_tma = cute.nvgpu.cpasync.tma_partition(
            tma_atom_kc,
            mcast_coord,
            mcast_layout,
            smem_tensor=sKC,
            gmem_tensor=cute.group_modes(gKC, 0, 2),
        )  # (TMA, Rest...)
        # Scale blocks: one (64, 4) f32 box per page, issued
        # alongside K with the SAME rPT-sourced page index (so a -1
        # OOB page zero-fills the scale stage exactly like the code
        # stages: s_t = z_t = 0 -> folded S = 0, self-masking).
        if cutlass.const_expr(not qflex):
            sSC = cute.make_tensor(
                sScale.iterator,
                cute.make_layout(
                    ((page_size, 4), 1, scale_stages),
                    stride=((4, 1), page_size * 4, page_size * 4),
                ),
            )  # ((PAGE, 4), 1, scale_stages)
            gSC = cute.local_tile(
                mSC, (page_size, 4), coord=(0, 0, (coord_hk, None))
            )  # (PAGE, 4, #NP)
        elif cutlass.const_expr(scale_ring_elems > 0):
            # Flex ring box: (128, ring_tot/128) contiguous-column
            # prefix of the per-page block (channel sections are
            # NOT staged — the consuming WG LDGs them per group).
            ring_cols = scale_ring_elems // 128
            sSC = cute.make_tensor(
                sScale.iterator,
                cute.make_layout(
                    ((128, ring_cols), 1, scale_stages),
                    stride=(
                        (1, 128),
                        scale_ring_elems,
                        scale_ring_elems,
                    ),
                ),
            )  # ((128, RING_COLS), 1, scale_stages)
            gSC = cute.local_tile(
                mSC, (128, ring_cols), coord=(0, 0, (coord_hk, None))
            )  # (128, RING_COLS, #NP)
        if cutlass.const_expr(scale_ring_elems > 0):
            sSC_tma, gSC_tma = cute.nvgpu.cpasync.tma_partition(
                tma_atom_sc,
                mcast_coord,
                mcast_layout,
                smem_tensor=sSC,
                gmem_tensor=cute.group_modes(gSC, 0, 2),
            )  # (TMA, Rest...)

        # Construct page table for this batch
        gPT = cute.make_tensor(page_table_iter + table_offset, (page_count,))
        cPT = cute.make_identity_tensor(page_count)

        # Partition page table
        pt_load = cute.make_tiled_copy(
            cpasync_atom,
            # 1 thread per page
            cute.make_ordered_layout((pages_s, 1), order=(1, 0)),
            (pages_s,),
        )
        lane_load_page = lane_idx < pages_s
        thr_pt_load = pt_load.get_slice(lane_idx)
        tPTgPT = thr_pt_load.partition_S(gPT)  # (CPY=1, #CPY=#TILE_S)
        tPTcPT = thr_pt_load.partition_S(cPT)  # (CPY=1, #CPY=#TILE_S)
        tPTsPT = thr_pt_load.partition_D(sPT)  # (CPY=1, #CPY=1, pt_stages)

        cute.arch.griddepcontrol_wait()

        # Prefetch page indices for first tile
        if lane_load_page:
            logical_page_idx = tPTcPT[0, kv_split_idx]
            if logical_page_idx < page_count:
                cute.copy(
                    cpasync_atom, tPTgPT[None, kv_split_idx], tPTsPT[None, 0, 0]
                )
            else:
                tPTsPT[0] = -1  # load OOB zeros
        cute.arch.sync_warp()
        cute.arch.cp_async_commit_group()

        # Load Q
        cute.copy(tma_atom_q, tBgQ_tma, tBsQ_tma, tma_bar_ptr=q_load_mbar)
        # Sequence loop
        pt_index = 0
        for s in cutlass.range(iters_s):
            # sPT PUBLISH ORDER. This rendezvous MUST precede the
            # cp.async below that overwrites sPT[pt_index_next]: TMA-VO
            # reads its sPT stage with an ordinary LDS and only arrives
            # here AFTERWARDS, so a wait placed after the write leaves the
            # ordering edge present but not guarding the write.
            #
            # pt_stages == 4 and TMA-VO trails by prefetch_iters, so QK
            # writing (s+1) % 4 while VO reads (s-p) % 4 collides only at
            # p == 3 -- one tile PAST the nominal lag, which
            # `assert pt_stages > prefetch_iters + 1` caps at 2. The ring
            # therefore carries exactly ONE tile of margin, and channel-K's
            # Q'-plane builder occupying the correction WG backpressures
            # MMA-VP -> V dequant -> TMA-VO until VO slips to lag 3. When
            # the overwrite won, VO issued its V TMA from the WRONG PHYSICAL
            # PAGES: one (batch, kv_head) item's O was computed from wrong
            # V-hat while K/S/P/m/l stayed exactly right, at ~1 launch in
            # 20..500 depending on builder duration.
            #
            # Hoisted, QK cannot publish (s+1) % 4 until VO has finished
            # iteration s-1, whose earliest in-flight read is s % 4.
            # Deadlock-free: the TMA-VO preamble primes this barrier
            # prefetch_iters times, so this wait needs VO iteration s-1,
            # which needs only QK's tma_order_k arrival from iteration s-1.
            tma_order_v_nbar.arrive_and_wait()
            # Prefetch page indices for next tile
            pt_index_next = 0 if pt_index == (pt_stages - 1) else pt_index + 1
            if s < iters_s - 1 and lane_load_page:
                tile_s_next = (s + 1) * kv_splits + kv_split_idx
                logical_page_idx = tPTcPT[0, tile_s_next]
                virt_page_idx_gmem = tPTgPT[None, tile_s_next]
                virt_page_idx_smem = tPTsPT[None, 0, pt_index_next]
                if logical_page_idx < page_count:
                    cute.copy(cpasync_atom, virt_page_idx_gmem, virt_page_idx_smem)
                else:
                    virt_page_idx_smem[0] = -1  # load OOB zeros
            cute.arch.sync_warp()
            cute.arch.cp_async_commit_group()

            # Load page indices. Lane i's cp.async of entry i is complete, and visible,
            # only to lane i once its own wait_group returns; every lane reads all
            # pages_s entries, so the warp barrier orders the other lanes' completed
            # copies before this lane's read (PTX: cp.async.wait_group, bar.warp.sync).
            cute.arch.cp_async_wait_group(1)
            cute.arch.sync_warp()
            rPT = sPT[None, pt_index].load().reshape((pages_m, tiles_sm))
            pt_index = pt_index_next

            # Load K
            codek_token = codek_producer.try_acquire()
            # Per-page code stages: acquire per (sm, dk, pm), fill with
            # one full-width box on the stage's mbarrier (the tx_count
            # covers the whole stage).
            for sm in cutlass.range_constexpr(tiles_sm):
                for dk in cutlass.range_constexpr(tiles_dk):
                    for pm in cutlass.range_constexpr(pages_m):
                        code_handle = codek_producer.acquire_and_advance(
                            codek_token
                        )
                        is_last_iter = (
                            sm == tiles_sm - 1
                            and dk == tiles_dk - 1
                            and pm == pages_m - 1
                        )
                        if is_last_iter:
                            tma_order_k_nbar.arrive()
                        else:
                            codek_token = codek_producer.try_acquire()

                        virtual_page_idx = rPT[pm, sm]
                        cute.copy(
                            tma_atom_kc,
                            gK_tma[None, dk, virtual_page_idx],
                            sK_tma[None, 0, code_handle.index],
                            tma_bar_ptr=code_handle.barrier,
                        )
                        # Scale stage for this page (once per page,
                        # not per dim-half: dk == 0 only). Page order
                        # (sm, pm) == the softmax WG's consumption
                        # order j = sm*pages_m + tok//64 — and the
                        # dequant WG's page order in the fold-off
                        # arm, which is the same walk. (No ring
                        # exists when BOTH tensors are channel-wise
                        # — their sections go by gmem LDG.)
                        if cutlass.const_expr(
                            dk == 0 and scale_ring_live
                        ):
                            scale_handle = (
                                scale_producer.acquire_and_advance()
                            )
                            cute.copy(
                                tma_atom_sc,
                                gSC_tma[None, virtual_page_idx],
                                sSC_tma[None, 0, scale_handle.index],
                                tma_bar_ptr=scale_handle.barrier,
                            )

        # Tail V loop
        for s in cutlass.range_constexpr(prefetch_iters):
            tma_order_v_nbar.arrive_and_wait()
            tma_order_k_nbar.arrive()
            if cutlass.const_expr(enable_blasst):
                if s < min(prefetch_iters, iters_s):
                    sp_handle = sp_consumer.wait_and_advance()
                    sp_handle.release()

    @cute.jit
    def _dequant_warpgroup(
        self,
        acc_dtype,
        code_stage_bytes,
        codek_consumer,
        codev_consumer,
        dequant_regs,
        dequant_warpgroup_ids,
        iters_s,
        k_producer,
        k_stages,
        kv_stage_bytes,
        kv_stages,
        mma_tile_k,
        mma_tile_m,
        nofold,
        pages_k,
        pages_m,
        pages_s,
        prefetch_iters,
        rebalance_regs,
        sCode,
        sScale,
        scale_consumer,
        tAsK,
        tAsV,
        tidx,
        tiles_dk,
        tiles_dm,
        tiles_sk,
        tiles_sm,
        v_producer,
        v_stages,
    ):
        """The dequant warpgroup: packed codes in the code rings -> e4m3/e2m1 operands.

        This is the warpgroup the code rings feed and the operand rings are fed by,
        so it owns the consumer walks on one side and the producer walks on the
        other, plus the byte maps that do the actual deposit. Its four warps run as
        two concurrent 64-thread streams: the first two warps deposit every K block
        of a tile while the last two deposit every V block, each stream consuming
        its own code ring and producing its own operand ring.

        Two modes live here. The fold path maps packed codes to the operand
        feed with the width- and axis-keyed deposits. `fold_scale_zero=False`
        is the classic-dequant ablation, which computes `e4m3(s*q + z)`
        element-wise instead.

        The byte maps stay nested here and take their output tensor and word count
        as explicit arguments, because each stream arm calls the same maps with its
        own staging tensors (`map_int4(rSrcS4[None, hh, j % 2], rOutS4U[None, hh],
        words_pt)`).
        """
        # Free registers
        # Read straight off `self` instead of taking 21 parameters that were
        # only ever `X = self.X` in the caller. Same expression, same object,
        # and the body below is untouched -- these are the names it uses.
        bits = self.bits
        code_stages = self.code_stages
        f4_feed = self.dequant_int2_to_f4
        page_size = self.page_size
        scale_stages = self.scale_stages
        if cutlass.const_expr(rebalance_regs):
            cute.arch.setmaxregister_decrease(dequant_regs)
            # Under the rebalance the thread index passes through `_opaque_i32`, so the
            # per-thread code and deposit addresses derived from it are computed here, inside
            # this role. Otherwise LLVM hoists them out of the persistent item loop to kernel
            # entry, where they stay live through every other role's code, and the dequant
            # warpgroup then spilled under its smaller budget. The mask, which changes no
            # value, tells the compiler the index is in [0, 128), so the divisions by powers
            # of two below stay shifts.
            dq_tidx = _opaque_i32(
                tidx - dequant_warpgroup_ids[0] * warpgroup_threads) & (warpgroup_threads - 1)
        else:
            dq_tidx = tidx - dequant_warpgroup_ids[0] * warpgroup_threads

        # Packed-code dequant. Per-code-stage (= per-page) thread partition
        # over the two 64-thread streams (every stream thread covers two
        # slots hh in {0, 1} of the 128-slot partition):
        #   K: the stage is one row-major (64, row_bytes) box, read as
        #      linear contiguous CHUNKS of `kf_chunk_words` u32 (capped at
        #      16 B, the load width); chunk c is token row c // kf_cpr,
        #      sub-run c % kf_cpr, with kf_cpr = row_words / chunk_words
        #      chunks per 64-row page. The deposit writes operand row
        #      pm*64+row (K_SW128: dims contiguous within a 128B token
        #      row); at INT4 its address is `deposit_lane_offsets`, at
        #      INT2/E4M3 the swizzled view's. The intra-phase LDS stride
        #      equals the chunk size, so the pattern is bank-uniform at
        #      both widths.
        #   V: on the main feeds V is stored exactly like K, so its stage is
        #      K's box and the V stream runs the K stream's map at both
        #      head tiles: thread t' owns the 16-byte code chunks t' + 64 q
        #      of the page, token row (t' + 64 q) // cpr. The MN-major V
        #      operand image is the K operand image with rows and columns
        #      swapped (MN_SW128 at INT4, the same bytes and swizzle as
        #      K_SW128, deposited through `deposit_lane_offsets`; the packed
        #      E2M1 panels on the f4 feed), so a chunk's expansion lands as
        #      contiguous channels of one token, and every quarter warp's
        #      loads and deposits are conflict-free at every operand-ring
        #      start row, as K's are. Only
        #      the INT2 E4M3 feed and the fold-off arm read a token column
        #      across dim-halves of a channel-row-shaped pool and deposit
        #      into MN_SW128.
        # The byte maps preserve code order, so NO in-register
        # transpose exists on either path: the V source byte order
        # that makes this true IS the mode-keyed pool order (the
        # channel-row order on the E4M3 feed, K's order on the main
        # feeds).
        words = mma_tile_k * bits // 64  # src u32/(tok, dim-half)
        # src u32 per thread slot per stage; equal to `words` by construction
        # because one dequant warpgroup is hardcoded, kept as the per-slot name
        # the staging shapes and `load_src` read.
        words_pt = words
        out_bytes = mma_tile_k // 2  # e4m3 out bytes per thread slot
        stage_words = code_stage_bytes // 4
        kf_chunk_words = min(4, words_pt)
        kf_runs = words_pt // kf_chunk_words
        kf_cpr = 2 * words // kf_chunk_words  # chunks per token row
        sCode_w = cute.recast_ptr(sCode.iterator, dtype=Int32)
        # The K stream's flat reader is built inside its own arm below
        # (the f4 and e4m3 streams partition the stage differently).
        if cutlass.const_expr(not f4_feed and bits == 2):
            # V stage words: (col, word, slice, stage); flat token-major
            # page stream (2*words words per token column, slice = the
            # words_pt-word sub-run). The V code ring's stages sit past
            # the K ring's in the flat sCode block, so the view is
            # rebased by code_stages_k stages (cv.index is V-ring-local),
            # the same rebasing the token-row V views use.
            codeV_w = cute.make_tensor(
                sCode_w + self.code_stages_k * stage_words,
                cute.make_layout(
                    (page_size, words_pt, 2, code_stages),
                    stride=(2 * words, 1, words_pt, stage_words),
                ),
            )

        # Swizzled operand-stage (m, k, stage) views (each iterator
        # carries its tensor's swizzle, SW128 for K and for the INT2 V
        # stages; same construction as the host's smem_layout_{k,v}_mk).
        # The INT2/E4M3 deposits store through them; the INT4 deposits
        # address the ring directly (`deposit_lane_offsets`).
        mk_tiler = cute.make_layout((mma_tile_m, mma_tile_k, kv_stages))
        sK_op = cute.composition(tAsK, mk_tiler)  # (tok, dim, stage)
        sV_op = cute.composition(tAsV, mk_tiler)  # (dim, tok, stage)

        # rmem staging is allocated outside the dynamic loops (DSL contract),
        # per stream arm below: DOUBLE-BUFFERED source words, because each
        # stream software-pipelines the code-stage LDS one stage ahead (the
        # unpipelined loop exposes the ~30cyc LDS latency per stage), plus
        # the thread's output bytes. The buffer index sits in the RIGHT mode
        # so each buffer slice is a contiguous words_pt run; the left-major
        # default would make the slice stride-2 and de-vectorize the 16B
        # prefetch LDS.
        # On the f4 path the deposit writes ONLY the 8 payload bytes of each
        # 16B b4x16_p64 chunk; the pad is left stale, and the hardware ignores
        # it (a silicon probe poisons it with 0xEE and passes). out_bytes
        # == 16*words_pt here, so the register image halves to 8*words_pt.
        #
        # The saving is INSTRUCTION COUNT, not bandwidth: the two 0xEEEEEEEE
        # pad materializations per source word were real per-chunk work the
        # compiler did not hoist. Registers are unchanged, so this is not a
        # pressure lever either.
        #
        # It is emphatically NOT an SMEM-bandwidth lever, and no narrower store
        # ever will be: in b4x16_p64 the payload of chunk p lives at bytes
        # [16p, 16p+8), i.e. banks (4p + j/4) % 32, so the payload occupies
        # exactly 16 of the 32 SMEM banks FOREVER. An 8B/lane store at 16B
        # stride fits only 8 lanes per bank cycle (lane 8 collides with lane 0),
        # the same 4 wavefronts a 16B store takes; both forms move 8 chunks per
        # cycle. Confirmed on both sides: tc-pipe wavefronts are bit-for-bit
        # identical across the change, and every deposit STS was already at
        # ideal wavefronts. Do not re-propose store width as an SMEM-bandwidth
        # fix.
        dep_bytes = 8 if f4_feed else 16
        rout_bytes = out_bytes // 2 if f4_feed else out_bytes
        # The e4m3 K stream's out bytes leave as kf_runs contiguous operand
        # slices per slot (the byte maps write out words in source-word
        # order, so run r's expansion is bytes [r*kf_out_bytes,
        # +kf_out_bytes)). kf_out_bytes counts the FULL e4m3 expansion.
        kf_out_bytes = out_bytes // kf_runs

        # FOLD-OFF per-token FMA coefficients: (s, z) each broadcast to a
        # bf16x2 word, so [., 0] = s_b and [., 1] = z_b. Each stream builds
        # ONLY its own tensor's table, at the top of its own tile, from the
        # scale stages it waits itself (both streams walk the whole ring;
        # the consumer group in `kernel` is the full dequant warp count).
        # The K thread maps 2*kf_runs flat chunks per page on different
        # token rows, so its table carries one (s, z) pair per (page, run);
        # the V thread maps one token column. The V stream reads V's
        # coefficients when it maps V, so no coefficient is carried across
        # tiles.
        rTabK = rTabV = None
        if cutlass.const_expr(nofold):
            rTabK = cute.make_rmem_tensor((pages_s, 2 * kf_runs, 2), Int32)
            rTabV = cute.make_rmem_tensor((pages_s, 2), Int32)

        # The f4-feed K deposit views. The f4 K stage is the
        # canonical b4x16_p64 panel image (MMA descriptor comment at
        # the KQ dispatch): 16B chunk of (token row, dim-panel p) at
        # byte p*2048 + row*16; one source u32 (16 codes = 16 dims =
        # one panel's chunk) becomes 8 data bytes + 8 pad bytes —
        # the SAME words_pt*16 output bytes per thread as the e4m3
        # expansion, so the output staging is viewed as (16B-chunk,
        # word) tiles. The chunk address view is a swizzle-free byte
        # recast of the ring (recast_ptr with dtype only strips the
        # SW128 of the alloc view).
        if cutlass.const_expr(f4_feed):
            # Split operand rings: K deposits target the K ring
            # (first k_stages physical stages) via sKf4b; V deposits
            # target the V ring (the next v_stages stages) via
            # sVf4b. Handle indices are ring-local on both sides.
            sKf4b = cute.make_tensor(
                cute.recast_ptr(tAsK.iterator, dtype=Uint8),
                cute.make_layout(
                    (16, mma_tile_m, mma_tile_k // 16, k_stages),
                    stride=(1, 16, 16 * mma_tile_m, kv_stage_bytes),
                ),
            )  # (byte-in-chunk, token row, dim-panel, stage)
            # The f4-feed V deposit. The V pool is stored exactly like K
            # (token rows, K's sigma order), so the V stream runs the K
            # stream's partition and two-mask expand, and deposits the SAME
            # panel image with the roles of rows and K swapped: 16B chunk of
            # (token k, channel panel i) at byte i*16*mma_tile_k + k*16,
            # channel nibbles LSB-first in bytes 0-7. The MMA-VP warp reads
            # that image as an MN-major E2M1 operand (no-swizzle canonical
            # layout, LBO = 8 tokens, SBO = one channel panel).
            sVf4b = cute.make_tensor(
                cute.recast_ptr(tAsK.iterator, dtype=Uint8)
                + k_stages * kv_stage_bytes,
                cute.make_layout(
                    (16, mma_tile_k, mma_tile_m // 16, v_stages),
                    stride=(1, 16, 16 * mma_tile_k, kv_stage_bytes),
                ),
            )  # (byte-in-chunk, token row, channel panel, stage)

            # Split dequant warp streams. The first half of the
            # dequant warps (stream_threads = 64 threads) deposits
            # ALL K blocks, the second half ALL V blocks,
            # CONCURRENTLY — each thread covers hh in {0,1}: the K
            # thread two flat chunks of the K stage, the V thread the same
            # two flat chunks of the V stage, token row c // 2 at channel
            # half c % 2 (effective thread id kt + hh*stream_threads of
            # the 128-slot partition).
            # Shared per-thread staging (both streams use the same
            # shapes): (word, hh, buf) double-buffered source and a
            # 2x-wide chunk output tile.
            stream_threads = warpgroup_threads // 2
            assert stream_threads >= page_size, (
                "K stream needs one thread per token row"
            )
            rSrcS = cute.make_rmem_tensor((words_pt, 2, 2), Int32)
            rOutS = cute.make_rmem_tensor(2 * rout_bytes, Uint8)
            rOutSU2 = cute.make_tensor(
                cute.recast_ptr(rOutS.iterator, dtype=Int32),
                cute.make_layout((dep_bytes // 4 * words_pt, 2)),
            )  # (out word, hh)
            rOutSC = cute.make_tensor(
                rOutS.iterator,
                cute.make_layout((dep_bytes, words_pt, 2)),
            )  # (payload byte, source word, hh)

        # e4m3 concurrent K/V dequant streams (INT4, and int2 with
        # dequant_int2_to_f4=0): stream thread t covers slots hh in
        # {0, 1} of the 128-slot partition, the K stream through its
        # 2*kf_runs flat chunks. INT4 V, stored like K, owns the same flat
        # chunks of its own stage; INT2/E4M3 V owns token column t at both
        # dim-halves, with its byte map and operand-stage deposits. Staging mirrors rSrcS/rOutS: a
        # double-buffered (word, hh, buf) source and a 2x-wide output.
        # The output views are sized in OUTPUT words (out_bytes // 4 per
        # slot), because a source word expands to two operand words at
        # INT4 and four at INT2.
        if cutlass.const_expr(not f4_feed):
            i4s_threads = warpgroup_threads // 2
            assert i4s_threads >= page_size, (
                "e4m3 K stream needs one thread per token row"
            )
            rSrcS4 = cute.make_rmem_tensor((words_pt, 2, 2), Int32)
            rOutS4 = cute.make_rmem_tensor(2 * out_bytes, Uint8)
            rOutS4U = cute.make_tensor(
                cute.recast_ptr(rOutS4.iterator, dtype=Int32),
                cute.make_layout((out_bytes // 4, 2)),
            )  # (out word, hh)
            rOutS4b = cute.make_tensor(
                rOutS4.iterator, cute.make_layout((out_bytes, 2))
            )  # (out byte, hh)
            # Flat-stage views: a stream thread's 2*words_pt source u32
            # arrive as 2*kf_runs contiguous chunks (run q = linear chunk
            # t + q*i4s_threads), and its 2*out_bytes leave as 2*kf_runs
            # contiguous kf_out_bytes operand slices, in run order — the
            # same memory as rSrcS4/rOutS4, reshaped run-major.
            rSrcS4R = cute.make_tensor(
                rSrcS4.iterator,
                cute.make_layout((kf_chunk_words, 2 * kf_runs, 2)),
            )  # (word, run, buf)
            rSrcS4F = cute.make_tensor(
                rSrcS4.iterator,
                cute.make_layout((2 * words_pt, 2)),
            )  # (word, buf) — one contiguous run for the byte map
            rOutS4Uf = cute.make_tensor(
                cute.recast_ptr(rOutS4.iterator, dtype=Int32),
                cute.make_layout(out_bytes // 2),
            )  # flat out words, source-word order
            rOutS4R = cute.make_tensor(
                rOutS4.iterator,
                cute.make_layout((kf_out_bytes, 2 * kf_runs)),
            )  # (out byte, run)
            if cutlass.const_expr(bits == 2):
                # int2 maps PER RUN (the fold-off K slots carry
                # different coefficients): run q's operand words are
                # the q-th kf_out_bytes//4 words of the same memory
                # rOutS4R slices by byte
                rOutS4UR = cute.make_tensor(
                    cute.recast_ptr(rOutS4.iterator, dtype=Int32),
                    cute.make_layout((kf_out_bytes // 4, 2 * kf_runs)),
                )  # (out word, run)

        # Byte maps (helpers only read rS / write rOutU — pipeline
        # handshakes stay inline in this frame, see module docstring)
        def load_src(tDsC, code_idx, rS, words_pt=words_pt):
            cute.autovec_copy(
                cute.make_tensor(
                    tDsC[None, code_idx].iterator.align(
                        min(16, words_pt * 4)
                    ),
                    cute.make_layout(words_pt),
                ),
                rS,
            )

        def release_code_stage(h):
            # Every thread of the stream arrives on the stage's empty barrier itself,
            # after the operand fence it executed behind its own LDS of the stage, so
            # each reader's fence precedes its own arrival (CROSS-PROXY ORDERING). The
            # stock release arrives from lane 0 only, which would need a warp barrier.
            pipe = h.get_origin()
            pipe.sync_object_empty.arrive(h.index, pipe.consumer_mask)

        def map_int4(rS, rOutU, words):
            # raw-code two-mask deposit, K and V: the operand byte
            # IS the 4-bit code. e4m3 pattern q decodes to
            # q * 2**-9 exactly (the subnormal band and the first
            # normal binade share one ulp), so the fold rescales
            # s' = 512 * s and nothing else changes — see
            # ALPHA_INV_INT4. A-role subnormal handling is
            # silicon-verified.
            # ONE LAYOUT, NO BRANCH: the pool always carries the
            # 8-slot store shuffle on this width (`k_pool_is_permuted`),
            # which pre-interleaves the
            # even/odd nibbles so each source word yields its two
            # operand words with two masks and one shift — the
            # same 3-op form the int2 f4 feed uses, 3 ops per 8
            # codes = 0.375 per element.
            # The shuffled pool read with these masks emits operand words
            # byte-identical to the unshuffled pool read with the PRMT
            # form; the V twin is identical.
            for wl in cutlass.range_constexpr(words):
                rOutU[2 * wl] = rS[wl] & Int32(0x0F0F0F0F)
                rOutU[2 * wl + 1] = (rS[wl] >> 4) & Int32(0x0F0F0F0F)

        def map_int2_k(rS, rOutU, words):
            # PRMT gather with table e4m3(8+q) (0x50..0x53): pack_lut
            # puts codes 4a..4a+3 one per selector nibble at shift
            # 16*(a//2)+2*(a%2). (Straight OR-magic is impossible on
            # K: the G1-locked mKq bit layout is nibble-spaced.)
            for wl in cutlass.range_constexpr(words):
                for aa in cutlass.range_constexpr(4):
                    sh = 16 * (aa // 2) + 2 * (aa % 2)
                    rOutU[4 * wl + aa] = _prmt_r(
                        Int32(LUT2C8),
                        Int32(LUT2C8),
                        (rS[wl] >> sh) & Int32(0x33333333),
                    )

        def map_int2_f4(rS, rOutU, words):
            # The 2->4-bit nibble deposit: the e2m1 nibble IS the
            # 2-bit code (q/2 exact: nibbles 0/1/2/3 decode to
            # 0/.5/1/1.5). The packer puts code i of a u32 at bit
            # 16*(i//8) + 4*(i%4) + 2*((i%8)//4) (VERIFIED): per 16-bit
            # half, codes 0-3 sit at nibble
            # bits [1:0] and codes 4-7 at nibble bits [3:2].
            # Gathering LSB-first nibble words (out0 nibble j = code
            # j, out1 nibble j = code 8+j) is therefore 2 LOP3+SHF
            # pairs per output word. Data bytes 0-7 of each 16B
            # chunk; bytes 8-15 are the pad and are NOT written, the
            # hardware ignores them, so they keep whatever the ring
            # stage last held.
            # ONE LAYOUT, NO BRANCH. The pool always carries sigma
            # on this feed, so the packer's interleave cancels and
            # each source word yields two masks -- the same 3-op form
            # the V stream uses for V. There is deliberately no
            # second, unpermuted layout: this function is reached
            # only under `const_expr(f4_feed)`, where
            # `k_pool_is_permuted` is always true, and an unreachable
            # second layout is a second contract a pool writer can
            # satisfy by mistake.
            # The two forms emit byte-identical operand words.
            for wl in cutlass.range_constexpr(words):
                w = rS[wl]
                rOutU[2 * wl + 0] = w & Int32(0x33333333)
                rOutU[2 * wl + 1] = (w >> 2) & Int32(0x33333333)

        def map_int2_v(rS, rOutU, words):
            # OR-magic: (w >> 2m) & 0x03030303 | 0x50505050 =
            # e4m3(8+q) per byte lane — the channel-row order linearizes
            # the bit layout so byte lane l of shift 2m is local dim
            # 4m+l.
            for wl in cutlass.range_constexpr(words):
                for aa in cutlass.range_constexpr(4):
                    rOutU[4 * wl + aa] = (
                        (rS[wl] >> (2 * aa)) & Int32(0x03030303)
                    ) | Int32(0x50505050)

        # ---- FOLD-OFF (fold_scale_zero=False) element-wise
        # dequant. One packed bf16 FMA per TWO elements, then a
        # convert to the e4m3 operand: the ablation baseline pays
        # what element-wise affine dequantization costs (see the
        # `_fma2_bf16` header for why this is deliberately not the
        # cheaper per-token table). ~15 ops per 4
        # elements against the fold feed's 1-2, which IS the
        # quantity the ablation exists to expose.
        #
        # Both maps share one primitive: place two int2 codes into
        # the mantissa fields of a bf16x2 word so each lane reads
        # y = 4 + q EXACTLY (MAGIC2L), take the magic base back off
        # (exact), FMA against the token's broadcast (s, z), then
        # pack four lanes to e4m3. Each lane's code is masked
        # SEPARATELY — the source word is fully packed with 16
        # codes, so a shared mask after the OR would admit a
        # neighbouring code's bits. See the MAGIC2L note for why the
        # base is subtracted instead of prefolded into z.
        def bf16x2_q(w, sh_lo, sh_hi):
            # SIGN FILL is the trap here, and the HI lane sits
            # EXACTLY on its bound. `w` is Int32, so `>>` is an
            # ARITHMETIC shift: after `w >> n`, bit p holds w[p+n]
            # only while p <= 31-n, and above that it holds a copy
            # of the sign bit. The kept bits are 5-6 (low lane) and
            # 21-22 (high lane), so a right shift is safe only while
            # n <= 25 and n <= 9 respectively. The shifts in use go
            # to 21 on the low lane (fine) and to exactly 9 on the
            # high lane, which reads w[30] and w[31] — real code
            # bits for local dim 15, not fill, but ONE more and the
            # mask would start admitting fabricated ones. That is a
            # silent wrong-number failure (a code of 3 where the
            # pool holds 0), so it is asserted rather than argued.
            #
            # SITTING ON THE BOUND IS SYSTEMATIC, not luck, and the
            # FOLD maps do it too — the last code of a fully packed
            # u32 has its MSB at bit 31, so the shift that brings
            # that code to the top kept bit is by construction the
            # largest safe one. Every int2 map lands on 31 exactly:
            # this one at 22+9, `map_int2_k` at 13+18 (mask
            # 0x33333333, shift <= 18), `map_int2_v` at 25+6 (mask
            # 0x03030303, shift <= 6). There is no slack ANYWHERE,
            # which is why the check belongs in code. It is asserted
            # only here because this is the arm being changed;
            # routing the shipped fold maps through a checked helper
            # would be a robustness-only edit to a byte-gated feed
            # path.
            assert sh_lo >= 0 or -sh_lo <= 25, (
                f"low-lane right shift {-sh_lo} > 25 would put "
                f"sign fill in mantissa bits 5-6")
            assert sh_hi >= 0 or -sh_hi <= 9, (
                f"high-lane right shift {-sh_hi} > 9 would put "
                f"sign fill in mantissa bits 21-22")
            lo = (w << sh_lo) if sh_lo >= 0 else (w >> -sh_lo)
            hi = (w << sh_hi) if sh_hi >= 0 else (w >> -sh_hi)
            y = ((lo & Int32(MASK2L_LO))
                 | (hi & Int32(MASK2L_HI))
                 | Int32(MAGIC2L))
            return _sub2_bf16(y, Int32(MAGIC2L))

        def map_int2_k_dq(rS, s_b, z_b, rOutU, words):
            # K source (the logical pack order): for output word aa, the code
            # of local dim 4aa+j sits at bit sh + 4j, with
            # sh = 16*(aa//2) + 2*(aa%2) -- the SAME geometry the
            # fold path's PRMT selector uses, read per-element here
            # instead of as a nibble gather. A lane wants its code at
            # bit 5 (low) or 21 (high), so the pair (j, j+1) needs
            # shifts (5 - bit_j) and (21 - bit_{j+1}).
            for wl in cutlass.range_constexpr(words):
                for aa in cutlass.range_constexpr(4):
                    sh = 16 * (aa // 2) + 2 * (aa % 2)
                    q01 = bf16x2_q(rS[wl], 5 - sh, 17 - sh)
                    q23 = bf16x2_q(rS[wl], 5 - (sh + 8),
                                   17 - (sh + 8))
                    rOutU[4 * wl + aa] = _e4m3x4_of_bf16x2(
                        _fma2_bf16(q01, s_b, z_b),
                        _fma2_bf16(q23, s_b, z_b),
                    )

        def map_int2_v_dq(rS, s_b, z_b, rOutU, words):
            # V source (the channel-row order): local dim 4aa+l sits at
            # bit 8l + 2aa -- the byte-spaced layout the fold path's
            # OR-magic wants. Pair (l=0,1) is at bits 2aa and 8+2aa,
            # pair (l=2,3) at 16+2aa and 24+2aa; same "code to bit 5
            # / 21" rule gives the four shifts.
            for wl in cutlass.range_constexpr(words):
                for aa in cutlass.range_constexpr(4):
                    q01 = bf16x2_q(rS[wl], 5 - 2 * aa, 13 - 2 * aa)
                    q23 = bf16x2_q(rS[wl], -11 - 2 * aa,
                                   -3 - 2 * aa)
                    rOutU[4 * wl + aa] = _e4m3x4_of_bf16x2(
                        _fma2_bf16(q01, s_b, z_b),
                        _fma2_bf16(q23, s_b, z_b),
                    )

        def dq_coeffs(s_q, z_q):
            # FOLD-OFF per-token FMA operands, built once per
            # (thread, page) and broadcast to both bf16 lanes. The
            # feed carries q itself (the magic base is subtracted in
            # bf16x2_q), so these are the pool's own (s, z) at their
            # natural magnitudes and the FMA evaluates q*s + z, the
            # reference's own expression. An OOB
            # page's TMA zero-fill gives s = z = 0, hence an
            # all-zero operand row: the same thing the baseline's
            # zero-filled K/V stage carried, which is what tma_mask
            # semantics assume.
            return (_cvt2_bf16x2(s_q, s_q),
                    _cvt2_bf16x2(z_q, z_q))

        # One operand stage = pages_m (resp. pages_k) code stages. Each
        # K/V block is software-pipelined: the (j+1)-th code stage's LDS
        # issues BEFORE the j-th stage's map+STS (double-buffered source
        # staging), hiding the LDS latency that dominates the unpipelined
        # loop. A code stage is released only behind the first operand-commit
        # fence after its LDS (CROSS-PROXY ORDERING): nothing orders the map
        # before a bare arrive, so a release there would let the TMA refill
        # overwrite the stage while its LDS is still in flight.
        nkc = tiles_sm * tiles_dk * pages_m
        nvc = tiles_sk * tiles_dm * pages_k
        if cutlass.const_expr(nofold):
            # Each stream holds a whole tile of scale stages at once,
            # exactly as the fold arm's softmax prefetch does (same assert
            # there, under fold_k) ...
            assert scale_stages >= pages_s, (
                "fold-off holds one tile's scale stages while it "
                "builds the tile's dequant tables"
            )
            # ... and a scale stage is refilled by the TMA-QK warp only
            # after BOTH streams have released it, which couples the ring
            # to the two TMA warps' ordering barrier. The chain is this.
            # TMA-VO's iteration t begins with
            # tma_order_k_nbar.arrive_and_wait(), and because TMA-VO's
            # prologue consumes prefetch_iters generations of that
            # barrier, the arrival it waits for is TMA-QK's arrive for
            # tile t + prefetch_iters. TMA-QK makes that arrive inside its
            # per-page loop, at the tile's last page, so it sits behind
            # the scale-ring acquires of the tile's earlier pages, and
            # each acquire blocks until the stage's previous occupant has
            # been released by all four dequant warps. The V stream
            # releases a tile's stages at its tile top, which it reaches
            # only after finishing the previous tile's V codes, and those
            # codes are what TMA-VO issues. So with a ring of
            # r = scale_stages / pages_s whole tiles, TMA-QK at tile s
            # waits for the V stream's tile top of tile s - r; that needs
            # tile s - r - 1's V codes; TMA-VO issues them only after TMA-QK's
            # arrive for tile s - r - 1 + prefetch_iters, which is tile s
            # or later whenever r < prefetch_iters, and TMA-QK cannot make
            # it while blocked. At scale_stages = pages_s the cycle closes
            # at tile 2 of every CTA with three or more tiles. The bound
            # below is the conservative supported minimum, not the tight
            # bound: at the usual shape (pages_s = 4, prefetch_iters = 2)
            # seven stages already break the cycle, because TMA-QK's
            # tma_order_k_nbar arrival precedes its acquire of the last
            # page's scale stage, and scale depth has no tile-multiple
            # restriction; no depth below the bound is validated, and the
            # anchor cell at eight sits on it. The commit that removed
            # K3's assert argued the V stream depended on nothing in the K
            # stream's current tile, which is true of the stream and false
            # of the TMA warp that feeds it.
            assert scale_stages >= prefetch_iters * pages_s, (
                "fold-off on the concurrent streams needs scale_stages >= "
                f"prefetch_iters * pages_s = {prefetch_iters} * {pages_s}; "
                f"got {scale_stages}. TMA-VO's iteration t waits on TMA-QK's "
                "tma_order_k_nbar arrival for tile t + prefetch_iters; that "
                "arrival sits behind TMA-QK's scale-ring acquires; a scale "
                "stage frees only after BOTH streams release it; the V stream "
                "releases a tile's stages at its tile top, after finishing "
                "the previous tile's V codes, which TMA-VO issues. At the "
                "shipped prefetch_iters = 2 and scale_stages = pages_s the "
                "cycle closes at tile 2; the bound "
                "is the conservative supported minimum (at the shipped shape "
                "seven stages already break the cycle), and no shallower "
                "depth is validated"
            )
            # Above that bound and below (prefetch_iters + 2) * pages_s
            # the K stream is throttled to the V stream's progress, which
            # costs time and nothing else. The fold ablation takes
            # scale_stages=16 from BITS_TUNING, which is
            # (prefetch_iters + 2) * pages_s at the shipped shape, so a
            # measured arm never runs throttled; code_stages=8 with
            # scale_stages=8 sits on the asserted minimum, live and throttled.
        # INT2: merge the per-tile deposit drains, on both feeds, except
        # the FP4 K stream (see merge_k_drain below).
        # Each fence_view_async_shared is a full async-proxy drain
        # (~490 long_sb samples on the 4 per-tile drains in the
        # headroom audit); one fence AFTER the whole tile burst
        # covers every prior STS of that burst, then the per-stage
        # commits arrive back-to-back (fence-then-arrive order
        # preserved; arrive counts per stage UNCHANGED — still one
        # commit per producer thread per stage, just deferred).
        # Deferring commits means the producer holds ALL the tile's
        # ring stages before publishing any, so the merged form is
        # only legal when blocks/tile <= ring stages (acquire of
        # block b+1 must never wait on our own uncommitted block).
        k_deposit_blocks = tiles_sm * tiles_dk
        v_deposit_blocks = tiles_sk * tiles_dm
        # Merged drains are the INT2 design on the e4m3 feed, except the f4 K
        # stream: its code stages are released behind the drain fences, so a K
        # code release at the tile's end would hold back the TMA-QK warp, which
        # TMA-VO trails through tma_order_k_nbar, by a whole tile. INT4 keeps
        # per-block fence-and-commit.
        merge_k_drain = (bits == 2) and k_deposit_blocks <= k_stages and not f4_feed
        merge_v_drain = (bits == 2) and v_deposit_blocks <= v_stages
        if cutlass.const_expr(bits == 2 and not f4_feed):
            # The int2 e4m3 streams merge by design, not by fit: an
            # operand ring too shallow to hold a tile's blocks is a
            # configuration error here, not a reason to fall back to
            # per-block drains silently.
            assert merge_k_drain and merge_v_drain, (
                "int2 e4m3 streams merge the per-tile deposit drains, which "
                "needs a tile's deposit blocks to fit each operand ring: "
                f"K {k_deposit_blocks} blocks vs {k_stages} stages, "
                f"V {v_deposit_blocks} blocks vs {v_stages} stages"
            )
        # CONCURRENT K / V dequant streams. A serialized single-stream
        # loop puts both deposit latencies on the pacer chain
        # back-to-back, and every deeper-K-lead reorder of it regressed
        # because delaying V deposits delays mma_order_vp -> KQ issue.
        # Here the first half of the dequant warps deposits ALL K blocks
        # while the second half deposits ALL V blocks in parallel.
        # Each stream consumes its OWN code ring (K:
        # codek_consumer, V: codev_consumer), and it must stay
        # that way. Under one shared FIFO, sibling walkers could
        # desync across a ring revolution and orphan a stage,
        # produced once and never released, which wedges the
        # whole CTA on an mbarrier. With per-stream rings every
        # wait is an own-stage wait paced by the walker's own
        # releases, so no lap is possible by construction.
        if cutlass.const_expr(f4_feed):
            if dq_tidx < stream_threads:
                # ---- K stream (all K blocks, both linear chunks/thread)
                # (thread, word, slot, stage): slot hh of thread t is
                # linear chunk t + hh*stream_threads of the flat stage,
                # which is token row (t + hh*stream_threads) // 2 at
                # dim-half (t + hh*stream_threads) % 2; the deposit
                # indices below carry that (row, dim chunk) meaning.
                codeKfS_w = cute.make_tensor(
                    sCode_w,
                    cute.make_layout(
                        (stream_threads, words_pt, 2, code_stages),
                        stride=(
                            words_pt,
                            1,
                            stream_threads * words_pt,
                            stage_words,
                        ),
                    ),
                )
                tDsCK2 = codeKfS_w[dq_tidx, None, None, None]
                _c1 = dq_tidx + stream_threads
                sf_rows = [dq_tidx // 2, _c1 // 2]
                sf_pan0 = [
                    (dq_tidx % 2) * words_pt,
                    (_c1 % 2) * words_pt,
                ]
                for s in cutlass.range(iters_s):
                    ck = codek_consumer.wait_and_advance()
                    for hh in cutlass.range_constexpr(2):
                        load_src(
                            tDsCK2[None, hh, None],
                            ck.index,
                            rSrcS[None, hh, 0],
                        )
                    k_code_handles = [ck]
                    k_kv_handles = [k_producer.acquire_and_advance()]
                    for j in cutlass.range_constexpr(nkc):
                        if cutlass.const_expr(j + 1 < nkc):
                            ckn = codek_consumer.wait_and_advance()
                            for hh in cutlass.range_constexpr(2):
                                load_src(
                                    tDsCK2[None, hh, None],
                                    ckn.index,
                                    rSrcS[None, hh, (j + 1) % 2],
                                )
                            k_code_handles.append(ckn)
                        kb, pm = j // pages_m, j % pages_m
                        for hh in cutlass.range_constexpr(2):
                            map_int2_f4(
                                rSrcS[None, hh, j % 2],
                                rOutSU2[None, hh],
                                words_pt,
                            )
                        for hh in cutlass.range_constexpr(2):
                            for wl in cutlass.range_constexpr(words_pt):
                                dstKc = cute.make_tensor(
                                    sKf4b[
                                        None,
                                        pm * page_size + sf_rows[hh],
                                        sf_pan0[hh] + wl,
                                        k_kv_handles[kb].index,
                                    ].iterator.align(16),
                                    cute.make_layout(dep_bytes),
                                )
                                cute.autovec_copy(
                                    rOutSC[None, wl, hh], dstKc
                                )
                        if cutlass.const_expr(pm == pages_m - 1):
                            if cutlass.const_expr(not merge_k_drain):
                                # STS -> async-proxy fence -> commit
                                cute.arch.fence_view_async_shared()
                                k_kv_handles[kb].commit()
                                # Release, behind the fence above, every code stage whose LDS
                                # precedes it: this block's, and the next stage, prefetched
                                # one ahead, which its own block then skips (CROSS-PROXY
                                # ORDERING).
                                for _jj in cutlass.range_constexpr(
                                        pages_m - min(1, j // pages_m) + min(1, nkc - 1 - j)):
                                    release_code_stage(k_code_handles[j - pages_m + 1 + min(1, j // pages_m) + _jj])
                            if cutlass.const_expr(j + 1 < nkc):
                                k_kv_handles.append(
                                    k_producer.acquire_and_advance()
                                )
                    if cutlass.const_expr(merge_k_drain):
                        # ONE async-proxy fence covers ALL the
                        # tile's K-panel STS, then the per-stage
                        # commits arrive back-to-back.
                        cute.arch.fence_view_async_shared()
                        for kb2 in cutlass.range_constexpr(
                            k_deposit_blocks
                        ):
                            k_kv_handles[kb2].commit()
                        # The tile's code stages, released behind the merged drain's
                        # fence (CROSS-PROXY ORDERING).
                        for _jj in cutlass.range_constexpr(nkc):
                            release_code_stage(k_code_handles[_jj])
            else:
                # ---- V stream (all V blocks, two slots/thread). The V
                # pool is stored exactly like K, so this is the K stream's
                # partition over K's code stage: slot hh is linear 16B
                # chunk vt + hh*stream_threads, token row c // 2 at
                # channel half c % 2, and its words_pt source words are
                # the chunks of channel panels (c % 2)*words_pt + wl of
                # that token in the MN-major V image (`sVf4b`). Since
                # stream_threads is even, slot 1 is slot 0's token row plus
                # stream_threads // 2 at the same panel, so a thread keeps one
                # runtime (row, panel) and slot 1 adds a trace-time offset:
                # the per-slot form keeps a second set of deposit addresses
                # live, and most persistent INT2 builds reload them from local
                # memory inside this loop.
                vt = dq_tidx - stream_threads
                codeVfS_w = cute.make_tensor(
                    sCode_w + self.code_stages_k * stage_words,
                    cute.make_layout(
                        (stream_threads, words_pt, 2, self.code_stages_v),
                        stride=(
                            words_pt,
                            1,
                            stream_threads * words_pt,
                            stage_words,
                        ),
                    ),
                )
                tDsCV2 = codeVfS_w[vt, None, None, None]
                v_row0 = vt >> 1
                v_pan00 = (vt & 1) * words_pt
                # Split code rings: the V walker owns EVERY stage
                # of its ring — no prologue walk over K codes, no
                # cross-stream skips. Its first wait is V(0) at
                # stage 0 phase 0, paced thereafter by its own
                # releases.
                for s in cutlass.range(iters_s):
                    cv = codev_consumer.wait_and_advance()
                    for hh in cutlass.range_constexpr(2):
                        load_src(tDsCV2[None, hh, None], cv.index,
                                 rSrcS[None, hh, 0])
                    v_code_handles = [cv]
                    v_kv_handles = [v_producer.acquire_and_advance()]
                    for j in cutlass.range_constexpr(nvc):
                        if cutlass.const_expr(j + 1 < nvc):
                            cvn = codev_consumer.wait_and_advance()
                            for hh in cutlass.range_constexpr(2):
                                load_src(tDsCV2[None, hh, None], cvn.index,
                                         rSrcS[None, hh, (j + 1) % 2])
                            v_code_handles.append(cvn)
                        vb, pk = j // pages_k, j % pages_k
                        for hh in cutlass.range_constexpr(2):
                            map_int2_f4(
                                rSrcS[None, hh, j % 2],
                                rOutSU2[None, hh],
                                words_pt,
                            )
                        for hh in cutlass.range_constexpr(2):
                            for wl in cutlass.range_constexpr(words_pt):
                                dstVc = cute.make_tensor(
                                    sVf4b[
                                        None,
                                        pk * page_size + v_row0
                                        + hh * (stream_threads // 2),
                                        v_pan00 + wl,
                                        v_kv_handles[vb].index,
                                    ].iterator.align(16),
                                    cute.make_layout(dep_bytes),
                                )
                                cute.autovec_copy(
                                    rOutSC[None, wl, hh], dstVc
                                )
                        if cutlass.const_expr(pk == pages_k - 1):
                            if cutlass.const_expr(not merge_v_drain):
                                cute.arch.fence_view_async_shared()
                                v_kv_handles[vb].commit()
                                # Release, behind the fence above, every code stage whose LDS
                                # precedes it: this block's, and the next stage, prefetched
                                # one ahead, which its own block then skips (CROSS-PROXY
                                # ORDERING).
                                for _jj in cutlass.range_constexpr(
                                        pages_k - min(1, j // pages_k) + min(1, nvc - 1 - j)):
                                    release_code_stage(v_code_handles[j - pages_k + 1 + min(1, j // pages_k) + _jj])
                            if cutlass.const_expr(j + 1 < nvc):
                                v_kv_handles.append(
                                    v_producer.acquire_and_advance()
                                )
                    if cutlass.const_expr(merge_v_drain):
                        # ONE async-proxy fence covers ALL the
                        # tile's V-panel STS, then the per-stage
                        # commits arrive back-to-back.
                        cute.arch.fence_view_async_shared()
                        for vb2 in cutlass.range_constexpr(
                            v_deposit_blocks
                        ):
                            v_kv_handles[vb2].commit()
                        # The tile's code stages, released behind the merged drain's
                        # fence (CROSS-PROXY ORDERING).
                        for _jj in cutlass.range_constexpr(nvc):
                            release_code_stage(v_code_handles[_jj])
        else:
            # ---- e4m3 CONCURRENT K/V dequant streams over the split rings
            # (INT4, and at bits==2 the e4m3 feed dequant_int2_to_f4=0,
            # fold-on or fold-off). The f4 arm's stream structure with this
            # width's e4m3 deposits: the first half of the dequant warps
            # deposits ALL K blocks while the second half deposits ALL V
            # blocks in parallel, each stream consuming its OWN code ring.
            # Drain policy: INT4 keeps per-block fence+commit; the int2
            # feed merges the per-tile drains (merge_k_drain /
            # merge_v_drain). No cross-stream walk exists: every wait is an
            # own-stage wait paced by the stream's own releases.
            # FOLD-OFF on the streams: each stream waits the tile's
            # pages_s scale stages at the top of its own tile, reads only
            # its own tensor's (s, z) for the token rows it maps (K: the
            # kfs_rows of its 2*kf_runs flat chunks; V: its token column),
            # fences the generic-proxy quad reads before releasing to the
            # TMA producer (the same ordering the softmax's scale release
            # carries), and releases. Both streams walk
            # every stage, so the ring's consumer group in `kernel` is
            # the full dequant warp count.
            if dq_tidx < i4s_threads:
                # ---- K stream (all K blocks; 2*kf_runs linear
                # chunks/thread)
                # (thread, word, run, stage): run q of thread t is linear
                # chunk t + q*i4s_threads of the flat stage, which is
                # token row kfs_rows[q] at sub-run kfs_cir[q]; the deposit
                # derives its (row, dim column) from those.
                codeKfS4_w = cute.make_tensor(
                    sCode_w,
                    cute.make_layout(
                        (
                            i4s_threads,
                            kf_chunk_words,
                            2 * kf_runs,
                            code_stages,
                        ),
                        stride=(
                            kf_chunk_words,
                            1,
                            i4s_threads * kf_chunk_words,
                            stage_words,
                        ),
                    ),
                )
                tDsCKf4S = codeKfS4_w[dq_tidx, None, None, None]
                # Built under range_constexpr (the sc_handles precedent:
                # a plain `for` refuses the list append)
                kfs_rows = []
                kfs_cir = []
                for _q in cutlass.range_constexpr(2 * kf_runs):
                    _c = dq_tidx + _q * i4s_threads
                    kfs_rows.append(_c // kf_cpr)
                    kfs_cir.append(_c % kf_cpr)
                # The INT4 K deposit addresses the ring directly: the ring's byte
                # pointer, the stage base, trace-time page and run offsets, and this
                # thread's two offsets for the 16-byte halves of each run
                # (`deposit_lane_offsets`). The INT2/E4M3 deposit stores through the
                # swizzled view at (kfs_rows, kfs_cir) instead.
                if cutlass.const_expr(bits == 4):
                    kf_row_bytes = kv_stage_bytes // mma_tile_m
                    assert kf_cpr * kf_out_bytes == kf_row_bytes
                    swz_k = tAsK.iterator.type.swizzle_type
                    assert (swz_k.num_base, swz_k.num_shift) == (4, 3) and (
                        16 << swz_k.num_bits == kf_row_bytes), (
                        f"the K swizzle {swz_k} does not span a {kf_row_bytes}-byte row")
                    k_ring = cute.recast_ptr(tAsK.iterator, dtype=Uint8)
                    k_page_off = page_size * kf_row_bytes
                    k_run_off = (i4s_threads // kf_cpr) * kf_row_bytes
                    # `cute.assume` states what the helper guarantees, that each offset is
                    # a multiple of 16, so every deposit pointer below is 16-byte aligned by
                    # type and its page and run offsets fold into the store's immediate.
                    # (`Pointer.align(16)` would instead round the address up at run time.)
                    k_lane = [cute.assume(c, divby=16) for c in self.deposit_lane_offsets(
                        dq_tidx, k_ring.toint() >> 7, kf_row_bytes, kf_out_bytes,
                        (kv_stage_bytes, k_page_off, k_run_off))]
                    rOutS4H = cute.make_tensor(
                        rOutS4.iterator,
                        cute.make_layout((16, kf_out_bytes // 16, 2 * kf_runs)),
                    )  # (byte, 16-byte half, run)
                for s in cutlass.range(iters_s):
                    if cutlass.const_expr(nofold):
                        # Tile s's scale stages, in the page order the
                        # TMA-QK warp issues them: wait them all, read
                        # the K quad of each of this thread's token rows
                        # (one per flat chunk), build the per-(page, run)
                        # K table, fence the generic-proxy reads, and
                        # early-release the whole tile (everything
                        # downstream reads registers).
                        sc_handles = []
                        for _pj in cutlass.range_constexpr(pages_s):
                            sc_handles.append(
                                scale_consumer.wait_and_advance()
                            )
                        sc_stage0 = sc_handles[0].index
                        rSZ4 = cute.make_rmem_tensor(4, acc_dtype)
                        for pj in cutlass.range_constexpr(pages_s):
                            stg_j = (sc_stage0 + pj) % scale_stages
                            for _q in cutlass.range_constexpr(2 * kf_runs):
                                sSZ = cute.make_tensor(
                                    (
                                        sScale.iterator
                                        + (kfs_rows[_q] * 4
                                           + stg_j * page_size * 4)
                                    ).align(16),
                                    cute.make_layout(4),
                                )
                                cute.autovec_copy(sSZ, rSZ4)
                                ks_b, kz_b = dq_coeffs(rSZ4[0], rSZ4[1])
                                rTabK[pj, _q, 0] = ks_b
                                rTabK[pj, _q, 1] = kz_b
                        # generic-proxy LDS -> async-proxy fence ->
                        # release: the TMA producer refills the stage
                        # through the async proxy, and without the
                        # bridge the refill may land before these reads.
                        # The release
                        # arrives from lane 0 only, so the warp barrier
                        # orders every lane's fence before lane 0's
                        # `mbarrier.arrive` (CROSS-PROXY ORDERING).
                        cute.arch.fence_view_async_shared()
                        cute.arch.sync_warp()
                        for sh in sc_handles:
                            sh.release()
                    ck = codek_consumer.wait_and_advance()
                    for _q in cutlass.range_constexpr(2 * kf_runs):
                        load_src(
                            tDsCKf4S[None, _q, None],
                            ck.index,
                            rSrcS4R[None, _q, 0],
                            kf_chunk_words,
                        )
                    k_code_handles = [ck]
                    k_kv_handles = [k_producer.acquire_and_advance()]
                    for j in cutlass.range_constexpr(nkc):
                        if cutlass.const_expr(j + 1 < nkc):
                            ckn = codek_consumer.wait_and_advance()
                            for _q in cutlass.range_constexpr(2 * kf_runs):
                                load_src(
                                    tDsCKf4S[None, _q, None],
                                    ckn.index,
                                    rSrcS4R[None, _q, (j + 1) % 2],
                                    kf_chunk_words,
                                )
                            k_code_handles.append(ckn)
                        kb, pm = j // pages_m, j % pages_m
                        if cutlass.const_expr(bits == 4):
                            # one map over both slots' words: out bytes
                            # land in run order (source-word order), so
                            # run q's 16-byte half m is rOutS4H[None, m, q]
                            map_int4(
                                rSrcS4F[None, j % 2],
                                rOutS4Uf,
                                2 * words_pt,
                            )
                        elif cutlass.const_expr(nofold):
                            # int2 fold-off: one map per run, each with
                            # its own token row's (s, z). kb enumerates
                            # (sm, dk), so the D-contraction tiles share
                            # a page's coefficients.
                            pgk = (kb // tiles_dk) * pages_m + pm
                            for _q in cutlass.range_constexpr(2 * kf_runs):
                                map_int2_k_dq(
                                    rSrcS4R[None, _q, j % 2],
                                    rTabK[pgk, _q, 0],
                                    rTabK[pgk, _q, 1],
                                    rOutS4UR[None, _q],
                                    kf_chunk_words,
                                )
                        else:
                            # int2 fold-on: the e4m3 K map, per run
                            for _q in cutlass.range_constexpr(2 * kf_runs):
                                map_int2_k(
                                    rSrcS4R[None, _q, j % 2],
                                    rOutS4UR[None, _q],
                                    kf_chunk_words,
                                )
                        if cutlass.const_expr(bits == 4):
                            k_stage = k_ring + cute.assume(
                                k_kv_handles[kb].index * kv_stage_bytes,
                                divby=kv_stage_bytes)
                            for _m in cutlass.range_constexpr(kf_out_bytes // 16):
                                k_dst = k_stage + k_lane[_m]
                                assert k_dst.alignment >= 16, k_dst.alignment
                                for _q in cutlass.range_constexpr(2 * kf_runs):
                                    cute.autovec_copy(
                                        rOutS4H[None, _m, _q],
                                        cute.make_tensor(
                                            k_dst + (pm * k_page_off + _q * k_run_off),
                                            cute.make_layout(16),
                                        ),
                                    )
                        else:
                            for _q in cutlass.range_constexpr(2 * kf_runs):
                                dstK = cute.local_tile(
                                    sK_op[
                                        None,
                                        None,
                                        k_kv_handles[kb].index,
                                    ],
                                    (1, kf_out_bytes),
                                    (
                                        pm * page_size + kfs_rows[_q],
                                        kfs_cir[_q],
                                    ),
                                )
                                cute.autovec_copy(
                                    rOutS4R[None, _q], dstK[0, None]
                                )
                        if cutlass.const_expr(pm == pages_m - 1):
                            if cutlass.const_expr(not merge_k_drain):
                                # STS -> async-proxy fence -> commit
                                cute.arch.fence_view_async_shared()
                                k_kv_handles[kb].commit()
                                # Release, behind the fence above, every code stage whose LDS
                                # precedes it: this block's, and the next stage, prefetched
                                # one ahead, which its own block then skips (CROSS-PROXY
                                # ORDERING).
                                for _jj in cutlass.range_constexpr(
                                        pages_m - min(1, j // pages_m) + min(1, nkc - 1 - j)):
                                    release_code_stage(k_code_handles[j - pages_m + 1 + min(1, j // pages_m) + _jj])
                            if cutlass.const_expr(j + 1 < nkc):
                                k_kv_handles.append(
                                    k_producer.acquire_and_advance()
                                )
                    if cutlass.const_expr(merge_k_drain):
                        # int2: ONE async-proxy fence covers ALL the
                        # tile's K STS, then the per-stage commits
                        # arrive back-to-back (the f4 stream arm's
                        # drain policy).
                        cute.arch.fence_view_async_shared()
                        for kb2 in cutlass.range_constexpr(
                            k_deposit_blocks
                        ):
                            k_kv_handles[kb2].commit()
                        # The tile's code stages, released behind the merged drain's
                        # fence (CROSS-PROXY ORDERING).
                        for _jj in cutlass.range_constexpr(nkc):
                            release_code_stage(k_code_handles[_jj])
            else:
                # ---- V stream. INT4 V is stored exactly like K
                # (`v_token_rows`), so its code stage is K's box and this
                # stream runs the K stream's map at both head tiles: thread
                # t' owns the 16-byte code chunks t' + 64 q of the page,
                # token row (t' + 64 q) // kf_cpr at channel run t' % kf_cpr,
                # each expanding to kf_out_bytes channels of that token. The
                # MN-major V stage (MN_SW128) has K_SW128's bytes and swizzle
                # with rows and columns swapped, so the deposit is the K
                # stream's `deposit_lane_offsets` over the V ring, and each
                # 8-lane quarter warp loads 128 contiguous code bytes and
                # deposits into 8 distinct 16-byte bank groups at every
                # operand-ring start row.
                # The INT2/E4M3 ablation and the fold-off arm keep their
                # token-column ownership of a channel-row-shaped pool.
                i4vt = dq_tidx - i4s_threads
                i4v_unit = i4vt % page_size
                i4v_subs = [
                    i4vt // page_size,
                    (i4vt + i4s_threads) // page_size,
                ]
                if cutlass.const_expr(bits == 4):
                    codeVfS4_w = cute.make_tensor(
                        sCode_w + self.code_stages_k * stage_words,
                        cute.make_layout(
                            (i4s_threads, kf_chunk_words, 2 * kf_runs,
                             self.code_stages_v),
                            stride=(kf_chunk_words, 1,
                                    i4s_threads * kf_chunk_words, stage_words),
                        ),
                    )
                    tDsCV4R = codeVfS4_w[i4vt, None, None, None]
                    # The deposit into the V stages, as the K stream's above: the
                    # V ring's base (the V stages follow the K ring's), trace-time
                    # page and run offsets, and this thread's two lane offsets. A
                    # V stage row is one token's mma_tile_m channel bytes.
                    vf_row_bytes = kv_stage_bytes // mma_tile_k
                    assert kf_cpr * kf_out_bytes == vf_row_bytes
                    swz_v = tAsV.iterator.type.swizzle_type
                    assert (swz_v.num_base, swz_v.num_shift) == (4, 3) and (
                        16 << swz_v.num_bits == vf_row_bytes), (
                        f"the V swizzle {swz_v} does not span a {vf_row_bytes}-byte "
                        f"token row")
                    v_ring = cute.recast_ptr(tAsK.iterator, dtype=Uint8)
                    v_base = v_ring + k_stages * kv_stage_bytes
                    v_page_off = page_size * vf_row_bytes
                    v_run_off = (i4s_threads // kf_cpr) * vf_row_bytes
                    v_lane = [cute.assume(c, divby=16) for c in self.deposit_lane_offsets(
                        i4vt, v_ring.toint() >> 7, vf_row_bytes, kf_out_bytes,
                        (kv_stage_bytes, k_stages * kv_stage_bytes, v_page_off,
                         v_run_off))]
                    rOutV4H = cute.make_tensor(
                        rOutS4.iterator,
                        cute.make_layout((16, kf_out_bytes // 16, 2 * kf_runs)),
                    )  # (byte, 16-byte half, run)
                else:
                    tDsCV4 = codeV_w[i4v_unit, None, None, None]
                for s in cutlass.range(iters_s):
                    if cutlass.const_expr(nofold):
                        # Tile s's scale stages (the same walk as the K
                        # stream's, prefetch_iters tiles later in wall
                        # time): read the V quad of this thread's token
                        # column from each page, build the V table,
                        # fence, early-release.
                        scv_handles = []
                        for _pj in cutlass.range_constexpr(pages_s):
                            scv_handles.append(
                                scale_consumer.wait_and_advance()
                            )
                        scv_stage0 = scv_handles[0].index
                        rSZ4v = cute.make_rmem_tensor(4, acc_dtype)
                        for pj in cutlass.range_constexpr(pages_s):
                            stg_jv = (scv_stage0 + pj) % scale_stages
                            sSZv = cute.make_tensor(
                                (
                                    sScale.iterator
                                    + (i4v_unit * 4
                                       + stg_jv * page_size * 4)
                                ).align(16),
                                cute.make_layout(4),
                            )
                            cute.autovec_copy(sSZv, rSZ4v)
                            vs_b, vz_b = dq_coeffs(rSZ4v[2], rSZ4v[3])
                            rTabV[pj, 0] = vs_b
                            rTabV[pj, 1] = vz_b
                        # generic-proxy LDS -> async-proxy fence -> warp
                        # barrier -> release (see the K stream). Names are
                        # the V stream's own: a name bound in only one arm
                        # of the dynamic if/else is the pattern every
                        # stream arm here follows.
                        cute.arch.fence_view_async_shared()
                        cute.arch.sync_warp()
                        for sh in scv_handles:
                            sh.release()
                    cv = codev_consumer.wait_and_advance()
                    if cutlass.const_expr(bits == 4):
                        for q in cutlass.range_constexpr(2 * kf_runs):
                            load_src(tDsCV4R[None, q, None], cv.index,
                                     rSrcS4R[None, q, 0], kf_chunk_words)
                    else:
                        for hh in cutlass.range_constexpr(2):
                            load_src(
                                tDsCV4[None, i4v_subs[hh], None],
                                cv.index,
                                rSrcS4[None, hh, 0],
                            )
                    v_code_handles = [cv]
                    v_kv_handles = [v_producer.acquire_and_advance()]
                    for j in cutlass.range_constexpr(nvc):
                        if cutlass.const_expr(j + 1 < nvc):
                            cvn = codev_consumer.wait_and_advance()
                            if cutlass.const_expr(bits == 4):
                                for q in cutlass.range_constexpr(2 * kf_runs):
                                    load_src(tDsCV4R[None, q, None], cvn.index,
                                             rSrcS4R[None, q, (j + 1) % 2],
                                             kf_chunk_words)
                            else:
                                for hh in cutlass.range_constexpr(2):
                                    load_src(
                                        tDsCV4[None, i4v_subs[hh], None],
                                        cvn.index,
                                        rSrcS4[None, hh, (j + 1) % 2],
                                    )
                            v_code_handles.append(cvn)
                        vb, pk = j // pages_k, j % pages_k
                        if cutlass.const_expr(bits == 4):
                            map_int4(rSrcS4F[None, j % 2], rOutS4Uf,
                                     2 * words_pt)
                        else:
                            for hh in cutlass.range_constexpr(2):
                                if cutlass.const_expr(nofold):
                                    # vb enumerates (sk, dm); head-dim tiles
                                    # share the same page's coefficients.
                                    pgv = (vb // tiles_dm) * pages_k + pk
                                    map_int2_v_dq(
                                        rSrcS4[None, hh, j % 2],
                                        rTabV[pgv, 0], rTabV[pgv, 1],
                                        rOutS4U[None, hh], words_pt,
                                    )
                                else:
                                    map_int2_v(rSrcS4[None, hh, j % 2],
                                               rOutS4U[None, hh], words_pt)
                        if cutlass.const_expr(bits == 4):
                            v_stage = v_base + cute.assume(
                                v_kv_handles[vb].index * kv_stage_bytes,
                                divby=kv_stage_bytes)
                            for _m in cutlass.range_constexpr(kf_out_bytes // 16):
                                v_dst = v_stage + v_lane[_m]
                                assert v_dst.alignment >= 16, v_dst.alignment
                                for q in cutlass.range_constexpr(2 * kf_runs):
                                    cute.autovec_copy(
                                        rOutV4H[None, _m, q],
                                        cute.make_tensor(
                                            v_dst + (pk * v_page_off + q * v_run_off),
                                            cute.make_layout(16),
                                        ),
                                    )
                        else:
                            for hh in cutlass.range_constexpr(2):
                                dstV = cute.local_tile(
                                    sV_op[None, None,
                                          k_stages + v_kv_handles[vb].index],
                                    (out_bytes, 1),
                                    (i4v_subs[hh], pk * page_size + i4v_unit),
                                )
                                cute.autovec_copy(rOutS4b[None, hh], dstV[None, 0])
                        if cutlass.const_expr(pk == pages_k - 1):
                            if cutlass.const_expr(not merge_v_drain):
                                cute.arch.fence_view_async_shared()
                                v_kv_handles[vb].commit()
                                # Release, behind the fence above, every code stage whose LDS
                                # precedes it: this block's, and the next stage, prefetched
                                # one ahead, which its own block then skips (CROSS-PROXY
                                # ORDERING).
                                for _jj in cutlass.range_constexpr(
                                        pages_k - min(1, j // pages_k) + min(1, nvc - 1 - j)):
                                    release_code_stage(v_code_handles[j - pages_k + 1 + min(1, j // pages_k) + _jj])
                            if cutlass.const_expr(j + 1 < nvc):
                                v_kv_handles.append(
                                    v_producer.acquire_and_advance()
                                )
                    if cutlass.const_expr(merge_v_drain):
                        # int2: the V twin of the merged K drain above
                        cute.arch.fence_view_async_shared()
                        for vb2 in cutlass.range_constexpr(
                            v_deposit_blocks
                        ):
                            v_kv_handles[vb2].commit()
                        # The tile's code stages, released behind the merged drain's
                        # fence (CROSS-PROXY ORDERING).
                        for _jj in cutlass.range_constexpr(nvc):
                            release_code_stage(v_code_handles[_jj])


    @cute.jit
    def _correction_warpgroup(
        self,
        _item,
        acc_dtype,
        blk_tile_n,
        coord_hk,
        correction_regs,
        enable_blasst,
        fold_p,
        iters_s,
        k_ch_off,
        k_channel,
        k_pages,
        k_parts,
        kb_ahead,
        kch_beta,
        kch_lift_mul,
        kch_s_mul,
        kv_split_idx,
        kv_splits,
        lane_idx,
        lane_store_max,
        lc_consumer,
        mQ,
        mSC_flat,
        mma_dtype,
        mma_tile_k,
        mma_tile_n,
        o_consumer,
        o_dtype,
        o_gmem,
        o_ring,
        page_count,
        page_table_iter,
        pages_s,
        pscale_lp,
        lp_table,
        lp_shift,
        lp_cell,
        q_load_mbar,
        qb_producer,
        qp_producer,
        qp_stages,
        rebalance_regs,
        sC,
        sL,
        sLC,
        sL_final_nbar,
        sM,
        sM_consumer_nbar,
        sM_final_nbar,
        sM_producer_nbar,
        sO_final_nbar,
        sQB,
        sQP,
        sSP_i32,
        seqlens_iter,
        sp_consumer,
        tBsQ,
        tCsO,
        tCtL,
        tCtO,
        tL_consumer_nbar,
        tL_producer_nbar,
        table_offset,
        tiles_dk,
        tiles_dm,
        v_ch_off,
        v_channel,
        v_pages,
        v_parts,
        vch_beta,
        vch_s_mul,
        warpgroup_tidx,
        warpgroup_widx,
        with_phase,
    ):
        """The correction warpgroup: the O rescale chain and the V-fold epilogue.

        What it does, in order: build the channel-K Q'-plane group and its scales; prime
        a depth-`o_stages` colmax window; drain each tile's fresh-written O accumulator
        out of tmem into per-thread registers, rescaling by the running max; and run the
        V-fold epilogue, which combines the hi and lo P rows, divides by the final
        pscale, adds the cross-warp-reduced `c_h`, and stores O.
        """
        # Read straight off `self` instead of taking 21 parameters that were
        # only ever `X = self.X` in the caller. Same expression, same object,
        # and the body below is untouched -- these are the names it uses.
        do_atomic_red = self.do_atomic_red
        do_none_red = self.do_none_red
        o_stages = self.o_stages
        persistent = self.persistent
        tma_mask = self.tma_mask
        if cutlass.const_expr(rebalance_regs):
            cute.arch.setmaxregister_increase(correction_regs)
        # A one-item CTA's exponent under token-wise K (`lp_table`, see the kernel
        # prologue), read before this warpgroup's set-up for the first tile, while the
        # producers fill it. A one-item CTA gives this warpgroup the shared-memory cell
        # `lp_cell` instead under channel-wise K, which it reads at the epilogue, and
        # under token-wise K where `lp_tail` holds, which it reads at the start of its
        # tail; a persistent CTA gives it the exponent itself.
        if cutlass.const_expr(lp_table is not None):
            cute.arch.griddepcontrol_wait()
            pscale_lp = self._pool_pscale_lp(self._vmax_hist_loads(lp_table, lane_idx),
                                             lane_idx, lp_shift)

        # Select copy atoms for O and L. Fold mode widens the tmem
        # loads/stores to 2*blk_tile_n columns: O columns [g, 2g) hold
        # the P-lo MMA results (they accumulate and must ride the same
        # correction rescale), L columns [g, 2g) carry the c_h partials.
        # The final smem O store stays blk_tile_n wide (hi/lo combined).
        nL = blk_tile_n * (2 if fold_p else 1)
        tmem_repeat_op_o = tcgen05.Repetition(nL)
        tmem_load_op_o = tcgen05.Ld32x32bOp(tmem_repeat_op_o)
        tmem_store_op_o = tcgen05.St32x32bOp(tmem_repeat_op_o)
        tmem_load_atom_o = cute.make_copy_atom(tmem_load_op_o, acc_dtype)
        tmem_store_atom_o = cute.make_copy_atom(tmem_store_op_o, acc_dtype)
        # Tile atoms and slice
        tCtO_dm = tCtO[mma_dice + (0, 0)]
        tmem_load_o = tcgen05.make_tmem_copy(tmem_load_atom_o, tCtO_dm)
        thr_load_o = tmem_load_o.get_slice(warpgroup_tidx)
        # Partition O and L
        # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #TILE_DM, o_stages)
        tOtO = thr_load_o.partition_S(tCtO)
        tOsO_w = thr_load_o.partition_D(tCsO)  # wide (nL): rmem shapes
        if cutlass.const_expr(fold_p):
            # Narrow (blk_tile_n) partition for the final smem O store
            # (the hi/lo-combined output is g wide)
            tmem_load_atom_o_n = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(blk_tile_n)),
                acc_dtype,
            )
            tmem_load_o_n = tcgen05.make_tmem_copy(
                tmem_load_atom_o_n, tCtO_dm
            )
            tOsO = tmem_load_o_n.get_slice(warpgroup_tidx).partition_D(
                tCsO
            )
        # Slice unused modes
        tOtO = tOtO[None, 0, 0, 0, None, None]  # (CPY, #TILE_DM, o_ring)
        tOsO_w = tOsO_w[None, 0, 0, 0, None]  # (CPY_w, #TILE_DM)
        if cutlass.const_expr(fold_p):
            tOsO = tOsO[None, 0, 0, 0, None]  # (CPY_n, #TILE_DM)
        else:
            tOsO = tOsO_w
        def colsum_load(phase):
            # placeholder: closure-captured by the loop regions
            # even when every call site is const-pruned
            # (channel-V); the real def below overwrites it.
            raise AssertionError("colsum_load unused (channel-V)")

        if cutlass.const_expr(not v_channel):
            # (CPY, #CPY_MMA, #CPY_M, #CPY_N, o_stages)
            tOtL = thr_load_o.partition_S(tCtL)
            tOtL = tOtL[None, 0, 0, 0, None]  # (CPY, o_stages)

            # colsum load helper (fold: [colsum | c] over nL cols)
            def colsum_load(
                phase,
                nL=nL,
                tOtL=tOtL,
                tOrO_shape=tOsO_w.shape[:1],
                tmem_load_atom_o=tmem_load_atom_o,
                tL_producer_nbar=tL_producer_nbar,
                tL_consumer_nbar=tL_consumer_nbar,
            ):
                with_phase(tL_producer_nbar, phase).arrive_and_wait()
                tOtL_s = tOtL[None, phase]
                tOrL_s = cute.make_rmem_tensor(tOrO_shape, Float32)
                cute.copy(tmem_load_atom_o, tOtL_s, tOrL_s)
                cute.arch.fence_view_async_tmem_load()
                with_phase(tL_consumer_nbar, phase).arrive()
                return tOrL_s.load().reshape(nL)

        # Initialize O and colsum in tmem
        tOrO = cute.make_rmem_tensor(tOsO_w.shape, acc_dtype)
        tOrO.fill(Float32(0))
        for phase in cutlass.range_constexpr(o_ring):
            cute.copy(tmem_store_atom_o, tOrO, tOtO[None, None, phase])
        # Zero every tL slot no tile may produce (iters_s < o_stages
        # tails consume all o_stages slots; slot 0 always gets tile 0)
        if cutlass.const_expr(not v_channel):
            for phase in cutlass.range_constexpr(1, o_stages, 1):
                cute.copy(
                    tmem_store_atom_o, tOrO[None, 0], tOtL[None, phase]
                )
        cute.arch.fence_view_async_tmem_store()

        # Initialize consumer barriers. One-shot
        # credits prime on item 0 only — each item ends with the
        # same +1 dangling credit per barrier (the compensation
        # arrivals enforce per-item closure), so the end-of-item
        # state IS the next item's init state.
        if cutlass.const_expr(persistent):
            if _item == 0:
                sM_consumer_nbar.arrive()
                if cutlass.const_expr(not v_channel):
                    for phase in cutlass.range_constexpr(o_stages):
                        with_phase(tL_consumer_nbar, phase).arrive()
        else:
            sM_consumer_nbar.arrive()
            if cutlass.const_expr(not v_channel):
                for phase in cutlass.range_constexpr(o_stages):
                    with_phase(tL_consumer_nbar, phase).arrive()

        # Initialize colsum in RF (fold: [colsum | c] over nL columns —
        # both accumulate in TRUE units with the m-only correction)
        colsum_p = cute.make_rmem_tensor((nL, o_stages), Float32)
        colsum_p.fill(Float32(0))

        def build_kgroup(s_buf, z_buf, kp, qp_h, qb_h):
            # placeholder: closure-captured by the loop regions
            # even when every call site is const-pruned (non-
            # channel-K); the real def below overwrites it.
            raise AssertionError("build_kgroup unused")

        def load_kscales(tb, kp, s_buf, z_buf):
            raise AssertionError("load_kscales unused")

        # ---- channel-K Q'-plane builder ----
        # Deliberately NOT on the TMA-issue warp: building it there
        # serialized the producer chain, halving issue_active for
        # +17% instructions. This WG has 4x the threads and measured
        # slack, and builds run kb_ahead+1 tiles ahead of every
        # consumer (the preamble seeds the window), so the inline
        # scale-load latency sits off the critical path entirely and
        # no prefetch registers are needed. Work is also ~6x smaller
        # per thread: Q is ITEM-HOISTED into registers (rows
        # round-robined to warps, lane owns the d-run
        # [4*lane, 4*lane+4)), and the po2-lift amax uses the EXACT
        # identity max|s_d*Q[n]| = max_d(s_d * |Q[n,d]|), since
        # per-lane |Q| is item-constant.
        #
        # THE LIFT IS PER ROW, and that is what keeps the amax
        # reduction warp-local. A plane-wide scalar lift needed an
        # smem round through `sKam` and a 128-arrive `kam_mbar` in
        # the middle of the build's dependency chain, because row n
        # lives in warp n % 4. Per row the reduction is exactly
        # `warp_fmax` over the lane's d-run: no named barrier, no
        # smem, and at g <= 4 each warp owns one row, so the
        # rendezvous was purely DISCARDING the per-row answer to make
        # a coarser one. Per row is also strictly TIGHTER
        # quantization, since each row gets its own lift instead of
        # sharing the max over all rows, and the host reference
        # mirrors it row-wise. Bias was already warp-local.
        if cutlass.const_expr(k_channel):
            cute.arch.mbarrier_wait(q_load_mbar, phase=0)
            sQP_nk = cute.composition(
                sQP,
                cute.make_layout(
                    (mma_tile_n, mma_tile_k, qp_stages)
                ),
            )
            sQ_nk = cute.composition(
                tBsQ,
                cute.make_layout(
                    (mma_tile_n, mma_tile_k, tiles_dk)
                ),
            )
            sQ_d4 = cute.flat_divide(sQ_nk, (1, 4))
            sQP_d4 = cute.flat_divide(sQP_nk, (1, 4))
            kb_rows = (blk_tile_n + warpgroup_warps - 1) // warpgroup_warps
            rQh = cute.make_rmem_tensor((kb_rows, 4), acc_dtype)
            rQh.fill(Float32(0))
            # PER-ROW |Q|: rMaxq[rr, j] is row (rr*4 + widx)'s own
            # |Q[.., 4*lane + j]|, NOT maxed across the warp's rows.
            # At g == 8 (kb_rows == 2) a warp owns two rows and the
            # old single-row-slot version deliberately collapsed
            # them; keeping them separate is what makes the lift
            # per row.
            rMaxq = cute.make_rmem_tensor((kb_rows, 4), acc_dtype)
            rMaxq.fill(Float32(0))
            rq4b = cute.make_rmem_tensor((1, 4), mma_dtype)
            rqfb = cute.make_rmem_tensor((1, 4), acc_dtype)
            for rr in cutlass.range_constexpr(kb_rows):
                n_h = rr * warpgroup_warps + warpgroup_widx
                if n_h < blk_tile_n:
                    cute.autovec_copy(
                        sQ_d4[None, None, n_h, lane_idx, 0], rq4b
                    )
                    rqfb.store(rq4b.load().to(acc_dtype))
                    for j in cutlass.range_constexpr(4):
                        qv_h = rqfb[0, j]
                        rQh[rr, j] = qv_h
                        rMaxq[rr, j] = cute.arch.fmax(qv_h, -qv_h)
            gPT_k = cute.make_tensor(
                page_table_iter + table_offset, (page_count,)
            )
            # VECTORIZED channel-scale view. `mSC_flat` is
            # (block_tot, (h_k, NP)) with the block mode at STRIDE 1,
            # so a lane's 4-channel run is 4 CONTIGUOUS elements;
            # dividing that mode by 4 gives a (4, block_tot/4, ...)
            # view whose leading mode is exactly the run, and
            # `autovec_copy` then issues ONE vector load instead of
            # four scalar ones. Same idiom as the Q load above
            # (`sQ_d4`). Legal because `quant_block_offsets` puts
            # `k_ch` at 128 and `head_dim` is 128, so both
            # `k_ch_off` and `k_ch_off + hd` are multiples of 4.
            assert k_ch_off % 4 == 0 and self.head_dim % 4 == 0, (
                f"vectorized channel-scale load needs 4-element "
                f"alignment; k_ch={k_ch_off}, head_dim={self.head_dim}")
            mSC_d4 = cute.flat_divide(mSC_flat, (4,))
            ks_col = k_ch_off // 4
            kz_col = (k_ch_off + self.head_dim) // 4

            def load_kscales(
                tb, kp, s_buf, z_buf,
                mSC_d4=mSC_d4, gPT_k=gPT_k,
                kv_splits=kv_splits, kv_split_idx=kv_split_idx,
                pages_s=pages_s, k_pages=k_pages,
                ks_col=ks_col, kz_col=kz_col,
                page_count=page_count,
                coord_hk=coord_hk, lane_idx=lane_idx,
                scale_dtype=self.scale_dtype, acc_dtype=acc_dtype,
            ):
                # raw (s_d, z_d) for this lane's d-run into slot
                # `kp` of the caller's per-part buffer (zeros on OOB).
                # PER-PART storage, not one shared register set: two
                # builds that alias the same s_raw serialize on a
                # write-after-read no matter how the acquires are
                # ordered.
                for j in cutlass.range_constexpr(4):
                    s_buf[kp, j] = Float32(0)
                    z_buf[kp, j] = Float32(0)
                lp0b = (
                    tb * kv_splits + kv_split_idx
                ) * pages_s + kp * k_pages
                if lp0b < page_count:
                    phys_b = gPT_k[lp0b]
                    rs4 = cute.make_rmem_tensor(4, scale_dtype)
                    rz4 = cute.make_rmem_tensor(4, scale_dtype)
                    cute.autovec_copy(
                        mSC_d4[
                            None, ks_col + lane_idx,
                            (coord_hk, phys_b),
                        ],
                        rs4,
                    )
                    cute.autovec_copy(
                        mSC_d4[
                            None, kz_col + lane_idx,
                            (coord_hk, phys_b),
                        ],
                        rz4,
                    )
                    for j in cutlass.range_constexpr(4):
                        s_buf[kp, j] = rs4[j].to(acc_dtype)
                        z_buf[kp, j] = rz4[j].to(acc_dtype)

            def build_kgroup(
                s_buf, z_buf, kp, qp_h, qb_h,
                rQh=rQh, rMaxq=rMaxq, sQB=sQB,
                sQP_d4=sQP_d4,
                kb_rows=kb_rows, acc_dtype=acc_dtype,
                mma_dtype=mma_dtype, kch_s_mul=kch_s_mul,
                kch_lift_mul=kch_lift_mul,
                kch_beta=kch_beta, lane_idx=lane_idx,
                warpgroup_widx=warpgroup_widx,
                blk_tile_n=blk_tile_n,
                warpgroup_warps=warpgroup_warps,
            ):
                s_eff = cute.make_rmem_tensor(4, acc_dtype)
                z_eff = cute.make_rmem_tensor(4, acc_dtype)
                for j in cutlass.range_constexpr(4):
                    s_eff[j] = s_buf[kp, j] * Float32(kch_s_mul)
                    z_eff[j] = (
                        z_buf[kp, j]
                        - Float32(kch_beta) * s_buf[kp, j]
                    )
                rv4b = cute.make_rmem_tensor((1, 4), acc_dtype)
                rp4b = cute.make_rmem_tensor((1, 4), mma_dtype)
                for rr in cutlass.range_constexpr(kb_rows):
                    n_b = rr * warpgroup_warps + warpgroup_widx
                    # EXACT per-row amax: products at the true
                    # argmax equal the elementwise form's (same f32
                    # values, same ladder m). `warp_fmax` covers
                    # d = 4*lane .. 4*lane+3 over 32 lanes = D, and
                    # row n_b is owned by THIS warp alone, so this
                    # is the whole reduction — it must run on every
                    # lane, hence outside the n_b guard below.
                    amax_p = Float32(0)
                    for j in cutlass.range_constexpr(4):
                        amax_p = cute.arch.fmax(
                            amax_p, s_eff[j] * rMaxq[rr, j]
                        )
                    amax = warp_fmax(amax_p)
                    m_g = _lift_exp(amax)
                    inv_m = exp2(Float32(0) - Float32(m_g))
                    lift_v = exp2(Float32(m_g))
                    if cutlass.const_expr(kch_lift_mul != 1.0):
                        # int4 raw-code feed: the lift carries
                        # the 512 (exact po2 FMUL) so the plane
                        # bytes and the ladder m stay unchanged
                        # — see kch_lift_mul at its definition.
                        lift_v = lift_v * Float32(kch_lift_mul)
                    if n_b < blk_tile_n:
                        for j in cutlass.range_constexpr(4):
                            rv4b[0, j] = (
                                s_eff[j] * rQh[rr, j] * inv_m
                            )
                        rp4b.store(rv4b.load().to(mma_dtype))
                        cute.autovec_copy(
                            rp4b,
                            sQP_d4[
                                None, None, n_b, lane_idx,
                                qp_h.index,
                            ],
                        )
                        if lane_idx == 0:
                            # bias ring rows [0, g) = per-row lift
                            sQB[n_b, qb_h.index] = lift_v
                # FENCE COALESCING: the bias is computed and stored
                # BEFORE the fence, so ONE `fence_view_async_shared`
                # orders both payloads and both handles commit after
                # it. Publishing QP after the bias butterfly is free
                # here because the builder runs `kb_ahead + 1` tiles
                # ahead of every consumer, so the plane is not awaited
                # when it is produced.
                for rr in cutlass.range_constexpr(kb_rows):
                    n_b = rr * warpgroup_warps + warpgroup_widx
                    bp = Float32(0)
                    for j in cutlass.range_constexpr(4):
                        bp += z_eff[j] * rQh[rr, j]
                    bn = cute.arch.warp_reduction_sum(bp)
                    if n_b < blk_tile_n:
                        if lane_idx == 0:
                            # rows [g, 2g) = per-row bias
                            sQB[blk_tile_n + n_b, qb_h.index] = bn
                # NO fence and NO commit here: the CALLER publishes
                # all k_parts payloads behind ONE fence. A fence is a
                # reordering barrier, so leaving it per-build would stop
                # ptxas interleaving the independent plane builds even
                # after the per-part acquires are hoisted -- and the
                # per-part acquire + fence pair IS the serializer (the
                # k_parts loop is range_constexpr, so its bodies are
                # already independent unrolled code).

            # seed the build-ahead window BEFORE the sM preamble
            # reads (those need softmax(0..o-1), which need the
            # planes — building after would deadlock)
            n_seed = cutlass.min(Int32(kb_ahead + 1), iters_s)
            # ONE SLOT PER PART. Two builds that alias one scale register
            # set serialize on a write-after-read however the acquires are
            # ordered, so per-part slots are what make the k_parts bodies
            # genuinely independent.
            # Part-major (row-major) slots: each part's 4-channel run is contiguous, so
            # the f32 pairs a DSL >= 4.6 register tensor is packed into are (j, j+1) of
            # ONE part and match the halves of that part's 128-bit scale load. Column-major
            # slots pair two parts' channels, and under DSL 4.8.0 the repacking stalls
            # on each load and costs INT2 channel-wise-K builds 25-32%.
            rSKB_pf = cute.make_rmem_tensor(
                cute.make_layout((k_parts, 4), stride=(4, 1)), acc_dtype)
            rZKB_pf = cute.make_rmem_tensor(
                cute.make_layout((k_parts, 4), stride=(4, 1)), acc_dtype)
            # Every part's loads are issued before the first acquire. An acquire is a
            # wait loop the compiler does not move loads across, so a part's loads
            # placed after the previous part's acquire start one page-table round trip
            # later, once per part.
            for tb_s in cutlass.range(n_seed, unroll=1):
                for kp_s in cutlass.range_constexpr(k_parts):
                    load_kscales(tb_s, kp_s, rSKB_pf, rZKB_pf)
                qp_hs, qb_hs = [], []
                for kp_s in cutlass.range_constexpr(k_parts):
                    qp_hs.append(qp_producer.acquire_and_advance())
                    qb_hs.append(qb_producer.acquire_and_advance())
                for kp_s in cutlass.range_constexpr(k_parts):
                    build_kgroup(rSKB_pf, rZKB_pf, kp_s,
                                 qp_hs[kp_s], qb_hs[kp_s])
                cute.arch.fence_view_async_shared()
                for kp_s in cutlass.range_constexpr(k_parts):
                    qp_hs[kp_s].commit()
                    qb_hs[kp_s].commit()
            rSKB_pf.fill(Float32(0))
            rZKB_pf.fill(Float32(0))
            pf_go = Int32(kb_ahead + 1 < iters_s)
            for _pf in cutlass.range(pf_go, unroll=1):
                for kp_s in cutlass.range_constexpr(k_parts):
                    load_kscales(
                        Int32(kb_ahead + 1), kp_s,
                        rSKB_pf, rZKB_pf
                    )

        # Load colmax of the first o_stages tiles into a depth-o_stages
        # running-max shift register: entering loop iter s it holds
        # M[s], .., M[s+o-1] (oldest first). Missing tiles (iters_s <
        # o_stages) act as EMPTY tiles at the last real running max:
        # the first real read initializes ALL slots and skipped reads
        # shift nothing, so tail weights land on the right slots.
        # (the pscale exponent is LAUNCH-GLOBAL, so the
        # old lp shift register / windowed sVmax reads / O pscale
        # re-basing are gone — corrections are pscale-free by
        # construction, exactly as in the baseline.)
        sM_hist = cute.make_rmem_tensor(o_stages, Float32)
        sM_hist.fill(Float32(0))
        for s in cutlass.range_constexpr(o_stages):
            if s == 0 or s < iters_s:
                if cutlass.const_expr(enable_blasst):
                    sp_handle = sp_consumer.wait_and_advance()
                    sp_handle.release()
                sM_producer_nbar.arrive_and_wait()
                sM_read = Float32(0)
                if lane_store_max:
                    sM_read = sM[lane_idx]
                sM_consumer_nbar.arrive()
                if cutlass.const_expr(s == 0):
                    for j in cutlass.range_constexpr(o_stages):
                        sM_hist[j] = sM_read
                else:
                    for j in cutlass.range_constexpr(o_stages - 1):
                        sM_hist[j] = sM_hist[j + 1]
                    sM_hist[o_stages - 1] = sM_read

        # ---- channel-V drain state ----
        # The VP MMA fresh-writes each tile's O slot, and this WG
        # DRAINS it into per-thread register accumulators (thread
        # t owns O d-row t): rAcc [nL] takes s'_d-scaled slot
        # drains (hi|lo columns, pscale units), rAccZ
        # [blk_tile_n] the z'_d * l_G outer-product terms (l =
        # the tile's consistent quantized-P column sums, tL
        # columns [g, 2g), pscale units — one inv_ps at the
        # epilogue covers everything), colsum_acc_v [nL] the
        # rebased true colsum.
        # Frame: accumulators live in exp2(-m_prev_v) units and
        # rebase by exp2(m_prev - m_tile) at each drain. Group
        # scales (s_d, z_d rows of the page's duplicated channel
        # section) come from gmem LDG via the page table — the
        # scale ring carries only token-wise sections. Every
        # channel group is contained in one sequence tile (Gt <=
        # 256 = sequence_tile, tiles are 256-token aligned), so
        # the tile's FIRST page always carries the right copy.
        if cutlass.const_expr(v_channel):
            rAcc = cute.make_rmem_tensor((nL, tiles_dm), acc_dtype)
            rAcc.fill(Float32(0))
            rAccZ = cute.make_rmem_tensor(blk_tile_n, acc_dtype)
            rAccZ.fill(Float32(0))
            colsum_acc_v = cute.make_rmem_tensor(nL, acc_dtype)
            colsum_acc_v.fill(Float32(0))
            m_prev_v = sM_hist[0]
            gPT_c = cute.make_tensor(
                page_table_iter + table_offset, (page_count,)
            )
            # drain-scale PREFETCH: the next tile's raw (s_d, z_d)
            # for this thread's d-row land in registers one tile
            # early, so the dependent gmem chain (page-table load
            # -> 2 scale LDGs) hides behind a full tile of drain
            # work. Prologue: tile 0.
            # One single-element register tensor per part. The compiler packs a wider f32
            # register tensor into f32 register pairs (sm_100 f32x2), so a part written
            # under its own branch has to be re-packed at the branch join, which waits for
            # that part's scale load right there instead of at the next tile's use.
            rSVD_pf = [cute.make_rmem_tensor(1, acc_dtype) for _ in range(v_parts)]
            rZVD_pf = [cute.make_rmem_tensor(1, acc_dtype) for _ in range(v_parts)]
            for pp in cutlass.range_constexpr(v_parts):
                rSVD_pf[pp].fill(Float32(0))
                rZVD_pf[pp].fill(Float32(0))
            for pp in cutlass.range_constexpr(v_parts):
                lp_pf = kv_split_idx * pages_s + pp * v_pages
                if lp_pf < page_count:
                    phys_pf = gPT_c[lp_pf]
                    rSVD_pf[pp][0] = Float32(
                        mSC_flat[
                            v_ch_off + warpgroup_tidx,
                            (coord_hk, phys_pf),
                        ]
                    )
                    rZVD_pf[pp][0] = Float32(
                        mSC_flat[
                            v_ch_off + self.head_dim
                            + warpgroup_tidx,
                            (coord_hk, phys_pf),
                        ]
                    )

        # Sequence loop
        softmax_phase = 0
        unroll = o_stages if not enable_blasst else 1
        keep_tile = not enable_blasst
        for s in cutlass.range(iters_s - o_stages, unroll=unroll):
            # Load skip predicate
            if cutlass.const_expr(enable_blasst):
                sp_handle = sp_consumer.wait_and_advance()
                keep_tile = sSP_i32[sp_handle.index] != 0
                sp_handle.release()

            if keep_tile:
                # channel-K: keep the build-ahead window full —
                # build tile s + kb_ahead + 1's planes now (0/1-
                # trip dynamic loop = the guarded-handle-safe
                # form; skipped past the last tile)
                if cutlass.const_expr(k_channel):
                    tb_n = s + kb_ahead + 1
                    nb_go = Int32(tb_n < iters_s)
                    for _bb in cutlass.range(nb_go, unroll=1):
                        # OVERLAP the k_parts plane builds. Every
                        # per-part acquire happens FIRST, then the build
                        # bodies run back to back with nothing between
                        # them, then ONE fence publishes every payload.
                        # Previously each part was
                        # acquire -> build -> fence -> commit, and BOTH
                        # the mbarrier acquire and the fence are barriers
                        # ptxas cannot schedule across, so the builds ran
                        # strictly serially however independent their
                        # arithmetic was. The bodies read DISJOINT slots
                        # of rSKB_pf/rZKB_pf, so no register aliasing
                        # serializes them either.
                        #
                        # Holding k_parts handles live at once is safe:
                        # `acquire()` returns an ImmutableResourceHandle
                        # built from `self.__state.clone()`, i.e. a value
                        # SNAPSHOT, not an alias of the producer's live
                        # state.
                        qp_hn, qb_hn = [], []
                        for kp_n in cutlass.range_constexpr(k_parts):
                            qp_hn.append(
                                qp_producer.acquire_and_advance()
                            )
                            qb_hn.append(
                                qb_producer.acquire_and_advance()
                            )
                        for kp_n in cutlass.range_constexpr(k_parts):
                            build_kgroup(
                                rSKB_pf, rZKB_pf, kp_n,
                                qp_hn[kp_n], qb_hn[kp_n],
                            )
                        cute.arch.fence_view_async_shared()
                        for kp_n in cutlass.range_constexpr(k_parts):
                            qp_hn[kp_n].commit()
                            qb_hn[kp_n].commit()
                        # prefetch the NEXT target's scales straight
                        # into the per-part slots (the compiler already
                        # sinks the wait past a staging copy, so a copy
                        # buys nothing)
                        nx_go = Int32(tb_n + 1 < iters_s)
                        for _px in cutlass.range(nx_go, unroll=1):
                            for kp_n in cutlass.range_constexpr(
                                k_parts
                            ):
                                load_kscales(
                                    tb_n + 1, kp_n,
                                    rSKB_pf, rZKB_pf
                                )

                # Acquire O of s-2 FIRST and issue its (fold: 2g-wide)
                # tmem load immediately. The correction WG is the
                # pipeline laggard, so the VP commit landed long ago and
                # this wait is nearly free; issuing the LDTM here
                # overlaps its latency with the tL/sM rendezvous below
                # instead of extending the post-sM VP-blocking segment.
                # Reading the destination registers is scoreboard-safe;
                # the load *view* fence is only needed before the buffer
                # is handed back.
                o_token = o_consumer.try_wait()
                o_handle = o_consumer.wait_and_advance(o_token)
                tOrO_s2 = cute.make_rmem_tensor(
                    (*tOsO_w.shape[:1], tiles_dm), acc_dtype
                )
                for dm in cutlass.range_constexpr(tiles_dm):
                    cute.copy(
                        tmem_load_atom_o,
                        tOtO[None, dm, softmax_phase],
                        tOrO_s2[None, dm],
                    )

                # Load colmax of s
                sM_producer_nbar.arrive_and_wait()
                if s == iters_s - o_stages - 1:
                    sM_final_nbar.arrive()
                sM_lane = Float32(0)
                if lane_store_max:
                    sM_lane = sM[lane_idx]
                sM_consumer_nbar.arrive()

                # Compute correction of s-2. Fold mode: the vector is nL
                # wide (columns [g, 2g) are the lo rows / c partials of
                # head n - g -> factor of head n % g); the pscale is one
                # exponent for the whole pool, so O needs no pscale re-base
                # here and l/c stay in true units.
                correction_lane = exp2(sM_hist[0] - sM_lane)
                correction_r = cute.make_rmem_tensor(nL, acc_dtype)
                for n in cutlass.range_constexpr(nL):
                    correction_r[n] = cute.arch.shuffle_sync(
                        correction_lane, n % blk_tile_n
                    )
                correction = correction_r.load()
                correction_om = correction
                corr_v_r = None
                if cutlass.const_expr(v_channel):
                    # drain rebase: previous drained tile's max ->
                    # THIS tile's max (hist[0] = M[s], pre-shift)
                    corr_v_lane = exp2(m_prev_v - sM_hist[0])
                    m_prev_v = sM_hist[0]
                    corr_v_r = cute.make_rmem_tensor(nL, acc_dtype)
                    for n in cutlass.range_constexpr(nL):
                        corr_v_r[n] = cute.arch.shuffle_sync(
                            corr_v_lane, n % blk_tile_n
                        )
                for j in cutlass.range_constexpr(o_stages - 1):
                    sM_hist[j] = sM_hist[j + 1]
                sM_hist[o_stages - 1] = sM_lane

                # Correct O of s-2 and notify MMA VP
                def rmw_release_o(
                    correction_om=correction_om,
                    tOrO_s2=tOrO_s2,
                    softmax_phase=softmax_phase,
                    o_handle=o_handle,
                    tOtO=tOtO,
                    tOsO_w=tOsO_w,
                    tiles_dm=tiles_dm,
                    acc_dtype=acc_dtype,
                    tmem_store_atom_o=tmem_store_atom_o,
                ):
                    correction_o = correction_om.reshape(tOsO_w.shape[:1])
                    for dm in cutlass.range_constexpr(tiles_dm):
                        tOrO_dm = cute.make_rmem_tensor(
                            tOsO_w.shape[:1], acc_dtype
                        )
                        tOrO_dm.store(
                            correction_o * tOrO_s2[None, dm].load()
                        )
                        cute.copy(
                            tmem_store_atom_o,
                            tOrO_dm,
                            tOtO[None, dm, softmax_phase],
                        )
                    cute.arch.fence_view_async_tmem_store()
                    o_handle.release()

                z_dp = Float32(0)
                if cutlass.const_expr(v_channel):
                    # DRAIN tile s's slot: fetch this tile's group
                    # (s_d, z_d) for this thread's d-row from the
                    # tile's first page (channel sections are
                    # duplicated per page; groups are tile-
                    # contained), rebase rAcc and add the scaled
                    # slot; the slot is CONSUMED — release after
                    # the LDTM view retires, no store-back.
                    s_dp = rSVD_pf[0][0] * Float32(vch_s_mul)
                    z_dp = (
                        rZVD_pf[0][0]
                        - Float32(vch_beta) * rSVD_pf[0][0]
                    )
                    rAcc.store(
                        corr_v_r.load() * rAcc.load()
                        + s_dp
                        * tOrO_s2.load().reshape((nL, tiles_dm))
                    )
                    cute.arch.fence_view_async_tmem_load()
                    o_handle.release()
                else:
                    rmw_release_o()

                # Load, correct and accumulate colsum of s-2, AFTER the O
                # release: the colsum consumption (LDTM-L + fences + tL
                # nbar rendezvous) is correction-WG-local state and only
                # gates the softmax's slot reuse a full window later, so
                # it does not belong on the VP-blocking segment above.
                if cutlass.const_expr(v_channel):
                    # consume this tile's part-0 [colsum | l]
                    # entry (pre-reduced per warp by the softmax
                    # WG; fixed-order 4-slot sum = deterministic),
                    # then the remaining parts (same running-max
                    # frame -> corr == 1 exactly).
                    lch = lc_consumer.wait_and_advance()
                    lcf = cute.make_rmem_tensor(nL, acc_dtype)
                    for n in cutlass.range_constexpr(nL):
                        a0 = sLC[n, 0, lch.index]
                        for w in cutlass.range_constexpr(
                            1, warpgroup_warps, 1
                        ):
                            a0 += sLC[n, w, lch.index]
                        lcf[n] = a0
                    lch.release()
                    colsum_acc_v.store(
                        corr_v_r.load() * colsum_acc_v.load()
                        + lcf.load()
                    )
                    for n in cutlass.range_constexpr(blk_tile_n):
                        rAccZ[n] = (
                            corr_v_r[n] * rAccZ[n]
                            + z_dp * lcf[blk_tile_n + n]
                        )
                    for pp in cutlass.range_constexpr(
                        1, v_parts, 1
                    ):
                        o_handle_p = o_consumer.wait_and_advance()
                        tOrO_pp = cute.make_rmem_tensor(
                            (*tOsO_w.shape[:1], tiles_dm),
                            acc_dtype,
                        )
                        for dm in cutlass.range_constexpr(tiles_dm):
                            cute.copy(
                                tmem_load_atom_o,
                                tOtO[None, dm, o_handle_p.index],
                                tOrO_pp[None, dm],
                            )
                        s_dp2 = (
                            rSVD_pf[pp][0] * Float32(vch_s_mul)
                        )
                        z_dp2 = (
                            rZVD_pf[pp][0]
                            - Float32(vch_beta) * rSVD_pf[pp][0]
                        )
                        rAcc.store(
                            rAcc.load()
                            + s_dp2
                            * tOrO_pp.load().reshape(
                                (nL, tiles_dm)
                            )
                        )
                        cute.arch.fence_view_async_tmem_load()
                        o_handle_p.release()
                        lch2 = lc_consumer.wait_and_advance()
                        lcf2 = cute.make_rmem_tensor(nL, acc_dtype)
                        for n in cutlass.range_constexpr(nL):
                            a2 = sLC[n, 0, lch2.index]
                            for w in cutlass.range_constexpr(
                                1, warpgroup_warps, 1
                            ):
                                a2 += sLC[n, w, lch2.index]
                            lcf2[n] = a2
                        lch2.release()
                        colsum_acc_v.store(
                            colsum_acc_v.load() + lcf2.load()
                        )
                        for n in cutlass.range_constexpr(
                            blk_tile_n
                        ):
                            rAccZ[n] = (
                                rAccZ[n]
                                + z_dp2 * lcf2[blk_tile_n + n]
                            )
                    # PREFETCH tile s+1's drain scales (consumed
                    # by the next iteration, or by the tail's first
                    # tile — a full tile of latency budget for the
                    # page-table + scale LDGs). NOTE: this sits
                    # inside `if keep_tile` — if BLASST skipping is
                    # ever enabled for flex configs (currently
                    # ctor-asserted off), a skipped tile would also
                    # skip the NEXT tile's prefetch and a later kept
                    # tile would read stale scales.
                    # The loads need no page guard. This loop stops o_stages tiles
                    # before the split's last tile, and the constructor admits only
                    # o_stages 2 and 3, so tile s + 1 is never the split's last tile and
                    # therefore never the sequence's last, partial one: every part of it
                    # starts on a page inside the sequence. A guarded load would reach
                    # its slot through a copy at the branch join, and that copy waits
                    # for the load right here instead of at the next tile's drain.
                    for pp in cutlass.range_constexpr(v_parts):
                        lp_pf = (
                            (s + 1) * kv_splits + kv_split_idx
                        ) * pages_s + pp * v_pages
                        phys_pf = gPT_c[lp_pf]
                        rSVD_pf[pp][0] = Float32(
                            mSC_flat[
                                v_ch_off + warpgroup_tidx,
                                (coord_hk, phys_pf),
                            ]
                        )
                        rZVD_pf[pp][0] = Float32(
                            mSC_flat[
                                v_ch_off + self.head_dim
                                + warpgroup_tidx,
                                (coord_hk, phys_pf),
                            ]
                        )
                else:
                    colsum_s = colsum_load(softmax_phase)
                    colsum_s *= correction
                    for p in cutlass.range_constexpr(o_stages):
                        if softmax_phase == p:
                            colsum_p[None, p].store(
                                correction * colsum_p[None, p].load()
                                + colsum_s
                            )

                # Next softmax producer phase
                if cutlass.const_expr(v_channel):
                    # o-slots advance per PART over the VIRTUAL
                    # ring (depth o_stages * v_parts)
                    softmax_phase += v_parts
                    if softmax_phase >= o_ring:
                        softmax_phase -= o_ring
                elif cutlass.const_expr(o_stages == 2):
                    softmax_phase ^= 1
                else:
                    softmax_phase += 1
                    if softmax_phase == o_stages:
                        softmax_phase = 0

        # The softmax's copy of the exponent (`lp_tail` in `kernel`), written before
        # its first column-maximum meeting, which this warpgroup has passed.
        if cutlass.const_expr(lp_cell is not None and not k_channel):
            pscale_lp = lp_cell[0]
        # Notify for final colmax if we didn't already
        if not keep_tile or iters_s <= o_stages:
            sM_final_nbar.arrive()

        # Handle if we never saw any in-bounds values
        if cutlass.const_expr(not tma_mask and do_atomic_red):
            if sM_hist[o_stages - 1] == -Float32.inf:
                sM_hist[o_stages - 1] = Float32(0)

        # Tail correction weights (fold: nL wide, exactly as in the
        # loop): ring read j merges tile iters_s-o+j (tL/O slot
        # (tail_phase+j) % o) at weight exp2(hist[j] - hist[o-1]);
        # weight o-1 == 1 and is not materialized. Empty slots
        # (iters_s < o_stages) carry zeros and a finite don't-care
        # weight thanks to the preamble's history padding.
        corr_tail_r = cute.make_rmem_tensor((nL, o_stages - 1), acc_dtype)
        for j in cutlass.range_constexpr(o_stages - 1):
            w_lane = exp2(sM_hist[j] - sM_hist[o_stages - 1])
            for n in cutlass.range_constexpr(nL):
                corr_tail_r[n, j] = cute.arch.shuffle_sync(
                    w_lane, n % blk_tile_n
                )
        correction = corr_tail_r[None, 0].load()
        correction_om = correction

        # Correct and accumulate final colsum (+ fold: reduce c_h)
        if cutlass.const_expr(fold_p):
            rC = cute.make_rmem_tensor(blk_tile_n, acc_dtype)
        tail_phase = softmax_phase if enable_blasst else iters_s % o_stages
        zeros_nL = cute.make_rmem_tensor(nL, Float32)
        zeros_nL.fill(Float32(0))
        colsum_final = zeros_nL.load()
        # Read every colsum_p slot HERE, unconditionally, and let the tail branches
        # below only select among these values. nvidia-cutlass-dsl 4.6.1 and 4.8.0
        # miscompile a colsum_p slot read placed inside the `== phase` branches: they
        # sink the add out of the branches and give every branch ONE slot's registers,
        # so the tail merges the wrong partial sums whenever the main loop ran
        # (iters_s > o_stages); with one main-loop tile, the slot it accumulated is
        # replaced by an untouched one. Token-wise V reads rel_l2 ~8 (the PTX shows
        # one `add.f32x2` operand shared by all three ring branches).
        # 4.5.2 computes the same values either way.
        colsum_p_slots = []
        if cutlass.const_expr(not v_channel):
            for p in cutlass.range_constexpr(o_stages):
                colsum_p_slots.append(colsum_p[None, p].load())
        if cutlass.const_expr(v_channel):
            # channel-V: the tail drains virtual tiles (real tile
            # x part) SEQUENTIALLY below — no ring prefetch here.
            pass
        elif cutlass.const_expr(o_stages == 2):
            for phase in cutlass.range_constexpr(o_stages):
                if tail_phase == phase:
                    # Accumulate in thread RF
                    colsum_prev = colsum_load(phase)
                    colsum_final = colsum_load(phase ^ 1)
                    colsum_prev += colsum_p_slots[phase]
                    colsum_final += colsum_p_slots[phase ^ 1]
                    colsum_final += correction * colsum_prev
        else:
            # Ring reads in tile order from tail_phase, then an
            # oldest-first weighted merge (weight o-1 == 1).
            tail_terms = cute.make_rmem_tensor((nL, o_stages), acc_dtype)
            ring = tail_phase
            for j in cutlass.range_constexpr(o_stages):
                for p in cutlass.range_constexpr(o_stages):
                    if ring == p:
                        tail_terms[None, j].store(
                            colsum_load(p) + colsum_p_slots[p]
                        )
                ring += 1
                if ring == o_stages:
                    ring = 0
            colsum_final = tail_terms[None, o_stages - 1].load()
            for j_ in cutlass.range_constexpr(o_stages - 1):
                j = o_stages - 2 - j_
                colsum_final += (
                    corr_tail_r[None, j].load()
                    * tail_terms[None, j].load()
                )
        if cutlass.const_expr(not v_channel):
            # Reduce colsum in warp RF
            rL_lane = Float32(0.0)
            for n in cutlass.range_constexpr(blk_tile_n):
                rL_n = cute.arch.warp_reduction_sum(colsum_final[n])
                if n == lane_idx:
                    rL_lane = rL_n
            # Store partial colsum in smem and notify
            if lane_store_max:
                sL[lane_idx, warpgroup_widx] = rL_lane
            if cutlass.const_expr(fold_p):
                # c_h cross-warp reduction, ORDERED. This is a
                # determinism invariant, not a style choice: relaxed
                # f32 atomic adds here made outputs
                # launch-nondeterministic. Per-warp REDUX sums of the
                # [g, 2g) columns are stored into this warp's sC column
                # (plain store, the sL pattern above), then, after the
                # sL_final wait has ordered all 4 warps' stores, EVERY
                # thread sums the 4 slots in fixed warp order. An
                # identical sequence everywhere gives an identical f32
                # result, so no broadcast step is needed.
                c_lane = Float32(0.0)
                for n in cutlass.range_constexpr(blk_tile_n):
                    c_n = cute.arch.warp_reduction_sum(
                        colsum_final[blk_tile_n + n]
                    )
                    if n == lane_idx:
                        c_lane = c_n
                if lane_store_max:
                    sC[lane_idx, warpgroup_widx] = c_lane
            # Wait to ensure reduction warp has reset sM_final_nbar
            sL_final_nbar.arrive_and_wait()
            if cutlass.const_expr(fold_p):
                for n in cutlass.range_constexpr(blk_tile_n):
                    c_sum = sC[n, 0]
                    for w in cutlass.range_constexpr(
                        1, warpgroup_warps, 1
                    ):
                        c_sum += sC[n, w]
                    rC[n] = c_sum

        # Load O of the last o_stages tiles (ring order from tail_phase;
        # the s-th wait is gated by the s-th remaining VP commit)
        tOrO_prev = tOrO_final = None
        if cutlass.const_expr(not v_channel):
            tOrO_tail = cute.make_rmem_tensor(
                (*tOsO_w.shape, o_stages), acc_dtype
            )
            ring = tail_phase
            for s in cutlass.range_constexpr(o_stages):
                o_handle = o_consumer.wait_and_advance()
                tOtO_s = tOtO[None, None, ring]
                tOrO_s = tOrO_tail[None, None, s]
                cute.copy(tmem_load_atom_o, tOtO_s, tOrO_s)
                cute.arch.fence_view_async_tmem_load()
                o_handle.release()  # Final release signals tmem dealloc
                ring += 1
                if ring == o_stages:
                    ring = 0
            tOrO_prev = tOrO_tail[None, None, 0].load()
            tOrO_final = tOrO_tail[None, None, o_stages - 1].load()

        # Correct and accumulate output
        if cutlass.const_expr(v_channel):
            # channel-V tail: drain the remaining virtual tiles
            # (real tile x part) SEQUENTIALLY — each wait pairs
            # with one remaining VP part-commit; tiles beyond the
            # sequence (iters_s < o_stages) have no commits and
            # are skipped whole (dynamic guard, WG-uniform). The
            # per-tile rebase weight comes from the sM history
            # (hist[j] = M[iters_s - o + j], padded entries repeat
            # M[0] so their weight is exactly 1); the last o
            # release signals the tmem dealloc.
            inv_lo = Float32(1.0 / (1 << self.LO_SHIFT))
            inv_ps = exp2(-pscale_lp)
            n_tail_tiles = cutlass.min(Int32(o_stages), iters_s)
            # Every tail tile's drain scales are in registers before the first drain.
            # The first tail tile's are already in the prefetch slots, because the
            # last loop iteration (or, with no loop iteration, the prologue) loaded
            # them. The later tail tiles' are loaded here, all at once, so the tail
            # pays one page-table and scale round trip instead of one per part. A
            # part past the last page loads nothing and keeps zeros, as before.
            pl_first = 1 if not enable_blasst else 0
            rSVT = [
                [cute.make_rmem_tensor(1, acc_dtype) for _ in range(v_parts)]
                for _ in range(o_stages)
            ]
            rZVT = [
                [cute.make_rmem_tensor(1, acc_dtype) for _ in range(v_parts)]
                for _ in range(o_stages)
            ]
            for jj in cutlass.range_constexpr(pl_first, o_stages, 1):
                for pp in cutlass.range_constexpr(v_parts):
                    rSVT[jj][pp].fill(Float32(0))
                    rZVT[jj][pp].fill(Float32(0))
                    tile_pl = iters_s - n_tail_tiles + jj
                    lp_pl = (
                        tile_pl * kv_splits + kv_split_idx
                    ) * pages_s + pp * v_pages
                    if lp_pl < page_count:
                        phys_pl = gPT_c[lp_pl]
                        rSVT[jj][pp][0] = Float32(
                            mSC_flat[
                                v_ch_off + warpgroup_tidx,
                                (coord_hk, phys_pl),
                            ]
                        )
                        rZVT[jj][pp][0] = Float32(
                            mSC_flat[
                                v_ch_off + self.head_dim
                                + warpgroup_tidx,
                                (coord_hk, phys_pl),
                            ]
                        )
            for v in cutlass.range(
                n_tail_tiles * v_parts, unroll=1
            ):
                jt = v // v_parts       # tail tile ordinal
                ppt = v - jt * v_parts  # part within the tile
                j_hist = jt + (Int32(o_stages) - n_tail_tiles)
                # rebase weight: part 0 of a tile rebases from
                # the previous drained frame; other parts share
                # the frame (weight exactly 1). Padded history
                # entries repeat M[0], so short sequences get
                # unit weights automatically.
                w_lane = Float32(1.0)
                if ppt == 0:
                    for jj in cutlass.range_constexpr(o_stages):
                        if j_hist == jj:
                            if cutlass.const_expr(jj == 0):
                                w_lane = exp2(
                                    m_prev_v - sM_hist[0]
                                )
                            else:
                                w_lane = exp2(
                                    sM_hist[jj - 1] - sM_hist[jj]
                                )
                corr_t = cute.make_rmem_tensor(nL, acc_dtype)
                for n in cutlass.range_constexpr(nL):
                    corr_t[n] = cute.arch.shuffle_sync(
                        w_lane, n % blk_tile_n
                    )
                oh_t = o_consumer.wait_and_advance()
                tOrO_t = cute.make_rmem_tensor(
                    (*tOsO_w.shape[:1], tiles_dm), acc_dtype
                )
                for dm in cutlass.range_constexpr(tiles_dm):
                    cute.copy(
                        tmem_load_atom_o,
                        tOtO[None, dm, oh_t.index],
                        tOrO_t[None, dm],
                    )
                # this virtual tile's raw scales, selected out of the
                # registers loaded before the loop (zeros past the sequence)
                s_raw_t = Float32(0)
                z_raw_t = Float32(0)
                for jj in cutlass.range_constexpr(o_stages):
                    for pp in cutlass.range_constexpr(v_parts):
                        if (jt == jj) & (ppt == pp):
                            if cutlass.const_expr(jj < pl_first):
                                s_raw_t = rSVD_pf[pp][0]
                                z_raw_t = rZVD_pf[pp][0]
                            else:
                                s_raw_t = rSVT[jj][pp][0]
                                z_raw_t = rZVT[jj][pp][0]
                s_dp_t = s_raw_t * Float32(vch_s_mul)
                z_dp_t = (
                    z_raw_t - Float32(vch_beta) * s_raw_t
                )
                rAcc.store(
                    corr_t.load() * rAcc.load()
                    + s_dp_t
                    * tOrO_t.load().reshape((nL, tiles_dm))
                )
                cute.arch.fence_view_async_tmem_load()
                oh_t.release()
                lch_t = lc_consumer.wait_and_advance()
                lcf_t = cute.make_rmem_tensor(nL, acc_dtype)
                for n in cutlass.range_constexpr(nL):
                    a_t = sLC[n, 0, lch_t.index]
                    for w in cutlass.range_constexpr(
                        1, warpgroup_warps, 1
                    ):
                        a_t += sLC[n, w, lch_t.index]
                    lcf_t[n] = a_t
                lch_t.release()
                colsum_acc_v.store(
                    corr_t.load() * colsum_acc_v.load()
                    + lcf_t.load()
                )
                for n in cutlass.range_constexpr(blk_tile_n):
                    rAccZ[n] = (
                        corr_t[n] * rAccZ[n]
                        + z_dp_t * lcf_t[blk_tile_n + n]
                    )
            # final colsum -> sL, then the reduction-warp
            # handshake. colsum_acc_v already holds FULL column
            # sums (the sLC entries are pre-reduced per warp), so
            # store into warp column 0 only — columns 1..3 keep
            # their per-item zero-init and the reduction warp's
            # fixed 4-column sum stays exact.
            for n in cutlass.range_constexpr(blk_tile_n):
                if n == lane_idx:
                    if warpgroup_widx == 0:
                        if lane_store_max:
                            sL[lane_idx, 0] = colsum_acc_v[n]
            sL_final_nbar.arrive_and_wait()
            # epilogue: O = (hi + 2^-LO_SHIFT*lo + z-term) / pscale
            rOutF = cute.make_rmem_tensor(
                (blk_tile_n, tiles_dm), acc_dtype
            )
            for n in cutlass.range_constexpr(blk_tile_n):
                for dm in cutlass.range_constexpr(tiles_dm):
                    rOutF[n, dm] = (
                        rAcc[n, dm]
                        + inv_lo * rAcc[blk_tile_n + n, dm]
                        + rAccZ[n]
                    ) * inv_ps
            output_final = rOutF.load()
        elif cutlass.const_expr(fold_p):
            # V-fold epilogue:
            #   O_h = (O_hi[h] + 2^-LO_SHIFT * O_lo[h]) / pscale + c_h
            # (pscale = 2^lp, the pool's one exponent, which every P tile of
            # the item was quantized with).
            rFin = cute.make_rmem_tensor((nL, tiles_dm), acc_dtype)
            rFin.store(
                tOrO_final.reshape((nL, tiles_dm))
                + correction_om * tOrO_prev.reshape((nL, tiles_dm))
            )
            for j in cutlass.range_constexpr(1, o_stages - 1, 1):
                rFin.store(
                    rFin.load()
                    + corr_tail_r[None, j].load()
                    * tOrO_tail[None, None, j].load().reshape(
                        (nL, tiles_dm)
                    )
                )
            inv_lo = Float32(1.0 / (1 << self.LO_SHIFT))
            if cutlass.const_expr(lp_cell is not None and k_channel):
                inv_ps = exp2(-lp_cell[0])
            else:
                inv_ps = exp2(-pscale_lp)
            rOutF = cute.make_rmem_tensor((blk_tile_n, tiles_dm), acc_dtype)
            for n in cutlass.range_constexpr(blk_tile_n):
                for dm in cutlass.range_constexpr(tiles_dm):
                    rOutF[n, dm] = (
                        rFin[n, dm]
                        + inv_lo * rFin[blk_tile_n + n, dm]
                    ) * inv_ps + rC[n]
            output_final = rOutF.load()
        else:
            output_prev = tOrO_prev.reshape((blk_tile_n, tiles_dm))
            output_final = tOrO_final.reshape((blk_tile_n, tiles_dm))
            output_final += correction * output_prev
            for j in cutlass.range_constexpr(1, o_stages - 1, 1):
                output_final += (
                    corr_tail_r[None, j].load()
                    * tOrO_tail[None, None, j].load().reshape(
                        (blk_tile_n, tiles_dm)
                    )
                )

        # Apply final normalization
        if cutlass.const_expr(do_atomic_red or do_none_red):
            # final normalization stored in sM
            sM_final_nbar.arrive_and_wait()
            normalization = sM.load()
            output_final *= normalization

        if cutlass.const_expr(o_gmem is not None):
            # `o_direct_for`: each thread writes the O elements it holds straight to global
            # memory. The global tile and its coordinate tensor are partitioned exactly as
            # sO is, so element e of this thread's fragment is O[d, (h, p)] with
            # (d, (h, p)) = tOcO[e], and an element outside the valid heads or tokens is
            # not written. The tile is head_dim wide (`blk_tile_d`), so every d is valid.
            tCgO_d, tCcO_d, o_hp_lim = o_gmem
            if cutlass.const_expr(fold_p):
                thr_o_st = tmem_load_o_n.get_slice(warpgroup_tidx)
            else:
                thr_o_st = thr_load_o
            tOgO = thr_o_st.partition_D(tCgO_d)[None, 0, 0, 0, None]
            tOcO = thr_o_st.partition_D(tCcO_d)[None, 0, 0, 0, None]
            rOut = cute.make_rmem_tensor(tOgO.shape, o_dtype)
            rOut.store(output_final.to(o_dtype).reshape(tOgO.shape))
            for e in cutlass.range_constexpr(cute.size(tOgO)):
                _, o_hp = tOcO[e]
                if cute.elem_less(o_hp, o_hp_lim):
                    tOgO[e] = rOut[e]
        else:
            # Store O to smem and notify
            tOsO.store(output_final.to(o_dtype).reshape(tOsO.shape))
            cute.arch.fence_view_async_shared()
            sO_final_nbar.arrive()

        # Debug
        if cutlass.const_expr(enable_blasst and debug_blasst):
            batches = mQ.shape[-1][1]
            # Append skip count to page counts
            tiles_skipped_ptr = seqlens_iter + batches
            tiles_kept = o_consumer.current_handle().count if iters_s > 1 else 1
            tiles_skipped = iters_s - tiles_kept
            if warpgroup_tidx == 0:
                gmem_add(tiles_skipped_ptr, Int32(tiles_skipped))


    @cute.jit
    def _softmax_warpgroup(
        self,
        acc_dtype,
        blk_tile_d,
        blk_tile_h,
        blk_tile_n,
        blk_tile_p,
        blk_tile_s,
        coord_b,
        coord_hg,
        coord_hk,
        coord_p,
        dual_s,
        enable_blasst,
        flex_ring_w,
        fold_k,
        fold_p,
        grouped_heads,
        iters_s,
        k_ch64,
        k_channel,
        k_is_token,
        k_parts,
        k_ring_w,
        k_tok64,
        kv_split_idx,
        kv_splits,
        lane_idx,
        lane_store_max,
        lc_producer,
        log2_threshold_scale_factor,
        mDbgS,
        mQD,
        mQraw,
        mma_dtype,
        mma_tile_m,
        mma_tile_n,
        o_consumer,
        o_dtype,
        p_producer,
        pages_m,
        pages_s,
        prediction,
        pscale_lp,
        lp_table,
        lp_shift,
        lp_cell,
        qb_consumer,
        qflex,
        sC,
        sL,
        sLC,
        sL_final_nbar,
        sM,
        sM_consumer_nbar,
        sM_final_nbar,
        sM_producer_nbar,
        sO_final_nbar,
        sQB,
        sSP_i32,
        sSP_i8,
        sSP_producer_nbar,
        sScale,
        s_consumer,
        scale_consumer,
        scale_ring_elems,
        scale_s_log2_e,
        seqlen,
        sp_producer,
        tCsO,
        tCsP,
        tCtL,
        tCtO,
        tCtS,
        tCtS2,
        tL_consumer_nbar,
        tL_producer_nbar,
        tiles_dm,
        tiles_s,
        tiles_sm,
        v_channel,
        v_is_token,
        v_parts,
        v_ring_off,
        v_ring_w,
        warpgroup_tidx,
        warpgroup_widx,
        with_phase,
    ):
        """The softmax warpgroup: the fold on S, the online softmax, and the P store.

        The largest role, and the one that owns the fold algebra on both sides of the
        exponential.

        Per tile, in order: acquire this tile's scale-ring stages and expand the per-token
        fold coefficients BEFORE the S wait, so the coefficient LDS overlaps waiting for
        the S MMA rather than extending the post-S critical path; load S from tmem; apply
        the K-fold, `S = s_t*S_raw + (z_t - BETA_FOLD*s_t)*QSum_h`, per element before the
        max-reduce; mask; reduce the column max; exp2; quantize P into the hi and lo e4m3
        row halves with the launch-global pscale and compute `c_h` from the quantized rows
        while they are still in registers; store P and accumulate the true-f32 colsum.

        Thread ownership, which is what makes the per-token coefficients scalars: the
        Ld32x32b S partition gives lane `warpgroup_tidx` the S ROW (= K token)
        `warpgroup_tidx` of every 128-token mma tile, across all `blk_tile_n` columns.
        """
        # Read straight off `self` instead of taking 21 parameters that were
        # only ever `X = self.X` in the caller. Same expression, same object,
        # and the body below is untouched -- these are the names it uses.
        bits = self.bits
        debug_s = self.debug_s
        do_atomic_red = self.do_atomic_red
        f4_feed = self.dequant_int2_to_f4
        o_stages = self.o_stages
        page_size = self.page_size
        scale_stages = self.scale_stages
        tma_mask = self.tma_mask

        # The tL ring slot this warpgroup writes next. It advances per tile
        # (per part under channel-V) at the bottom of the sequence loop.
        softmax_phase = 0
        # `not v_channel` is load-bearing: under channel-V the
        # [colsum | l] transport moves off the tL tmem/named-barrier
        # channel onto the sLC smem ring, so `colsum_load` NEVER runs
        # (its channel-V arm raises). These are COMPENSATION arrives for
        # a tail read that therefore does not exist, and a stray arrival
        # on a live named-barrier id makes a LATER wait release early --
        # the exact failure class under investigation. Unreachable at the
        # shipped iters_s (the guards need iters_s <= 2), which is why it
        # has never fired, but wrong at short sequences.
        #
        # PERSISTENT ITEM CLOSURE. The correction tail also RELEASES every tL
        # slot it reads, so on a slot no tile writes it arrives on that phase's
        # tL consumer barrier with no softmax acquire to match. A one-item CTA
        # never notices, but a persistent CTA carries each barrier's count into
        # its next item, whose protocol assumes exactly the one pending release
        # per phase that a full item leaves: after an item of one tile the
        # unmatched release completes a generation on its own, so the next item's
        # first acquire of that phase waits for a release that only follows its
        # own write, and the CTA deadlocks (a ragged batch with a request of at
        # most two tiles sharing a CTA with a longer one).
        # The softmax therefore also arrives on the consumer barrier of every
        # phase it compensates, which is the acquire it would have made, so
        # every item, whatever its length, ends with the state it began with.
        if cutlass.const_expr(not v_channel):
            if cutlass.const_expr(o_stages == 2):
                if iters_s == 1:
                    with_phase(tL_producer_nbar, 1).arrive()
                    if cutlass.const_expr(self.persistent):
                        with_phase(tL_consumer_nbar, 1).arrive()
            else:
                # o_stages == 3: compensate the
                # tail's colsum_load of every tL slot no tile will
                # produce (correction zero-inits those slots' payloads).
                for p in cutlass.range_constexpr(1, o_stages, 1):
                    if iters_s <= p:
                        with_phase(tL_producer_nbar, p).arrive()
                        if cutlass.const_expr(self.persistent):
                            with_phase(tL_consumer_nbar, p).arrive()

        # Construct copy atom for S
        tmem_repeat_op_s = blk_tile_n
        if cutlass.const_expr(mma_tile_n == blk_tile_n and tiles_sm in (2, 4)):
            tmem_repeat_op_s *= tiles_sm
        tmem_repeat_op_s = tcgen05.Repetition(tmem_repeat_op_s)
        tmem_load_op_s = tcgen05.Ld32x32bOp(tmem_repeat_op_s)
        tmem_load_atom_s = cute.make_copy_atom(tmem_load_op_s, acc_dtype)
        # Tile atom and slice
        tCtS_stage = tCtS[mma_dice + (None, 0)]
        tmem_load_s = tcgen05.make_tmem_copy(tmem_load_atom_s, tCtS_stage)
        thr_load_s = tmem_load_s.get_slice(warpgroup_tidx)
        # Partition S and P
        # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #CPY_SM, stages)
        tStS = thr_load_s.partition_S(tCtS)
        tSsP = thr_load_s.partition_D(tCsP)
        # Slice unused modes
        tStS = tStS[None, 0, 0, 0, None, None]  # (CPY, #CPY_SM, s_stages)
        if cutlass.const_expr(dual_s):
            tStS2 = thr_load_s.partition_S(tCtS2)
            tStS2 = tStS2[None, 0, 0, 0, None, None]
        tSsP = tSsP[None, 0, 0, 0, None, None]  # (CPY, #CPY_SM, p_stages)

        # Construct copy atom for L. Fold mode widens the channel to
        # 2*blk_tile_n columns: [0, g) = colsum (true f32 P sums, exactly
        # the baseline), [g, 2g) = the per-thread c_h partials (the reference's
        # consistent quantized-P form — see the P store below) riding the
        # same tmem strip / nbar protocol.
        nL = blk_tile_n * (2 if fold_p else 1)
        if cutlass.const_expr(not v_channel):
            tmem_repeat_op_l = tcgen05.Repetition(nL)
            tmem_load_op_l = tcgen05.Ld32x32bOp(tmem_repeat_op_l)
            tmem_store_op_l = tcgen05.St32x32bOp(tmem_repeat_op_l)
            tmem_load_atom_l = cute.make_copy_atom(tmem_load_op_l, acc_dtype)
            tmem_store_atom_l = cute.make_copy_atom(
                tmem_store_op_l, acc_dtype
            )
            # Tile atom and slice
            tCtL_phase = tCtL[mma_dice + (softmax_phase,)]
            tmem_load_l = tcgen05.make_tmem_copy(tmem_load_atom_l, tCtL_phase)
            thr_load_l = tmem_load_l.get_slice(warpgroup_tidx)
            # Partition L
            # (CPY, #CPY_MMA, #CPY_M, #CPY_N, stages)
            tStL = thr_load_l.partition_S(tCtL)
            # Slice unused modes
            # (CPY, stages)
            tStL = tStL[None, 0, 0, 0, None]
            tSrL_shape = thr_load_l.partition_D(tCtL_phase).shape[:1]

        ##############################
        # fp8-fold state
        ##############################
        # Thread ownership: the Ld32x32b S partition gives lane
        # warpgroup_tidx the S ROW (= K token) warpgroup_tidx of every
        # 128-token mma tile, all blk_tile_n columns — so all per-token
        # fold coefficients are per-(thread, sm) scalars.
        lp_words = None
        if cutlass.const_expr(fold_k):
            tok_row = warpgroup_tidx % page_size  # row within scale block
            tok_half = warpgroup_tidx // page_size  # page within mma tile
            beta_fold = Float32(self.BETA_FOLD[bits])
            lo_scale = Float32(1 << self.LO_SHIFT)
            assert scale_stages >= pages_s  # softmax holds a full tile

            # QSum_h = sum_d Q_e4m3[h, d] (f32 accumulate over the raw
            # e4m3 values the MMA eats; the per-tensor 1/qs_Q descale is
            # folded into scale_s host-side — __call__ docstring).
            # Computed redundantly by every thread (one-time cost,
            # L1-broadcast gmem reads) instead of a cross-warp reduction:
            # zero extra barriers.
            cute.arch.griddepcontrol_wait()  # PDL: gmem reads below
            # A one-item CTA's exponent (`lp_table`, see the kernel prologue): its
            # table loads issue here, so that their latency overlaps the QSum loads'.
            if cutlass.const_expr(lp_table is not None):
                lp_words = self._vmax_hist_loads(lp_table, lane_idx, self.vmax_hist_vec)
            rQS = cute.make_rmem_tensor(blk_tile_n, acc_dtype)
            rQS2 = None
            if cutlass.const_expr(k_tok64):
                # K token-64: per-D-half sums (rQS = lo, rQS2 = hi)
                rQS2 = cute.make_rmem_tensor(blk_tile_n, acc_dtype)
            # Every row's loads are issued before any row is summed, each from an
            # in-bounds (clamped) row; an out-of-bounds row's sum is then replaced by 0
            # with a select (not a multiply, so a NaN or a negative value in the clamped
            # row cannot leak as NaN or -0.0). A per-row `if in_bounds:` around load +
            # sum would serialize the rows: when the compiler keeps those branches, each
            # row's loads wait for the previous row's sum, one global round trip per row
            # at every work item's start.
            rQ8_rows = []
            ok_rows = []
            for n in cutlass.range_constexpr(blk_tile_n):
                hg_g = coord_hg * blk_tile_h + (n % blk_tile_h)
                p_g = coord_p * blk_tile_p + (n // blk_tile_h)
                # Guard OOB packed-head rows (npo2-padded head tiles):
                # TMA zero-fills their Q smem rows, so their S columns
                # are 0; QSum = 0 keeps the fold at 0 there too.
                ok_rows.append((hg_g < grouped_heads) & (p_g < prediction))
                rQ8_rows.append(mQraw[
                    ((cutlass.min(hg_g, grouped_heads - 1),
                      cutlass.min(p_g, prediction - 1)),
                     None, (coord_hk, coord_b))
                ].load())
            # The exponent here, between the issue of the Q rows' loads and their sums,
            # so that the table words are dead before the sums' registers are live.
            if cutlass.const_expr(lp_table is not None):
                pscale_lp = self._pool_pscale_lp(lp_words, lane_idx, lp_shift,
                                                 self.vmax_hist_vec)
                # Published for the correction warpgroup and for this warpgroup's
                # own per-tile reads where the build has the cell (`lp_cell`, see
                # the kernel prologue). Thread 0 writes it before its first
                # column-maximum meeting, a barrier every reader passes before its
                # first read.
                if cutlass.const_expr(lp_cell is not None):
                    if warpgroup_tidx == 0:
                        lp_cell[0] = pscale_lp
            for n in cutlass.range_constexpr(blk_tile_n):
                qsum_n = Float32(0)
                qsum_hi = Float32(0)
                rQ8 = rQ8_rows[n].to(acc_dtype)
                for dq in cutlass.range_constexpr(blk_tile_d):
                    if cutlass.const_expr(
                        k_tok64 and dq >= blk_tile_d // 2
                    ):
                        qsum_hi += rQ8[dq]
                    else:
                        qsum_n += rQ8[dq]
                rQS[n] = Float32(cutlass.select_(ok_rows[n], qsum_n, Float32(0)))
                if cutlass.const_expr(k_tok64):
                    rQS2[n] = Float32(
                        cutlass.select_(ok_rows[n], qsum_hi, Float32(0)))

            # debug_s: single dumping CTA
            is_dbg_cta = False
            if cutlass.const_expr(debug_s):
                is_dbg_cta = (
                    (kv_split_idx == 0)
                    & (coord_hk == 0)
                    & (coord_b == 0)
                    & (coord_hg == 0)
                    & (coord_p == 0)
                )

        # PER-ROW QUERY DESCALE (`q_descale`). The descale d_h multiplies the
        # whole folded S of row h, so it joins scale_s at the exp2 as one
        # per-row factor, read once per item: `rQD` holds it per (row, M-tile)
        # for the exp2's elementwise product, and `qd_lane` is this lane's own
        # row's factor for the column maximum (selected by `n == lane_idx`, the
        # same idiom the max uses, because a dynamically indexed register array
        # spills). The maximum is taken before the factor is applied, which is
        # why d_h must be positive. Without `q_descale` the kernel keeps the
        # single scalar scale_s_log2_e, unchanged.
        rQD = None
        qd_lane = scale_s_log2_e
        if cutlass.const_expr(mQD is not None):
            cute.arch.griddepcontrol_wait()  # PDL: the descale may be produced upstream
            rQD = cute.make_rmem_tensor((blk_tile_n, tiles_sm), acc_dtype)
            for n in cutlass.range_constexpr(blk_tile_n):
                hg_d = coord_hg * blk_tile_h + (n % blk_tile_h)
                d_n = Float32(1.0)
                # Packed-head rows past the group are padding (their S
                # columns are 0 and never stored); they keep factor 1.
                if hg_d < grouped_heads:
                    d_n = mQD[coord_b, coord_hk * grouped_heads + hg_d]
                for smi in cutlass.range_constexpr(tiles_sm):
                    rQD[n, smi] = scale_s_log2_e * d_n
                if n == lane_idx:
                    qd_lane = scale_s_log2_e * d_n

        # Causal masking for spec decode verification
        masked_iters_s = masked_start_s = masked_coord_s = 0
        check_safe_max = False
        if cutlass.const_expr(not tma_mask):
            is_last_split = kv_split_idx == (tiles_s - 1) % kv_splits
            is_prev_split = kv_split_idx == (tiles_s - 2) % kv_splits
            # 2 masked tiles
            if 0 < seqlen % blk_tile_s < prediction:
                # 1 split masks 2 tiles
                if kv_splits == 1:
                    masked_start_s = 0
                    masked_iters_s = 2
                    masked_coord_s = tiles_s - 2
                # 2 splits mask 1 tile each
                elif is_last_split or is_prev_split:
                    masked_start_s = 0
                    masked_iters_s = 1
                    masked_coord_s = (
                        (tiles_s - 1) if is_last_split else (tiles_s - 2)
                    )
                    # if last split only has 1 tile then some cols will never
                    # see inbounds values, so P = exp(-inf + inf) -> nan
                    check_safe_max = is_last_split and iters_s == 1
            # 1 masked tile
            elif prediction > 1 or seqlen % blk_tile_s != 0:
                # 1 split masks 1 tile
                if is_last_split:
                    masked_start_s = 0
                    masked_iters_s = 1
                    masked_coord_s = tiles_s - 1

        if cutlass.const_expr(enable_blasst):
            # Tile skip tracking
            sM_lane_prev = -Float32.inf
            # Per-batch BLASST threshold, matching trtllm:
            # effective threshold_p = scale_factor / seqlen
            log2_threshold_p = log2_threshold_scale_factor - cute.math.log2(
                Float32(seqlen)
            )

        # Scale-quad prefetch ONE TILE EARLY: the per-tile scale-ring
        # acquire, the (s, z) quad LDS and the early release move one tile
        # ahead of the S-wait, so tile t+1's ring spin overlaps tile t's
        # S-wait slack and its stages are released to the TMA-QK scale gate
        # a full tile sooner. The coefficient expansion (rSK/rZQ/rSV/rZV)
        # stays at the top of the iteration, reading the quad register
        # buffer rSZ2. The consumer order is plain FIFO over tiles
        # 0..iters_s-1 across BOTH sequence loops, so the prefetch pipeline
        # carries across the loop boundary.
        # NOTE: scale_consumer is referenced through this single name only,
        # and every walk is a wait_and_advance with a real release — no
        # state-only advance.
        # This is the only scale path of the fold: an in-loop acquire measures
        # slower at INT4. The staging is feed-independent; only
        # the coefficient algebra is keyed on the width and the feed, and
        # the expansion at the top of the iteration carries every arm
        # (bits == 4, the f4 feed, the e4m3 int2 feed). It runs whenever
        # the scale ring carries per-token elements; channel-K with
        # channel-V has none (scale_ring_elems == 0) and every coefficient
        # comes from the bias ring below.
        scale_prefetch = cutlass.const_expr(
            fold_k and scale_ring_elems > 0
        )
        if cutlass.const_expr(scale_prefetch):
            rSZ2 = cute.make_rmem_tensor(
                (4 if not qflex else flex_ring_w, tiles_sm), acc_dtype
            )
            if iters_s > 0:
                # Prologue fetch: tile 0's quads (same body as the
                # in-loop fetch below — kept textually inline at both
                # sites; a shared closure would hide the participant
                # name from the loop-carry machinery).
                scale_handles = []
                for _pj in cutlass.range_constexpr(pages_s):
                    scale_handles.append(
                        scale_consumer.wait_and_advance()
                    )
                scale_stage0 = scale_handles[0].index
                for smi in cutlass.range_constexpr(tiles_sm):
                    stg_j = (
                        scale_stage0 + smi * pages_m + tok_half
                    ) % scale_stages
                    if cutlass.const_expr(not qflex):
                        sSZ = cute.make_tensor(
                            (
                                sScale.iterator
                                + (tok_row * 4 + stg_j * page_size * 4)
                            ).align(16),
                            cute.make_layout(4),
                        )
                        cute.autovec_copy(sSZ, rSZ2[None, smi])
                    else:
                        # flex ring sections: K pairs first, V pairs
                        # after (each token's slice of its section)
                        if cutlass.const_expr(k_ring_w > 0):
                            sKZ = cute.make_tensor(
                                (
                                    sScale.iterator
                                    + (tok_row * k_ring_w
                                       + stg_j * scale_ring_elems)
                                ).align((k_ring_w * self.scale_dtype.width) // 8),
                                cute.make_layout(k_ring_w),
                            )
                            for e in cutlass.range_constexpr(k_ring_w):
                                rSZ2[e, smi] = Float32(sKZ[e])
                        if cutlass.const_expr(v_ring_w > 0):
                            sVZ = cute.make_tensor(
                                (
                                    sScale.iterator
                                    + (v_ring_off + tok_row * 2
                                       + stg_j * scale_ring_elems)
                                ).align((2 * self.scale_dtype.width) // 8),
                                cute.make_layout(2),
                            )
                            for e in cutlass.range_constexpr(2):
                                rSZ2[k_ring_w + e, smi] = sVZ[e]
                # ASYNC-PROXY ORDERING before the scale release. The quad LDS above
                # is a GENERIC-proxy read and the release is the ordinary mbarrier
                # release/acquire handoff, which lacks the generic->async proxy
                # bridge, so TMA-QK's ASYNC-proxy overwrite of the reacquired stage
                # need not be ordered after these reads and can land first; the
                # reader would then see the next occupant of the stage, one full
                # scale-ring turn ahead. fence.proxy.async is the documented bridge;
                # it is executed by EVERY softmax thread (no elect predicate). The
                # two scale-PREFETCH sites carry the same ordering
                # argument, not that experiment. A proxy fence acts only within its own
                # thread and the release arrives from lane 0 of each warp, so the warp
                # barrier orders all 32 lanes' fences before lane 0's `mbarrier.arrive`
                # (CROSS-PROXY ORDERING).
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                for sh in scale_handles:
                    sh.release()

        # channel-V part-store locals: pre-declared before the dynamic
        # sequence loop (region-conversion rule: names updated inside
        # a dynamic region need a prior value; dead registers on every
        # other config — DCE'd)
        mask = Float32(1.0)
        ph = Int32(0)

        # Sequence loop
        num_loops = 2 if not tma_mask else 1
        for loop in cutlass.range_constexpr(num_loops):
            is_masked_loop = loop == 1
            for s in cutlass.range(
                masked_start_s if is_masked_loop else 0,
                masked_iters_s if is_masked_loop else (iters_s - masked_iters_s),
                1,
            ):
                # K-fold coefficient expansion, BEFORE the S wait. This
                # tile's per-token (s, z) quads are already in registers
                # (rSZ2): the prefetch below (the prologue fetch, for
                # tile 0) waited the tile's pages_s scale stages during
                # the previous tile, copied the quads, fenced and
                # released the stages. The K-fold and the P-quant share
                # that register copy, not a ring stage, so no scale
                # stage is held across the S-wait or the P store.
                # Expanding here overlaps the coefficient prep with
                # waiting for the S MMA instead of extending the post-S
                # critical path.
                if cutlass.const_expr(fold_k):
                    rSK = cute.make_rmem_tensor(
                        (blk_tile_n, tiles_sm), acc_dtype
                    )
                    rZQ = cute.make_rmem_tensor(
                        (blk_tile_n, tiles_sm), acc_dtype
                    )
                    rSK2 = None
                    if cutlass.const_expr(k_tok64):
                        rSK2 = cute.make_rmem_tensor(
                            (blk_tile_n, tiles_sm), acc_dtype
                        )
                    # channel-K coefficient staging: one vector load of
                    # the g lifts and one of the g biases, against
                    # blk_tile_n scalar LDS each before. `sQB`'s
                    # coefficient mode is stride 1 and holds
                    # [lift[0,g) | bias[g,2g)], so dividing it by
                    # blk_tile_n gives a (g, 2, qp_stages) view whose
                    # leading mode is exactly one coefficient vector.
                    qb_d, rQBs, rQBz = None, None, None
                    if cutlass.const_expr(k_channel):
                        qb_d = cute.flat_divide(sQB, (blk_tile_n,))
                        rQBs = cute.make_rmem_tensor(
                            blk_tile_n, acc_dtype
                        )
                        rQBz = cute.make_rmem_tensor(
                            blk_tile_n, acc_dtype
                        )
                    rSV = cute.make_rmem_tensor(tiles_sm, acc_dtype)
                    rZV = cute.make_rmem_tensor(tiles_sm, acc_dtype)
                    if cutlass.const_expr(scale_prefetch):
                        # Coefficients from the quad buffer fetched one
                        # tile ago (prologue, or the previous
                        # iteration). The fold algebra is keyed on the
                        # width and the feed, and every arm is certified:
                        #   bits == 4: the raw-code feed is q * 2**-9,
                        #     exact in e4m3 (the subnormal band and the
                        #     first binade share the 2**-9 ulp), so
                        #     s' = 512*s_t (ALPHA_INV_INT4, an exact
                        #     exponent shift) and z' = z_t
                        #     (BETA_FOLD[4] = 0). s'_v = 512*s_v rides
                        #     inside the P operand, paired with the lp-9
                        #     shift at the decode top so the quantized P
                        #     bytes do not move, and the c_h rcp(s'_v)
                        #     reads the same scaled register.
                        #   f4 feed: q/2, exact in e2m1, so s' = 2*s_t
                        #     and z' = z_t (BETA_FOLD_F4 = 0; e2m1 cannot
                        #     carry the +8 recentering). s'_v = 2*s_v,
                        #     paired with the lp-1 shift, an exact identity
                        #     with the e4m3(8+q)/BETA_FOLD[2]=8 feed.
                        #   e4m3 int2 feed: e4m3(q + BETA_FOLD[2]), so
                        #     s' = s_t and z' = z_t - BETA_FOLD*s_t, with
                        #     the V twin s'_v = s_v and
                        #     z'_v = z_v - BETA_FOLD*s_v.
                        for smi in cutlass.range_constexpr(tiles_sm):
                            if cutlass.const_expr(k_tok64):
                                # K token-64: (s_lo, z_lo, s_hi, z_hi)
                                # per token, one pair per D half.
                                if cutlass.const_expr(bits == 4):
                                    s_lo = rSZ2[0, smi] * Float32(
                                        self.ALPHA_INV_INT4)
                                    z_lo = rSZ2[1, smi]
                                    s_hi = rSZ2[2, smi] * Float32(
                                        self.ALPHA_INV_INT4)
                                    z_hi = rSZ2[3, smi]
                                elif cutlass.const_expr(f4_feed):
                                    s_lo = rSZ2[0, smi] + rSZ2[0, smi]
                                    z_lo = rSZ2[1, smi]
                                    s_hi = rSZ2[2, smi] + rSZ2[2, smi]
                                    z_hi = rSZ2[3, smi]
                                else:
                                    s_lo = rSZ2[0, smi]
                                    s_hi = rSZ2[2, smi]
                                    z_lo = rSZ2[1, smi] - beta_fold * s_lo
                                    z_hi = rSZ2[3, smi] - beta_fold * s_hi
                                for n in cutlass.range_constexpr(
                                    blk_tile_n
                                ):
                                    rSK[n, smi] = s_lo
                                    rSK2[n, smi] = s_hi
                                    rZQ[n, smi] = (
                                        z_lo * rQS[n]
                                        + z_hi * rQS2[n]
                                    )
                            elif cutlass.const_expr(k_is_token):
                                s_k, z_k = rSZ2[0, smi], rSZ2[1, smi]
                                if cutlass.const_expr(bits == 4):
                                    s_k = s_k * Float32(
                                        self.ALPHA_INV_INT4)
                                    z_kp = z_k
                                elif cutlass.const_expr(f4_feed):
                                    s_k = s_k + s_k
                                    z_kp = z_k
                                else:
                                    z_kp = z_k - beta_fold * s_k
                            if cutlass.const_expr(v_is_token):
                                # The V pair follows the K section of the
                                # quad: slots k_ring_w + e, which are 2
                                # and 3 at the legacy config.
                                s_v = rSZ2[k_ring_w + 0, smi]
                                z_v = rSZ2[k_ring_w + 1, smi]
                                if cutlass.const_expr(bits == 4):
                                    rSV[smi] = s_v * Float32(
                                        self.ALPHA_INV_INT4)
                                    rZV[smi] = z_v
                                elif cutlass.const_expr(f4_feed):
                                    rSV[smi] = s_v + s_v
                                    rZV[smi] = z_v
                                else:
                                    rSV[smi] = s_v
                                    rZV[smi] = z_v - beta_fold * s_v
                            if cutlass.const_expr(
                                k_is_token and not k_tok64
                            ):
                                for n in cutlass.range_constexpr(
                                    blk_tile_n
                                ):
                                    rSK[n, smi] = s_k
                                    rZQ[n, smi] = z_kp * rQS[n]
                        # Fetch tile t+1's quads NOW — ahead of this
                        # tile's S-wait. Global tile ordinal: main loop
                        # t = s; masked loop t = (iters_s -
                        # masked_iters_s) + (s - masked_start_s). The
                        # fetch is skipped past the last tile (the ring
                        # only ever produces iters_s tiles).
                        if cutlass.const_expr(is_masked_loop):
                            has_next_scale = (
                                s - masked_start_s + 1
                            ) < masked_iters_s
                        else:
                            has_next_scale = s + 1 < iters_s
                        if has_next_scale:
                            scale_handles = []
                            for _pj in cutlass.range_constexpr(pages_s):
                                scale_handles.append(
                                    scale_consumer.wait_and_advance()
                                )
                            scale_stage0 = scale_handles[0].index
                            for smi in cutlass.range_constexpr(tiles_sm):
                                # token = smi*128 + warpgroup_tidx ->
                                # page j = smi*pages_m + tok_half; ring
                                # slot (stage0 + j) % scale_stages.
                                stg_j = (
                                    scale_stage0
                                    + smi * pages_m
                                    + tok_half
                                ) % scale_stages
                                if cutlass.const_expr(not qflex):
                                    sSZ = cute.make_tensor(
                                        (
                                            sScale.iterator
                                            + (
                                                tok_row * 4
                                                + stg_j * page_size * 4
                                            )
                                        ).align(16),
                                        cute.make_layout(4),
                                    )
                                    cute.autovec_copy(
                                        sSZ, rSZ2[None, smi]
                                    )
                                else:
                                    if cutlass.const_expr(k_ring_w > 0):
                                        sKZ = cute.make_tensor(
                                            (
                                                sScale.iterator
                                                + (tok_row * k_ring_w
                                                   + stg_j
                                                   * scale_ring_elems)
                                            ).align((k_ring_w * self.scale_dtype.width) // 8),
                                            cute.make_layout(k_ring_w),
                                        )
                                        for e in cutlass.range_constexpr(
                                            k_ring_w
                                        ):
                                            rSZ2[e, smi] = Float32(sKZ[e])
                                    if cutlass.const_expr(v_ring_w > 0):
                                        sVZ = cute.make_tensor(
                                            (
                                                sScale.iterator
                                                + (v_ring_off
                                                   + tok_row * 2
                                                   + stg_j
                                                   * scale_ring_elems)
                                            ).align((2 * self.scale_dtype.width) // 8),
                                            cute.make_layout(2),
                                        )
                                        for e in cutlass.range_constexpr(
                                            2
                                        ):
                                            rSZ2[k_ring_w + e, smi] = (
                                                Float32(sVZ[e])
                                            )
                            # Early release: everything downstream (the
                            # K-fold, the P-quant s_v, c_h) reads only
                            # the register quads.
                            # generic-proxy LDS -> async-proxy fence -> release: the
                            # TMA producer refills the stage through the async proxy,
                            # and without the bridge the refill may land before these
                            # reads. fence.proxy.async is the documented bridge; it is
                            # executed by EVERY softmax thread (no elect predicate).
                            # The warp barrier orders all 32 lanes' fences before lane
                            # 0's `mbarrier.arrive`, as at the prologue fetch.
                            cute.arch.fence_view_async_shared()
                            cute.arch.sync_warp()
                            for sh in scale_handles:
                                sh.release()

                    # channel-K: fold coefficients come from the bias
                    # ring instead of per-token quads — the fold is
                    # S = 2^m_G[h] * S_raw + bias_G[h] (per-head
                    # constants; the per-token machinery above is
                    # inert since k_ring_w == 0). Ring rows: the lift
                    # 2^m_G[h] at [0, g), bias_G[h] at [g, 2g). The lift
                    # is read PER ROW, one value per head row, never a
                    # plane-wide scalar broadcast into every rSK[n].
                    if cutlass.const_expr(k_channel):
                        qb_handles = []
                        for _kp in cutlass.range_constexpr(k_parts):
                            qb_handles.append(
                                qb_consumer.wait_and_advance()
                            )
                        # VECTORIZED coefficient read. `sQB` is
                        # (2*blk_tile_n, qp_stages) with the coefficient
                        # mode at stride 1, so the g lifts are one
                        # contiguous run and the g biases are the next —
                        # `qb_d` slices each as a single vector load
                        # instead of `blk_tile_n` scalar LDS apiece.
                        def qb_read(idx, dst_s, dst_z, qb_d=qb_d):
                            cute.autovec_copy(qb_d[None, 0, idx], dst_s)
                            cute.autovec_copy(qb_d[None, 1, idx], dst_z)

                        if cutlass.const_expr(k_parts == 1):
                            # Gt=256: ONE plane serves every M-tile, so
                            # read the coefficient vectors ONCE and reuse
                            # the registers. Previously each M-tile
                            # re-read the identical sQB slot.
                            qb_read(qb_handles[0].index, rQBs, rQBz)
                        for smi in cutlass.range_constexpr(tiles_sm):
                            if cutlass.const_expr(
                                k_parts == 2 * tiles_sm
                            ):
                                # ch64: two groups per M-tile. `tok_half`
                                # is WARP-UNIFORM (warpgroup_tidx//64, so
                                # warps 0/1 are half 0 and warps 2/3 are
                                # half 1), so this is a uniform branch —
                                # load only the selected group's vectors
                                # instead of loading BOTH and blending
                                # with a multiply/add per row.
                                iA = qb_handles[2 * smi].index
                                iB = qb_handles[2 * smi + 1].index
                                if tok_half == 0:
                                    qb_read(iA, rQBs, rQBz)
                                else:
                                    qb_read(iB, rQBs, rQBz)
                            elif cutlass.const_expr(k_parts > 1):
                                qb_read(qb_handles[smi].index,
                                        rQBs, rQBz)
                            for n in cutlass.range_constexpr(
                                blk_tile_n
                            ):
                                rSK[n, smi] = rQBs[n]
                                rZQ[n, smi] = rQBz[n]
                        for qh in qb_handles:
                            qh.release()

                # Load S from tmem and notify BMM1
                s_token = s_consumer.try_wait()
                s_handle = s_consumer.wait_and_advance(s_token)
                tStS_s = tStS[None, None, s_handle.index]
                tSrS_s = cute.make_rmem_tensor(tSsP.shape[:-1], acc_dtype)
                cute.copy(tmem_load_atom_s, tStS_s, tSrS_s)
                if cutlass.const_expr(dual_s):
                    tSrS_s2 = cute.make_rmem_tensor(
                        tSsP.shape[:-1], acc_dtype
                    )
                    cute.copy(
                        tmem_load_atom_s,
                        tStS2[None, None, s_handle.index],
                        tSrS_s2,
                    )
                cute.arch.fence_view_async_tmem_load()
                s_handle.release()

                if cutlass.const_expr(k_ch64):
                    # per-thread group select: this thread's token
                    # page-half picks its group's chain (even -> S1,
                    # odd -> S2); downstream fold is the standard
                    # single-chain form on the selected values
                    mh_sel = Float32(Int32(tok_half))
                    tSrS_s.store(
                        (Float32(1.0) - mh_sel) * tSrS_s.load()
                        + mh_sel * tSrS_s2.load()
                    )

                # K-fold apply: S = s_t * S_raw + (z_t - BETA_FOLD*s_t)
                # * QSum_h per element, BEFORE the max-reduce/exp2. The
                # uniform sm/qs_Q factor stays in scale_s_log2_e at the
                # exp2 (baseline log2-space convention) — only the
                # per-token coefficients cannot ride it.
                tSrS_f = tSrS_s
                if cutlass.const_expr(fold_k):
                    tSrS_f = cute.make_rmem_tensor(
                        (blk_tile_n, tiles_sm), acc_dtype
                    )
                    # This store has TWO candidate products for the add's
                    # contraction (rSK*S here and z_kp*rQS in the rZQ
                    # precompute), and LLVM's choice between them is
                    # per-element heterogeneous. That is why the folded S
                    # moves by one bf16 ulp on head%4==0 columns whenever
                    # the code around the precompute changes shape, and
                    # why a packed-f32x2 form of this site (fma2, and
                    # mul2+add2, which ptxas re-fuses to FFMA2) drifts the
                    # folded-S tap on a third of the lanes by 1-2 ulp. It
                    # stays scalar.
                    if cutlass.const_expr(k_tok64):
                        # S = s'_lo * S_lo + s'_hi * S_hi + (z'_lo *
                        # QSum_lo + z'_hi * QSum_hi)  [rZQ holds the
                        # combined bias term]
                        tSrS_f.store(
                            rSK.load()
                            * tSrS_s.load().reshape(
                                (blk_tile_n, tiles_sm)
                            )
                            + rSK2.load()
                            * tSrS_s2.load().reshape(
                                (blk_tile_n, tiles_sm)
                            )
                            + rZQ.load()
                        )
                    else:
                        tSrS_f.store(
                            rSK.load()
                            * tSrS_s.load().reshape((blk_tile_n, tiles_sm))
                            + rZQ.load()
                        )
                    # debug_s tap: folded S of tile 0, pre-mask and pre-scale
                    if cutlass.const_expr(debug_s and not is_masked_loop):
                        if (s == 0) & is_dbg_cta:
                            for smi in cutlass.range_constexpr(tiles_sm):
                                for n in cutlass.range_constexpr(
                                    blk_tile_n
                                ):
                                    mDbgS[
                                        smi * mma_tile_m + warpgroup_tidx,
                                        n,
                                    ] = tSrS_f[n, smi]

                # Apply causal mask
                if cutlass.const_expr(is_masked_loop):
                    masked = cute.make_rmem_tensor(
                        (blk_tile_h, blk_tile_p, tiles_sm), acc_dtype
                    )
                    masked.store(tSrS_f.load().reshape(masked.shape))
                    offset_s = (masked_coord_s + s) * blk_tile_s + warpgroup_tidx
                    offset_p = seqlen - prediction + coord_p * blk_tile_p
                    for sm in cutlass.range_constexpr(tiles_sm):
                        for p in cutlass.range_constexpr(blk_tile_p):
                            masked_p = masked[None, p, sm]
                            is_oob = offset_s + sm * mma_tile_m > offset_p + p
                            mask = -Float32.inf if is_oob else Float32(0)
                            masked_p.store(masked_p.load() + mask)
                    scores = masked.load().reshape((blk_tile_n, tiles_sm))
                else:
                    scores = tSrS_f.load().reshape((blk_tile_n, tiles_sm))

                # Reduce colmax in thread RF
                rM = cute.make_rmem_tensor_like(sM)
                rM.store(
                    scores.reduce(
                        cute.ReductionOp.MAX,
                        init_val=-Float32.inf,
                        reduction_profile=(None, 0),
                    )
                )

                # Reduce colmax in warp RF
                rM_lane = Float32(0)
                for n in cutlass.range_constexpr(blk_tile_n):
                    rM[n] = warp_fmax(rM[n])  # warp reduction
                    # Avoid dynamic register indexing (creates spills)
                    if n == lane_idx:
                        rM_lane = rM[n]
                if cutlass.const_expr(mQD is not None):
                    rM_lane *= qd_lane  # this lane's row's scale and descale
                else:
                    rM_lane *= scale_s_log2_e  # apply scale

                # Compute skip predicate
                keep_tile = not enable_blasst
                if cutlass.const_expr(enable_blasst):
                    # warp reduction
                    lane_keep_tile = rM_lane - sM_lane_prev >= log2_threshold_p
                    lane_keep_tile &= lane_store_max  # oob lanes skip
                    lane_keep_tile |= s < o_stages  # correction loop is s-2
                    warp_keep_tile = warp_or(Int32(lane_keep_tile))

                    # warpgroup reduction
                    sp_handle = sp_producer.acquire_and_advance()
                    with cute.arch.elect_one():
                        sSP_i8[warpgroup_widx, sp_handle.index] = Int8(
                            warp_keep_tile
                        )
                    sp_handle.commit()
                    sSP_producer_nbar.sync()
                    keep_tile = sSP_i32[sp_handle.index] != 0

                if keep_tile:
                    p_token = p_producer.try_acquire()

                    sM_consumer_nbar.arrive_and_wait()
                    if lane_store_max:
                        smem_fmax(
                            sM.iterator + sM.layout(lane_idx), rM_lane
                        )
                    # Wait for empty P buffer
                    # Here so we can interleave ex2 with convert ops
                    p_handle = p_producer.acquire_and_advance(p_token)
                    tSsP_s = tSsP[None, None, p_handle.index]

                    # Load colmax
                    sM_producer_nbar.arrive_and_wait()
                    colmax = sM.load()
                    if cutlass.const_expr(fold_p):
                        # POOL-WIDE pow-2 pscale: lp = floor(log2(448 /
                        # max s_v)) from the cache's count table
                        # (`_pool_pscale_lp`); the hi/lo residual structure
                        # absorbs the conservatism (certified flat to 64x).
                        # Constant
                        # -> no sVmax window, no per-tile log2. Under
                        # channel-wise K read from `lp_cell` here, at its use
                        # (see the kernel prologue); token-wise K keeps the
                        # register it computed.
                        if cutlass.const_expr(lp_cell is not None and k_channel):
                            lp_now = lp_cell[0]
                        else:
                            lp_now = pscale_lp
                        pscale = exp2(lp_now)
                    if cutlass.const_expr(enable_blasst):
                        if lane_store_max:
                            sM_lane_prev = sM[lane_idx]
                    # Handle if we never saw any in-bounds values
                    if cutlass.const_expr(is_masked_loop and do_atomic_red):
                        if check_safe_max:
                            rM = cute.make_rmem_tensor_like(colmax)
                            rM.store(colmax)
                            for n in cutlass.range_constexpr(blk_tile_n):
                                if rM[n] == -Float32.inf:
                                    rM[n] = Float32(0)
                            colmax = rM.load()

                    # Compute online softmax
                    if cutlass.const_expr(mQD is not None):
                        probs = exp2(rQD.load() * scores - colmax)
                    else:
                        probs = exp2(scale_s_log2_e * scores - colmax)

                    # Store P to smem and notify BMM2
                    if cutlass.const_expr(fold_p):
                        # P hi/lo e4m3 rows (the certified reference's p_hilo): s_v
                        # folded INSIDE the operand (a per-token scale on
                        # the contraction dim cannot be applied after the
                        # MMA), running pow-2 pscale, lo = residual at
                        # 2^LO_SHIFT * pscale. hi rows at n in [0, g), lo
                        # rows at n in [g, 2g) (blocked halves); rows
                        # [2g, mma_tile_n) are never read (the correction
                        # WG loads only 2g O columns). Addressing goes
                        # through tCsP — the same composed (m, n) C-view
                        # the baseline P store partitions — per-token
                        # column slices, blk_tile_n contiguous bytes.
                        cvecS = None
                        if cutlass.const_expr(
                            v_channel and v_parts > 1
                        ):
                            cvecS = cute.make_rmem_tensor(
                                (blk_tile_n, tiles_sm), acc_dtype
                            )
                        rHi = cute.make_rmem_tensor(blk_tile_n, mma_dtype)
                        rLo = cute.make_rmem_tensor(blk_tile_n, mma_dtype)
                        inv_lo = Float32(1.0 / (1 << self.LO_SHIFT))
                        cvec = None
                        for smi in cutlass.range_constexpr(tiles_sm):
                            if cutlass.const_expr(v_channel):
                                # channel-V: s_v is per-CHANNEL and
                                # drains post-MMA in the correction
                                # WG; P carries pscale only.
                                w2 = probs[None, smi] * pscale
                            else:
                                w2 = probs[None, smi] * (
                                    rSV[smi] * pscale
                                )
                            rHi.store(w2.to(mma_dtype))
                            hi_f = rHi.load().to(acc_dtype)
                            res = (w2 - hi_f) * lo_scale
                            rLo.store(res.to(mma_dtype))
                            pC = tCsP[
                                (warpgroup_tidx, None),
                                0,
                                0,
                                smi,
                                p_handle.index,
                            ]  # (mma_tile_n,) — token's P column
                            cute.autovec_copy(
                                rHi,
                                cute.local_tile(
                                    pC, (blk_tile_n,), (0,)
                                ),
                            )
                            cute.autovec_copy(
                                rLo,
                                cute.local_tile(
                                    pC, (blk_tile_n,), (1,)
                                ),
                            )
                            # c_h partial from the QUANTIZED P rows —
                            # the certified reference's consistent
                            # form c = sum_rows (r @ (z'_v *
                            # rcp(s_v))) * dsp, NOT the plan text's
                            # f32-p form. Consistency makes the
                            # P-quant noise multiply the CENTERED
                            # value V_hat/s (about +-qmax/2) instead
                            # of the raw feed (mean qmax/2 +
                            # BETA_FOLD), which is load-bearing at
                            # int2, where the feed's common mode is
                            # 9.5. The rcp clamp (+1e-30) keeps
                            # zero-filled OOB scale stages at 0.
                            # channel-V: the same consistent sum
                            # WITHOUT the per-token zr (z_d is
                            # per-channel, applied at the drain) —
                            # the [g, 2g) tL columns carry
                            # l = sum (hi + lo/2^4) in pscale
                            # units (the drain's single inv_ps
                            # scaling covers it).
                            if cutlass.const_expr(v_channel):
                                contrib = (
                                    hi_f
                                    + rLo.load().to(acc_dtype)
                                    * inv_lo
                                )
                            else:
                                zr = rZV[smi] / (
                                    rSV[smi] + Float32(1e-30)
                                )
                                contrib = (
                                    hi_f
                                    + rLo.load().to(acc_dtype)
                                    * inv_lo
                                ) * zr
                            if cutlass.const_expr(
                                v_channel and v_parts > 1
                            ):
                                # sub-tile group parts: keep the
                                # per-smi l partials separate (the
                                # per-part tL stores mask/select)
                                cvecS[None, smi].store(contrib)
                            elif cutlass.const_expr(smi == 0):
                                cvec = contrib
                            else:
                                cvec = cvec + contrib
                        if cutlass.const_expr(not v_channel):
                            cvec = cvec * exp2(-lp_now)  # / pscale
                    else:
                        tSsP_s.store(
                            probs.to(mma_dtype).reshape(tSsP_s.shape)
                        )
                    cute.arch.fence_view_async_shared()
                    p_handle.commit()

                    # Correction bookkeeping, AFTER the P commit
                    # (fix iteration 2 — see the pre-commit comment).

                    # Accumulate per-thread colsum (true f32 P — the
                    # reference's cheap l_mode='f32P'); fold mode also carries
                    # the c_h partial (computed above from the quantized
                    # P rows) in columns [g, 2g).
                    colsum = probs[None, 0]
                    for sm in cutlass.range_constexpr(1, tiles_sm, 1):
                        colsum += probs[None, sm]
                    if cutlass.const_expr(v_channel):
                        # channel-V: PRE-REDUCED [colsum | l] smem
                        # ring, one entry per group part (the tL
                        # tmem/named-barrier channel cannot run at
                        # v_parts rate). Each warp REDUX-sums its 32
                        # tokens' contributions per column into its
                        # sLC warp slot; the correction WG sums the 4
                        # slots in fixed order. A part covers smi =
                        # pp // psk_s and (when two parts share an
                        # smi chunk, Gt=64) the tokens of one
                        # 64-token page half — a thread contributes
                        # iff its token is in the part (mask).
                        psk_s = v_parts // tiles_sm
                        prS = cute.make_rmem_tensor(
                            (blk_tile_n, tiles_sm), acc_dtype
                        )
                        prS.store(probs)
                        for pp in cutlass.range_constexpr(v_parts):
                            smi_p = pp // psk_s if psk_s >= 1 else pp
                            if cutlass.const_expr(psk_s > 1):
                                mask = Float32(
                                    Int32(tok_half == pp % psk_s)
                                )
                            else:
                                mask = Float32(1.0)
                            lch = lc_producer.acquire_and_advance()
                            for n in cutlass.range_constexpr(
                                blk_tile_n
                            ):
                                if cutlass.const_expr(v_parts > 1):
                                    cv_p = prS[n, smi_p] * mask
                                    lv_p = cvecS[n, smi_p] * mask
                                else:
                                    cv_p = colsum[n]
                                    lv_p = cvec[n]
                                cv_r = cute.arch.warp_reduction_sum(
                                    cv_p
                                )
                                lv_r = cute.arch.warp_reduction_sum(
                                    lv_p
                                )
                                if n == lane_idx:
                                    sLC[
                                        n, warpgroup_widx, lch.index
                                    ] = cv_r
                                    sLC[
                                        blk_tile_n + n,
                                        warpgroup_widx,
                                        lch.index,
                                    ] = lv_r
                            # order every lane's store before lane
                            # 0's release arrive
                            cute.arch.sync_warp()
                            lch.commit()
                    else:
                        tSrL = cute.make_rmem_tensor(tSrL_shape, acc_dtype)
                        if cutlass.const_expr(fold_p):
                            for n in cutlass.range_constexpr(blk_tile_n):
                                tSrL[n] = colsum[n]
                                tSrL[blk_tile_n + n] = cvec[n]
                        else:
                            tSrL.store(colsum.reshape(tSrL.shape))

                        # Store per-thread colsum to tmem
                        with_phase(
                            tL_consumer_nbar, softmax_phase
                        ).arrive_and_wait()
                        cute.copy(
                            tmem_store_atom_l, tSrL, tStL[None, softmax_phase]
                        )
                        cute.arch.fence_view_async_tmem_store()
                        with_phase(tL_producer_nbar, softmax_phase).arrive()

                    # Advance the tL ring slot
                    if cutlass.const_expr(v_channel and v_parts > 1):
                        # tL phases advance per PART
                        softmax_phase += v_parts
                        if softmax_phase >= o_stages:
                            softmax_phase -= o_stages
                        if softmax_phase >= o_stages:
                            softmax_phase -= o_stages
                    elif cutlass.const_expr(o_stages == 2):
                        softmax_phase ^= 1
                    else:
                        softmax_phase += 1
                        if softmax_phase == o_stages:
                            softmax_phase = 0


    @cute.kernel
    def kernel(
        self,
        # MMA
        blk_tile_shpd: cute.Tile,
        mma_tile_mnk: cute.Tile,
        tiled_mma_kq: cute.TiledMma,
        tiled_mma_vp: cute.TiledMma,
        mma_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        # Page Table
        seqlens_iter: Union[cute.Pointer, Int32],
        table_offsets_iter: Optional[cute.Pointer],
        page_table_iter: cute.Pointer,
        # K
        smem_layout_k_mma: cute.ComposedLayout,
        smem_layout_k_tma: cute.ComposedLayout,
        # V
        smem_layout_v_mma: cute.ComposedLayout,
        # Code ring: plain ((box), #boxes, code_stages) layouts over the packed
        # pool bytes.
        smem_layout_code_k_tma: Union[cute.ComposedLayout, cute.Layout],
        smem_layout_code_v_tma: Union[cute.ComposedLayout, cute.Layout],
        # Packed-code TMA
        tma_atom_kc: Optional[cute.CopyAtom],
        mKC: Optional[cute.Tensor],
        tma_atom_vc: Optional[cute.CopyAtom],
        mVC: Optional[cute.Tensor],
        tma_atom_sc: Optional[cute.CopyAtom],
        mSC: Optional[cute.Tensor],
        mSC_flat: Optional[cute.Tensor],
        # Debug folded-S tap (debug_s mode)
        mDbgS: Optional[cute.Tensor],
        # Q
        smem_layout_q: cute.ComposedLayout,
        tma_atom_q: cute.CopyAtom,
        mQ: cute.Tensor,
        mQraw: Optional[cute.Tensor],  # plain-stride Q (the QSum reads)
        # O
        smem_layout_o: cute.ComposedLayout,
        tma_atom_o: Optional[cute.CopyAtom],
        mO: Optional[cute.Tensor],
        mOraw: Optional[cute.Tensor],  # plain-stride O (the correction's direct store)
        # Rest
        mL: Optional[cute.Tensor],  # LSE output (Float32, natural log), unsplit builds only
        mL_partial: Optional[cute.Tensor],
        mM_partial: Optional[cute.Tensor],
        scale_s_log2_e: Float32,
        scale_o: Float32,
        # f32: the P-scale exponent of a build that reads no table, or None (see
        # THE P-SCALE EXPONENT below).
        pscale_lp: Optional[Float32],
        # uint32 [256]: the cache's count table of page maxima, from which the
        # kernel takes the pool-wide power-of-2 pscale exponent for the P hi/lo
        # rows, lp = floor(log2(448 / max s_v over the pool)) (`_pool_pscale_lp`;
        # certified at up to 64x conservatism). None where the build
        # reads no exponent from the cache.
        mVMaxHist: Optional[cute.Tensor],
        # fp32 [B, HQ] query descale, or None for the kernel without the
        # per-row factor (the descale then rides in scale_s_log2_e).
        mQD: Optional[cute.Tensor],
        log2_threshold_scale_factor: Optional[Float32],
        # int32 [W, 4]: the graph split build's work list, one (item, split, k, seq_len)
        # entry per CTA (`call_graph`), or None for every other build.
        mWork: Optional[cute.Tensor],
        # The engine page table's row stride, with `mWork` only.
        graph_max_pages: Optional[Int32],
    ):
        ##############################
        # Static variables
        ##############################
        # Smem alloc helper
        svector_align = 16
        stensor_align = 128
        smem = utils.SmemAllocator()

        # No multicast
        mcast_coord = 0
        mcast_layout = cute.make_layout((1, 1, 1, 1))  # vmnk

        # Alias types
        q_dtype = k_dtype = mma_dtype
        o_dtype = out_dtype
        acc_dtype = Float32

        # Shapes for MMA tile indexing
        blk_tile_s, blk_tile_h, blk_tile_p, blk_tile_d = blk_tile_shpd
        blk_tile_hp = (blk_tile_h, blk_tile_p)  # multimode tiler
        blk_tile_n = blk_tile_h * blk_tile_p  # linearized tiler
        mma_tile_m, mma_tile_n, mma_tile_k = mma_tile_mnk
        tiles_sm = blk_tile_s // mma_tile_m
        tiles_sk = blk_tile_s // mma_tile_k
        tiles_dm = cute.ceil_div(blk_tile_d, mma_tile_m)
        tiles_dk = cute.ceil_div(blk_tile_d, mma_tile_k)
        page_size = self.page_size
        pages_s = blk_tile_s // page_size
        pages_m = mma_tile_m // page_size
        pages_k = mma_tile_k // page_size

        # Static control flow
        do_kernel_red = self.do_kernel_red
        do_atomic_red = self.do_atomic_red
        do_none_red = self.do_none_red
        store_lse = mL is not None
        is_varlen = isinstance(seqlens_iter, cute.Pointer)
        enable_blasst = log2_threshold_scale_factor is not None
        # fold_k gates the K-SIDE fold machinery (K-fold FMAs, QSum, the
        # scale ring) and fold_p the P-SIDE machinery (sVmax/pscale, P hi/lo
        # rows, c_h, widened L/O tmem, V-fold epilogue). The two are always
        # equal, because no configuration separates them. They stay as two
        # names because 48 sites read one or the other, and collapsing them is
        # a rename to make on its own.
        # `fold_scale_zero=False` (ctor-scoped to int2 + e4m3 feed + legacy
        # quant) is the classic-dequant ablation: the dequant WG applies
        # (s, z) to the codes and publishes REAL dequantized e4m3 operands,
        # so the whole fold is off and the numbers are still RIGHT.
        # `nofold` therefore drives both fold gates AND turns the scale
        # ring's consumer into the dequant WG.
        nofold = not self.fold_scale_zero
        fold_k = not nofold
        fold_p = fold_k
        bits = self.bits
        # The int2 K AND V feeds are packed e2m1 direct-SMEM. The ctor
        # asserts bits == 2.
        f4_feed = self.dequant_int2_to_f4
        persistent = self.persistent
        # Quant-granularity flex consts. Legacy compiles with all of these
        # False/inert; every flex arm below is const_expr-gated so the
        # shipped token-128 path stays byte-identical.
        qflex = not self.qoff["legacy"]
        k_is_token = self.k_axis == "token"
        v_is_token = self.v_axis == "token"
        v_channel = self.v_axis == "channel"
        scale_ring_elems = self.qoff["ring_tot"]
        # Scale ring live == the quant config gives it elements. One of its
        # two consumers always exists (the softmax WG in the fold arm, the
        # dequant streams in the fold-off arm), so the ring is dead only when
        # both tensors are channel-wise and it has no elements. A
        # produced-never-consumed ring deadlocks its producer at stage
        # scale_stages+1, so the TMA issue and both consumers key off exactly
        # this one predicate.
        scale_ring_live = scale_ring_elems > 0
        v_ch_off = self.qoff["v_ch"] if v_channel else 0
        v_ring_off = self.qoff["v_ring"] if v_is_token else 0
        # per-token ring elements per tensor (flex sections): K token-128 ->
        # (s, z); K token-64 -> (s_lo, z_lo, s_hi, z_hi); channel -> none
        k_ring_w = ((2 if self.k_group == self.head_dim else 4)
                    if k_is_token else 0)
        # K token-64: ring carries (s_lo, z_lo, s_hi, z_hi) per token; the
        # QK MMA runs two accumulator chains (mma_k halves of the D
        # contraction) into tCtS / tCtS2
        k_tok64 = k_is_token and self.k_group == self.head_dim // 2
        # K channel-64: two groups per 128-token M-tile — the QK MMA runs
        # each M-tile TWICE (one full-D chain per group's Q' plane) into
        # tCtS / tCtS2; the softmax selects per thread by token page-half.
        # Structural form (k_parts > tiles_sm), NOT `sequence_tile // k_group > 2`
        # — see the k_ch64_c note beside the tmem accounting.
        k_ch64 = (self.k_axis == "channel" and self.k_group < mma_tile_m)
        dual_s = k_tok64 or k_ch64
        v_ring_w = 2 if v_is_token else 0
        flex_ring_w = k_ring_w + v_ring_w
        # channel-V dequant feed constants: V codes feed e4m3(q + beta) (or
        # e2m1 q/2 under f4), so the per-channel drain uses s'_d = s_d (2*s_d
        # under f4) and z'_d = z_d - beta*s_d (beta = 0 under f4/int4).
        # int4 raw-code V feed at channel-V: the drain compensates the
        # alpha = 2**-9 as s'_d = 512 * s_d (s_v never rides inside P on this
        # axis). Pure f32 drain multiplies; the 512 cancels the operand's
        # 2**-9 bit-exactly.
        vch_s_mul = (512.0 if bits == 4
                     else (2.0 if f4_feed else 1.0))
        vch_beta = 0.0 if f4_feed else self.BETA_FOLD[bits]
        # channel-V group parts per sequence tile: each part (one group's
        # token range) gets its OWN fresh-written o-slot + [colsum | l]
        # vector, drained separately (Gt=256 -> 1, 128 -> 2, 64 -> 4 at
        # sequence_tile=256). The O ring deepens to o_stages * v_parts
        # virtual slots (mbarrier pipeline — not named-barrier-limited) so
        # the o_stages-TILE drain lag still fits; the [colsum | l] transport
        # moves off the tL tmem/named-barrier channel onto a PRE-REDUCED
        # smem ring (softmax warp-reduces to [nL, 4 warps] = 256B/stage,
        # PipelineAsync at the same o_ring depth) — the named-barrier file
        # is full at o_stages == 3 and cannot hold v_parts-rate phases.
        v_parts = ((self.sequence_tile // self.v_group) if v_channel else 1)
        v_parts = max(v_parts, 1)
        v_pages = (self.v_group // self.page_size) if v_channel else 0
        o_ring = self._o_stages * v_parts if v_channel else self._o_stages
        lc_stages = o_ring + v_parts  # one tile of slack over the drain lag
        # channel-K: the TMA-QK warp builds per-group Q' planes
        # (e4m3(s'_d * Q * 2^-m_G), the B operand of the QK MMA) + a bias
        # ring entry (2^m_G, bias_G[h] = sum_d z'_d * Q[h, d]) — the hot
        # softmax K-fold becomes S = 2^m * S_raw + bias[h]. Groups per tile:
        # Gt=256 -> 1 plane / tile, Gt=128 -> 1 per M-tile (K channel-64 is
        # stage e: dual planes + dual S accumulators).
        k_channel = self.k_axis == "channel"
        k_parts = ((self.sequence_tile // self.k_group) if k_channel else 1)
        k_parts = max(k_parts, 1)
        k_pages = (self.k_group // self.page_size) if k_channel else 0
        k_ch_off = self.qoff["k_ch"] if k_channel else 0
        kch_s_mul = 2.0 if f4_feed else 1.0
        kch_beta = 0.0 if f4_feed else self.BETA_FOLD[bits]
        # int4 raw-code feed at channel-K: the alpha = 2**-9 is absorbed in
        # the PUBLISHED LIFT (lift' = 512 * 2^m) rather than in s'_d. Growing
        # s'_d 512x would shift the po2 ladder m by +9 against _lift_exp's
        # clamp at m = 10, which has no such headroom guarantee; scaling the
        # lift is the same algebra with no clamp exposure — the plane bytes
        # and m are unchanged, and 2^(m+9) * ((q * 2**-9) @ Qp) =
        # 2^m * (q @ Qp) exactly. Bias rows are untouched (beta = 0).
        kch_lift_mul = self.ALPHA_INV_INT4 if bits == 4 else 1.0
        # Q'-plane build-ahead: the CORRECTION WG builds tile t's planes at
        # its iteration t - kb_ahead - 1 (the preamble seeds the first
        # kb_ahead + 1 tiles), so planes exist before the KQ MMA and the
        # softmax bias reads need them, acyclic by construction. The ring
        # must hold the full build-ahead window.
        kb_ahead = self._o_stages if k_channel else 0
        qp_stages = ((kb_ahead + 2) * k_parts) if k_channel else 0
        # THE P-SCALE EXPONENT. Where s_v rides inside P (the fold with
        # token-wise V), lp = floor(log2(448 / max s_v over the pool)) comes from
        # the cache's count table (`_pool_pscale_lp`): in a one-item CTA the
        # softmax warpgroup reads it, and the correction warpgroup either reads it
        # too (token-wise K) or reads the softmax's copy in `sLp` (channel-wise K,
        # and token-wise K where `lp_tail` holds);
        # in a persistent CTA the prologue reads it once into `sLp` and every
        # pscale consumer (P-quant, c_h, both V-fold epilogues) reads it from
        # there per item. Channel-wise V carries
        # no s_v in P, so max P = 1 and lp is the constant floor(log2(448)) = 8;
        # the fold-off ablation reads no exponent, and gets the same constant.
        # Such a build takes that constant as the launch scalar `pscale_lp`, the
        # way the host's exponent arrives, not as a constant in the kernel: with
        # the constant, the compiler keeps the correction warpgroup's channel-wise
        # K scale addresses live in registers across their guarded loads instead of
        # recomputing them, and persistent channel-wise-K builds run up to 4.4%
        # slower. With the scalar, these builds' PTX differs from the constant
        # form's only by the O store's commit and waits and by the persistent item
        # closure.
        # The table's exponent is re-anchored ONCE, where it is computed, so that
        # every consumer reads consistent units:
        # - int4 raw-code V feed: the feed is q * 2**-9, so s_v rides x512 inside
        #   the P operand (w = P * (512*s_v), one FMUL at the scale-quad copy).
        #   lp - 9 keeps w * pscale = P * s_v * 2^lp, so the quantized P hi/lo
        #   bytes stay BIT-IDENTICAL (the FMUL's product cancels the shift
        #   before any rounding).
        # - f4 feed: the feed is q/2, so s_v rides DOUBLED inside P
        #   (w = P * (2*s_v)), and lp - 1 keeps the P bytes identical to the
        #   flag-off path in the same way.
        # Both are anchored on the flag-off exponent floor(log2(448 / max s_v));
        # the table must not pre-compensate the feed, or the shift is applied
        # twice. Channel-wise V is exempt from both (the drain's s'_d carries the
        # 512 or the 2, see vch_s_mul).
        reads_lp = fold_p and v_is_token
        lp_shift = 0
        if cutlass.const_expr(reads_lp and bits == 4):
            lp_shift = -9
        if cutlass.const_expr(reads_lp and f4_feed):
            lp_shift = -1
        ##############################
        # Warp specialization
        ##############################
        # Warp assignments
        warpgroup_id = 0
        mma_kq_warp_id = warpgroup_id * warpgroup_warps + 0
        mma_vp_warp_id = warpgroup_id * warpgroup_warps + 1
        tma_qk_warp_id = warpgroup_id * warpgroup_warps + 2
        tma_vo_warp_id = warpgroup_id * warpgroup_warps + 3
        reduction_warp_id = mma_kq_warp_id
        warpgroup_id += 1

        softmax_warpgroup_ids = (warpgroup_id,)
        warpgroup_id += 1

        correction_warpgroup_id = warpgroup_id
        warpgroup_id += 1

        dequant_warpgroup_ids = (warpgroup_id,)
        warpgroup_id += 1
        assert self.threads_per_cta == warpgroup_id * warpgroup_threads

        # Register budgets.
        # A CTA of four warpgroups can hold at most 128 registers per thread (the SM's 64K over
        # 512 threads). In the builds below the correction warpgroup needs more than that, and
        # the dequant warpgroup, which needs the fewest, hands it 128 - dequant_regs of its own
        # with setmaxnreg, so the correction warpgroup's peak does not decide where ptxas
        # spills for every role. The MMA/TMA and softmax warpgroups keep 128 and run no
        # setmaxnreg: compiled under a lower count, the MMA warps rebuilt uniform operands
        # inside their tile loops, and the softmax warpgroup has no registers to spare.
        #
        # The correction warpgroup drains every channel-wise V group's O slot into register
        # accumulators and builds the Q' plane of every channel-wise K group, for each of the
        # tile's head rows. Its state overflows 128 registers, and giving it more is faster in
        # these builds:
        # - channel-wise V at INT4, with either K axis;
        # - channel-wise V at INT2 with channel-wise K, when the sequence tile holds more than
        #   one K group or the tile has 8 head rows;
        # - token-wise V with channel-wise K at 8 head rows, when the sequence tile holds two K
        #   groups, or one at INT4.
        # Everywhere else the rebalance is not needed or measures slower. That includes every
        # build with token-wise K and V, and channel-wise K of four groups per sequence tile
        # with token-wise V, where it lengthens the MMA KQ warp's tile loop. Only the default
        # tiling of each quantization pair is measured, and the TMA-mask variant, which
        # restructures the softmax loop, is not, so a build whose tiling differs from the
        # default (other groups per tile or another O-ring depth) or that sets `tma_mask`
        # keeps the launch count.
        heads8 = self.q_head_tile == 8
        measured = self._default_tiling and not self.tma_mask
        if cutlass.const_expr(v_channel):
            rebalance_regs = bool(measured and (
                self.bits == 4 or (k_channel and (k_parts > 1 or heads8))))
        else:
            rebalance_regs = bool(measured and k_channel and heads8 and (
                k_parts == 2 or (k_parts == 1 and self.bits == 4)))
        launch_regs = 64 * 1024 // self.threads_per_cta
        dequant_regs = 104
        correction_regs = 152
        assert launch_regs == 128, launch_regs
        assert dequant_regs + correction_regs == 2 * launch_regs, (
            "the correction warpgroup takes exactly what the dequant warpgroup gives back")

        # Read thread indices
        kv_splits, tiles_hp, tiles_hb = cute.arch.grid_dim()
        kv_split_idx, coord_hp, coord_hb = cute.arch.block_idx()
        # Capture the LAUNCHED z extent and this CTA's linear
        # z id before both names are rebound below (grid_z < items when
        # persistent; the item loop walks bidz, bidz + grid_z_rt, ...).
        grid_z_rt = tiles_hb
        bidz = coord_hb
        if cutlass.const_expr(do_none_red):
            kv_splits, kv_split_idx = (1, 0)
        # THE GRAPH SPLIT BUILD (`mWork`) takes its piece from the work list, at the very top:
        # every thread loads this CTA's 16-byte entry, a slot past the end of the list (item
        # -1) exits the whole CTA before the tensor-memory allocation, the descriptor prefetch
        # and every barrier, and the entry then stands where the split build reads its grid:
        # the item (request * HKV + KV head) for the z block index, the split for the x block
        # index, and the request's split count k for the x grid extent. The y block index stays
        # the block's own, 0 in the grid (1, 1, W). Each coordinate goes through
        # `make_warp_uniform` (`_graph_work_entry`), so the roles use them as they use the block
        # indices. The list is
        # written by the copy before the graph replays, never by the grid this one depends on,
        # so it is read before the dependent-launch wait. The length rides to the per-item
        # metadata path below, which publishes it through the length cell.
        graph_split = mWork is not None
        graph_seqlen = None
        if cutlass.const_expr(graph_split):
            coord_hb, kv_split_idx, kv_splits, graph_seqlen = self._graph_work_entry(
                mWork, bidz)
        tidx, _, _ = cute.arch.thread_idx()
        # In a persistent build the kernel-entry %tid.x goes through an opaque `mov` so
        # the compiler keeps this one value live instead of re-reading %tid.x at its uses.
        # When %tid.x is re-read inside the persistent item loop (and its inner loops),
        # ptxas can compile the item loop as divergent, and every MMA/TMA issue then
        # rebuilds its uniform operands under ELECT. A non-persistent build has no item
        # loop and keeps the plain read, which costs it no live register. The mask (1023,
        # the largest thread index a CTA can have) changes no value and keeps the
        # divisions below shifts.
        if cutlass.const_expr(self.persistent):
            tidx = _opaque_i32(tidx) & 1023
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // warp_threads)
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // warpgroup_threads)
        warpgroup_tidx = tidx % warpgroup_threads
        warpgroup_widx = warp_idx % warpgroup_warps
        init_warp = 1  # warp 0 does all pipeline inits for now

        # Unpack multimodes
        grouped_heads, prediction = mQ.shape[0]
        # SPLIT INT2 TOKEN-V WITH FOUR QUERY HEADS PER KV HEAD (`lp_tail`). In these split
        # builds, under token-wise K, the correction warpgroup takes the exponent from the
        # softmax's copy in `sLp` at the start of its tail, instead of reading the table
        # itself and holding the result in a register through its tile loop (channel-wise K
        # already takes the softmax's copy, at its epilogue). The split cells with four or
        # more tiles per CTA run faster this way, while those with one or two tiles per CTA
        # do not, and at INT4 the same change measures 0.4 to 3% slower, so those builds
        # keep the correction's table read.
        lp_tail = (reads_lp and self.do_kernel_red and f4_feed
                   and grouped_heads == 4)
        # PERSISTENT INT2 TOKEN-WISE K AND V AT THREE QUERY HEADS PER KV HEAD
        # (`compact_close`). In these builds the persistent exit-early chain tests the
        # correction warpgroup first, which makes its whole closure, and then makes one
        # first-item test for the two warps that only prime an ordering barrier (MMA-VP,
        # TMA-VO), instead of one branch per role each calling `_close_empty_item`. The
        # arrivals are the same; only the chain, which no full item executes, is shorter.
        # The shortened chain measures faster or equal on every case of the ratio-three
        # guard set except a one-tile-per-item cell, which pays about 0.6% for it. Other
        # KV-head counts at three query heads per KV head take the same object shape and
        # were not timed; the default tiling, no TMA mask and no query descale are
        # required because only those were timed.
        compact_close = (persistent and self.bits == 2 and f4_feed
                         and self.fold_scale_zero and self._default_tiling
                         and not self.tma_mask and mQD is None
                         and not k_channel and not v_channel and grouped_heads == 3)
        heads_k, batches = tiles_hb = mQ.shape[2]
        tiles_hp = cute.ceil_div(mQ.shape[0], blk_tile_hp)
        coord_hb = cute.idx2crd(coord_hb, tiles_hb)
        coord_hp = cute.idx2crd(coord_hp, tiles_hp)
        coord_hg, coord_p = coord_hp
        coord_hk, coord_b = coord_hb

        ##############################
        # Prefetch Seqlen
        ##############################
        cpasync_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), Int32, num_bits_per_copy=32
        )
        if cutlass.const_expr(is_varlen):
            scalar_layout = cute.make_layout(1)
            seqlen_smem = smem.allocate_tensor(Int32, scalar_layout)
            table_offset_smem = smem.allocate_tensor(Int32, scalar_layout)
            # A persistent build: coord_b changes per item, so this staging runs
            # inside the item loop instead (same protocol, same warp).
            if cutlass.const_expr(not persistent):
                if warp_idx == init_warp:
                    if cutlass.const_expr(graph_split):
                        # The graph split build: the work entry's length, and the row of
                        # the engine's [B, max_pages] page table, through the same two
                        # cells and the same set-up barrier. No dependent-launch wait: the
                        # entry is not the previous grid's output.
                        with cute.arch.elect_one():
                            seqlen_smem[0] = graph_seqlen
                            table_offset_smem[0] = coord_b * graph_max_pages
                    else:
                        seqlen_gmem = cute.make_tensor(
                            seqlens_iter + coord_b, scalar_layout
                        )
                        table_offset_gmem = cute.make_tensor(
                            table_offsets_iter + coord_b, scalar_layout
                        )
                        cute.arch.griddepcontrol_wait()
                        with cute.arch.elect_one():
                            cute.copy(cpasync_atom, seqlen_gmem, seqlen_smem)
                            cute.copy(
                                cpasync_atom, table_offset_gmem, table_offset_smem
                            )
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
            init_warp += 1

        ##############################
        # Prefetch TMA descriptor
        ##############################
        lp_warp = init_warp
        if warp_idx == init_warp:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_kc)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_vc)
            if cutlass.const_expr(tma_atom_sc is not None):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sc)
            if cutlass.const_expr(tma_atom_o is not None):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)
            # Where `lp_tail` holds and the softmax reads the count table in 16-byte loads
            # (`vmax_hist_vec`), the softmax warpgroup's first two global reads, this CTA's
            # query rows and then the count table, start after the set-up barrier. In a CTA
            # of one or two tiles, the case this is for, they lie on its critical path. This
            # warp asks L2 for both before that barrier, so those reads can hit L2. A
            # prefetch is a hint: it writes no register and no shared memory and changes no
            # value a later load returns, and the softmax still reads every value with its
            # own loads after its own `griddepcontrol.wait`.
            if cutlass.const_expr(lp_tail and self.vmax_hist_vec):
                with cute.arch.elect_one():
                    for n in cutlass.range_constexpr(blk_tile_n):
                        _l2_prefetch(mQraw[
                            ((cutlass.min(coord_hg * blk_tile_h + (n % blk_tile_h),
                                          grouped_heads - 1),
                              cutlass.min(coord_p * blk_tile_p + (n // blk_tile_h),
                                          prediction - 1)),
                             None, (coord_hk, coord_b))
                        ].iterator.toint(), blk_tile_d * mQraw.element_type.width // 8)
                    _l2_prefetch(mVMaxHist.iterator.toint(), VMAX_BINS * Uint32.width // 8)
        init_warp += 1

        ##############################
        # Tmem Allocation
        ##############################
        tmem_alloc_cols = self.tmem_alloc_cols
        tmem_ptr_smem_ptr = smem.allocate_array(Int32)
        # THE EXPONENT CELL `sLp` (`_pool_pscale_lp`): the 4-byte tmem pointer above
        # (after the two 4-byte varlen scalars, or first) leaves a 4-byte gap before
        # the first 8-byte mbarrier array, and the cell fills it, so no address moves
        # and the shared-memory budget that sets `kv_stages` needs no share for it.
        # - A persistent CTA: the prologue writes it (below), and the softmax and
        #   correction warpgroups read it once per item where they are dispatched.
        # - A one-item CTA under channel-wise K: the softmax warpgroup writes it after
        #   its table read, and both warpgroups read it at each use (`lp_cell`), the
        #   softmax per tile where it quantizes P and the correction at its epilogue,
        #   so neither holds lp in a register across its tile loop. Held there (and
        #   with the correction reading the table itself), the INT4 64/8
        #   channel-wise-K, token-wise-V split build runs 3.4% slower at B = 8,
        #   against 0.3% with the cell.
        # - A one-item (split) CTA under token-wise K where `lp_tail` holds: the softmax warpgroup
        #   writes it after its table read and keeps its own copy in a register; the
        #   correction warpgroup reads it once, at the start of its tail.
        # A persistent CTA's warpgroups reading the cell at each use as well measured
        # up to 1.6% faster than the per-item read on some builds and up to 1.7%
        # slower on others, so they keep the per-item read. Any other one-item CTA
        # under token-wise K has no cell: both its warpgroups read the table.
        sLp = None
        if cutlass.const_expr(reads_lp and (persistent or k_channel or lp_tail)):
            sLp = smem.allocate_tensor(Float32, cute.make_layout(1))
        if warp_idx == init_warp:
            cute.arch.alloc_tmem(tmem_alloc_cols, tmem_ptr_smem_ptr)
        init_warp += 1
        # A persistent CTA: the descriptor-prefetch warp takes the exponent from the
        # table into `sLp` once, beside the tmem allocation; the first item's set-up
        # barrier publishes it to every warp, which read it per item, and it never
        # changes after. A one-item CTA has no read here: its softmax warpgroup reads
        # the table (`lp_table`) while the first tile is being produced, and so does
        # its correction warpgroup under token-wise K, because a read here, before
        # the set-up barrier, puts the table's cold-L2 latency in front of every
        # role: 2% at the one-wave INT2 split cells and all of the INT4 cost.
        lp_table = None
        if cutlass.const_expr(reads_lp and persistent):
            if warp_idx == lp_warp:
                cute.arch.griddepcontrol_wait()
                lp = self._pool_pscale_lp(self._vmax_hist_loads(mVMaxHist, lane_idx),
                                          lane_idx, lp_shift)
                if lane_idx == 0:
                    sLp[0] = lp
        if cutlass.const_expr(reads_lp and not persistent):
            lp_table = mVMaxHist

        ##############################
        # Shared-memory carve-out
        ##############################
        # ALL SmemAllocator calls live here, above the item loop: the DSL
        # flattener cannot carry the stateful allocator object through a
        # dynamic `for`, and the loop body only needs the resulting
        # pointers/tensors (loop-invariant). Allocation ORDER and alignment
        # exactly mirror the original inline sequence, so every smem
        # address is byte-identical for all modes (flag-off bit gates).
        if cutlass.const_expr(do_atomic_red):
            reduction_mbars_ptr = smem.allocate_array(
                Int64, max_reduction_iters * 2
            )
        q_load_mbar = smem.allocate_array(Int64, 1)
        kv_stages_co = smem_layout_k_tma.shape[-1]
        k_stages_co = (
            self.k_op_stages
            if self.k_op_stages is not None
            else min(2, kv_stages_co // 2)
        )
        # Trace-time bound (the ctor cannot see kv_stages): each split
        # operand ring needs >= 2 stages.
        assert 2 <= k_stages_co <= kv_stages_co - 2, (
            f"operand ring split {k_stages_co}/{kv_stages_co - k_stages_co} "
            f"needs >= 2 stages per ring (kv_stages={kv_stages_co}); tune "
            f"k_op_stages or free smem")
        # Split operand rings: K, then V, kv_stages*2 Int64 in total.
        k_pipeline_ptr = smem.allocate_array(Int64, k_stages_co * 2)
        v_pipeline_ptr = smem.allocate_array(
            Int64, (kv_stages_co - k_stages_co) * 2
        )
        # Split code rings: two independent mbarrier blocks, K then V,
        # code_stages*2 Int64 in total (code_stages_k + code_stages_v ==
        # code_stages).
        codek_pipeline_ptr = smem.allocate_array(
            Int64, self.code_stages_k * 2
        )
        codev_pipeline_ptr = smem.allocate_array(
            Int64, self.code_stages_v * 2
        )
        scale_pipeline_ptr = smem.allocate_array(
            Int64, self.scale_stages * 2
        )
        if cutlass.const_expr(k_channel):
            qp_pipeline_ptr = smem.allocate_array(Int64, qp_stages * 2)
            qb_pipeline_ptr = smem.allocate_array(Int64, qp_stages * 2)
        s_pipeline_ptr = smem.allocate_array(Int64, self.s_stages * 2)
        p_pipeline_ptr = smem.allocate_array(Int64, self.p_stages * 2)
        o_pipeline_ptr = smem.allocate_array(Int64, o_ring * 2)
        sSP_i32 = None
        if cutlass.const_expr(enable_blasst):
            sp_pipeline_ptr = smem.allocate_array(Int64, self.sp_stages * 2)
            sSP_i32 = smem.allocate_tensor(
                Int32, cute.make_layout(self.sp_stages)
            )
        tAsK = smem.allocate_tensor(
            k_dtype, smem_layout_k_mma.outer, stensor_align,
            smem_layout_k_mma.inner
        )  # (MMA, #MMA_M, #MMA_K, kv_stages)
        tBsQ = smem.allocate_tensor(
            q_dtype, smem_layout_q.outer, stensor_align, smem_layout_q.inner
        )  # (MMA, #MMA_N, #MMA_K, q_stages)
        sCode = smem.allocate_tensor(
            Uint8,
            cute.make_layout(
                (page_size * mma_tile_k * bits // 8, self.code_stages)
            ),
            stensor_align,
        )  # flat (stage_bytes, code_stages)
        sScale = None
        if cutlass.const_expr(scale_ring_elems > 0):
            sScale = smem.allocate_tensor(
                self.scale_dtype,
                cute.make_layout((scale_ring_elems, self.scale_stages)),
                svector_align,
            )
        tBsP_nm_layout_co = sm100_utils.make_smem_layout_b(
            tiled_mma_vp,
            (None, mma_tile_n, mma_tile_m * tiles_sm),
            mma_dtype,
            self.p_stages,
        )
        tBsP_nm = smem.allocate_tensor(
            mma_dtype, tBsP_nm_layout_co.outer, stensor_align,
            tBsP_nm_layout_co.inner
        )
        sPT = smem.allocate_tensor(
            Int32, cute.make_layout((pages_s, self.pt_stages)), svector_align
        )
        sM = smem.allocate_tensor(
            acc_dtype, cute.make_layout(blk_tile_n), svector_align
        )
        sL = smem.allocate_tensor(
            acc_dtype,
            cute.make_layout((blk_tile_n, warpgroup_warps)),
            svector_align,
        )
        sC = None
        if cutlass.const_expr(fold_p):
            sC = smem.allocate_tensor(
                acc_dtype,
                cute.make_layout((blk_tile_n, warpgroup_warps)),
                svector_align,
            )
        # The channel-K builder needs no cross-warp amax slots and no
        # `kam_mbar` here. It would if ONE po2 lift covered the whole [g, D]
        # plane, whose rows live in different warps, but the lift is PER ROW
        # (see `build_kgroup`) and a row is owned by exactly one warp, so the
        # reduction is warp-local.

        # channel-V: pre-reduced [colsum | l] smem ring — the softmax WG
        # warp-reduces each virtual tile's nL-vector into its warp column,
        # the correction WG sums the 4 columns (fixed order, deterministic).
        sLC = None
        if cutlass.const_expr(v_channel):
            sLC = smem.allocate_tensor(
                acc_dtype,
                cute.make_layout(
                    (2 * blk_tile_n, warpgroup_warps, lc_stages)
                ),
                svector_align,
            )
            lc_pipeline_ptr = smem.allocate_array(Int64, lc_stages * 2)
        # channel-K: Q'-plane ring (QK MMA B-operand layout x qp_stages) +
        # bias ring x qp_stages. The bias-ring row layout is
        #   rows [0, g)   : the PER-ROW lift 2^m_G[h]
        #   rows [g, 2g)  : bias_G[h]
        # It was (1 + g) with a single shared lift at row 0; per-row lifts are
        # what let the amax reduction stay warp-local. Consumers must index the
        # lift by h, not read row 0 -- see the K-fold in the softmax WG.
        sQP = None
        sQB = None
        if cutlass.const_expr(k_channel):
            smem_layout_qp = sm100_utils.make_smem_layout_b(
                tiled_mma_kq, mma_tile_mnk, mma_dtype, qp_stages
            )
            sQP = smem.allocate_tensor(
                mma_dtype, smem_layout_qp.outer, stensor_align,
                smem_layout_qp.inner
            )
            sQB = smem.allocate_tensor(
                acc_dtype,
                cute.make_layout((2 * blk_tile_n, qp_stages)),
                svector_align,
            )
        if cutlass.const_expr(do_atomic_red):
            sR = smem.allocate_tensor(
                acc_dtype,
                cute.make_layout((blk_tile_n, max_reduction_iters, 2)),
                svector_align,
            )

        ##############################
        # Persistent item loop
        ##############################
        # Static round-robin over items bidz, bidz + grid_z_rt, ...: the
        # ENTIRE per-item flow below, INCLUDING the mbarrier/pipeline init
        # section, re-runs per item. At the end-of-item boundary sync every
        # ring is drained and every mbarrier quiesced, so re-initializing
        # them per item is race-free and resets all pipeline phase state
        # to 0 — each item executes the certified per-kernel ring
        # protocols verbatim (no cross-item slot/parity carry anywhere).
        # tmem allocation, TMA descriptor prefetch, and dependent-grid
        # launch are hoisted out (the amortized prologue). Non-persistent
        # compiles trace items_this_cta = 1 (constant trip-1 loop).
        items_this_cta = 1
        if cutlass.const_expr(persistent):
            num_items = cute.size(tiles_hb)
            items_this_cta = cute.ceil_div(num_items - bidz, grid_z_rt)
        for _item in cutlass.range(items_this_cta, unroll=1):
            # Body-local init-warp bookkeeping: reassigning a name captured
            # from outside a dynamic `for` makes it loop-carried IR, which
            # breaks the trace-time warp-budget assert below.
            init_warp_l = init_warp
            # A persistent build chooses the warp roles afresh for every item, at both widths:
            # - One fresh read of %tid.x per item gives the warp index and the warpgroup index
            #   that every role choice and init-warp test in the item reads, the reduction warp's
            #   included, so no role index lives across the item loop's back edge
            #   (`_tid_x_fresh` says why that matters). The mask (1023, the largest thread
            #   index a CTA can have) changes no value and tells the compiler the read is not
            #   negative, so both divisions are shifts rather than signed floor divisions.
            # - Each of the four warp checks of the role dispatch below compares its own opaque
            #   copy of the warp index (`_opaque_i32` says why).
            # The warp's place in its warpgroup (`warpgroup_widx`), which the softmax and
            # correction roles receive, and the warp index the epilogue compares after the loop
            # still come from the kernel-entry read: making them fresh too made some persistent
            # INT4 builds slower. A non-persistent build runs the loop once and keeps the
            # kernel-entry values, so its binary is unchanged.
            if cutlass.const_expr(persistent):
                tid_fresh = _tid_x_fresh() & 1023
                warp_idx_l = cute.arch.make_warp_uniform(tid_fresh // warp_threads)
                warpgroup_idx_l = cute.arch.make_warp_uniform(tid_fresh // warpgroup_threads)
                _role_w = _opaque_i32
            else:
                warp_idx_l, warpgroup_idx_l = warp_idx, warpgroup_idx
                _role_w = (lambda w: w)
            # Names bound ONLY inside a `cutlass.const_expr` guard below, declared
            # here so they can be PASSED to a warp-role method. Inside one function
            # this was unnecessary: every read sat behind the same guard, so the DSL
            # pruned the branch and never evaluated the name. A call argument is
            # evaluated unconditionally, so an unbound name raises "free variable
            # referenced before assignment in enclosing scope" instead. Same idiom
            # the file already uses for `codek_producer`, and the same reason
            # `sSP_i32` and `sp_consumer` carry it.
            #
            # Some of these are bound in BOTH arms of an if/else and so are always
            # live; they are listed anyway because "assigned under a conditional" is
            # what the check can see, and a redundant `= None` costs nothing.
            #
            # NOT LISTED, and it matters: `sCode` and `correction_regs` are bound
            # BEFORE this loop (the smem carve-out and the register split). A
            # `= None` here would SHADOW the real value rather than pre-declare it,
            # and it did -- `'NoneType' object has no attribute 'iterator'`. Only
            # names bound INSIDE the loop body belong in this block; a name bound
            # above the loop needs its pre-init above the loop, or none at all when
            # every arm of its guard assigns it, which is the case for both of these.
            code_stage_bytes = None
            page_count = None
            sL_final_nbar = None
            sM_consumer_nbar = None
            sM_final_nbar = None
            sM_producer_nbar = None
            sO_final_nbar = None
            sSP_i8 = None
            seqlen = None
            tCtL = None
            tL_consumer_nbar = None
            tL_producer_nbar = None
            table_offset = None
            if cutlass.const_expr(persistent):
                item_lin = bidz + _item * grid_z_rt
                coord_hb = cute.idx2crd(item_lin, tiles_hb)
                coord_hk, coord_b = coord_hb
                # Per-item seqlen/table-offset staging (varlen): same
                # protocol and warp as the (persistent-disabled) pre-loop
                # block; all-warp visibility comes from the init sync
                # below, exactly like the non-persistent path.
                if cutlass.const_expr(is_varlen):
                    if warp_idx_l == 1:
                        seqlen_gmem = cute.make_tensor(
                            seqlens_iter + coord_b, scalar_layout
                        )
                        table_offset_gmem = cute.make_tensor(
                            table_offsets_iter + coord_b, scalar_layout
                        )
                        cute.arch.griddepcontrol_wait()
                        with cute.arch.elect_one():
                            cute.copy(
                                cpasync_atom, seqlen_gmem, seqlen_smem
                            )
                            cute.copy(
                                cpasync_atom,
                                table_offset_gmem,
                                table_offset_smem,
                            )
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
            ##############################
            # Pipeline Allocation + Init
            ##############################
            # Initialize named barriers
            softmax_threads = warpgroup_threads
            correction_threads = warpgroup_threads
            reduction_threads = warp_threads
            mma_threads = warp_threads
            tma_threads = warp_threads
            # Shared KV pipeline requires ordering MMA + TMA
            # Prefer to keep MMA/TMA in separate warps even if we order their execution
            # Compiler optimization is easier with less warp uniform register pressure
            # Small pages increase pressure on TMA warps (More TMAs per unrolled block tile)
            # Large head_dim increases pressure on MMA warps (More MMAs per unrolled block tile)
            tma_order_k_nbar = nbar(1, tma_threads + tma_threads)
            tma_order_v_nbar = nbar(2, tma_threads + tma_threads)
            mma_order_kq_nbar = nbar(3, mma_threads + mma_threads)
            mma_order_vp_nbar = nbar(4, mma_threads + mma_threads)
            if cutlass.const_expr(self.o_stages == 2):
                sM_producer_nbar = nbar(5, softmax_threads + correction_threads)
                sM_consumer_nbar = nbar(6, softmax_threads + correction_threads)
                tL_producer_nbar = nbar(7, softmax_threads + correction_threads)
                tL_consumer_nbar = nbar(9, softmax_threads + correction_threads)
                sM_final_nbar = nbar(11, correction_threads + reduction_threads)
                sL_final_nbar = nbar(12, correction_threads + reduction_threads)
                sO_final_nbar = nbar(13, correction_threads + tma_threads)
            else:
                # o_stages == 3: the tL rings need 3 ids each (with_phase), so
                # ids 7..15 are fully occupied.
                sM_producer_nbar = nbar(5, softmax_threads + correction_threads)
                sM_consumer_nbar = nbar(6, softmax_threads + correction_threads)
                tL_producer_nbar = nbar(7, softmax_threads + correction_threads)
                tL_consumer_nbar = nbar(10, softmax_threads + correction_threads)
                sM_final_nbar = nbar(13, correction_threads + reduction_threads)
                sL_final_nbar = nbar(14, correction_threads + reduction_threads)
                sO_final_nbar = nbar(15, correction_threads + tma_threads)
            # BLASST is out of scope for this kernel (`__call__` asserts
            # `threshold_scale_factor is None`), so this barrier is never
            # synced; it shares id 14 with `sL_final_nbar` at o_stages == 3.
            sSP_producer_nbar = nbar(14, softmax_threads)

            # named barrier stage helper
            def with_phase(nbar_, phase):
                return nbar(nbar_.barrier_id + phase, nbar_.num_threads)

            # Alias thread cooperatives
            thr_cg = lambda t: CooperativeGroup(Agent.Thread, t)
            elect_one_cooperative = thr_cg(1)
            warpgroup_cooperative = thr_cg(warpgroup_threads)
            mma_group = elect_one_cooperative
            tma_group = elect_one_cooperative
            softmax_group = warpgroup_cooperative
            correction_group = warpgroup_cooperative
            # The dequant warps run as a K stream (first half) and a V
            # stream (second half) that deposit CONCURRENTLY; each operand
            # ring's producer commits arrive from its OWN stream's threads
            # only, so the producer group is half the dequant threads.
            dequant_stream_group = thr_cg(warpgroup_threads // 2)
            # FOLD-OFF scale-ring releases come from EVERY dequant warp:
            # BOTH streams wait and release every stage (each reads its own
            # tensor's coefficients from it), so the count is the full
            # dequant warp count.
            dequant_scale_release_group = thr_cg(warpgroup_warps)
            # Fold-on scale-ring releases come from the softmax warpgroup's
            # four warps (lane 0 of each), so the group is one warpgroup's
            # warp count.
            softmax_release_group = thr_cg(warpgroup_warps)

            # Initialize cluster colmax + colsum mbar (even if this split exits early)
            if cutlass.const_expr(do_atomic_red):
                if warp_idx_l == init_warp_l:
                    if lane_idx < max_reduction_iters * 2:
                        mbar_ptr = reduction_mbars_ptr + lane_idx
                        arrive_count = 1
                        expect_tx_bytes = blk_tile_n * acc_dtype.width // 8
                        cute.arch.mbarrier_init(mbar_ptr, arrive_count)
                        cute.arch.mbarrier_init_fence()
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, expect_tx_bytes)
                init_warp_l += 1
                cute.arch.cluster_arrive_relaxed()

            # Initialize Q load mbarrier
            if warp_idx_l == init_warp_l:
                expect_tx_bytes = cute.size_in_bytes(q_dtype, smem_layout_q)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(q_load_mbar, 1)
                    cute.arch.mbarrier_init_fence()
                    cute.arch.mbarrier_arrive_and_expect_tx(q_load_mbar, expect_tx_bytes)
            init_warp_l += 1

            # Initialize pipelines
            # KV/operand ring: produced by the dequant WG (async threads) instead
            # of TMA; the MMA consumer side is byte-for-byte the baseline's.
            # Producer discipline mirrors the P pipeline below (STS ->
            # fence_view_async_shared -> commit).
            kv_stages = smem_layout_k_tma.shape[-1]
            kv_stage_bytes = mma_tile_m * mma_tile_k * k_dtype.width // 8
            # SPLIT K/V operand rings. The physical kv_stages ring is
            # partitioned into an independent K ring (stages [0, k_stages))
            # and V ring (stages [k_stages, kv_stages)) with separate
            # mbarrier pairs and credits, so a K-panel slot acquire in the
            # dequant WG only ever waits on MMA-KQ releases (hungry, fast)
            # and never queues behind the V ring's VP(s)<-P(s)<-softmax
            # credit chain. Barrier storage is kv_stages*2 Int64 in total.
            # DSL WARNING: a pipeline participant must be referenced through
            # exactly ONE name inside the dynamic loops. A second alias (Python
            # ternary or const-if rebind) makes the loop-carry machinery clone
            # its index/phase state per name, the two walkers desync, and the
            # device deadlocks. Hence every acquire and wait below goes
            # through the one participant name.
            # The int2 split point is swept on silicon, and it moved when the
            # deposits became concurrent: 3/6 was best while they were
            # serialized, and 2/7 is best now. The K stream keeps pace with
            # the hungry MMA-KQ on 1 tile of K buffering, and the V ring
            # wants every spare stage of slack against the VP<-P<-softmax
            # credit chain. INT4 takes 3 through `fold_runtime.BITS_TUNING`.
            k_stages = (
                self.k_op_stages
                if self.k_op_stages is not None
                else min(2, kv_stages // 2)
            )
            v_stages = kv_stages - k_stages
            assert k_stages >= 2 and v_stages >= 2, (
                f"split K/V rings need >= 2 stages each; got kv_stages="
                f"{kv_stages} -> {k_stages}/{v_stages}"
            )
            # Producer group: each concurrent stream commits only its own
            # ring, so half the dequant threads.
            op_producer_group = dequant_stream_group
            k_producer, k_consumer = pipeline.PipelineAsyncUmma.create(
                num_stages=k_stages,
                producer_group=op_producer_group,
                consumer_group=mma_group,
                barrier_storage=k_pipeline_ptr,
                defer_sync=True,
            ).make_participants()
            v_producer, v_consumer = pipeline.PipelineAsyncUmma.create(
                num_stages=v_stages,
                producer_group=op_producer_group,
                consumer_group=mma_group,
                barrier_storage=v_pipeline_ptr,
                defer_sync=True,
            ).make_participants()

            # Code rings: the TMA warps produce one page of packed K/V bytes
            # per stage, and the dequant streams consume them. K is produced
            # by the TMA-QK warp and consumed and released by the dequant K
            # stream; V by TMA-VO and the V stream. Same elect-one TMA
            # producer group; the release group is the owning stream's two
            # warps.
            code_stage_bytes = page_size * mma_tile_k * bits // 8  # 1 page
            codek_producer, codek_consumer = pipeline.PipelineTmaAsync.create(
                num_stages=self.code_stages_k,
                producer_group=tma_group,
                # Every thread of the owning stream releases its stages
                # (`release_code_stage`), so the empty barrier counts that
                # stream's threads. Each code ring is released by its own stream
                # only, and every wait a stream performs is on a stage its own
                # releases gate, so the TMA producer can never lap a waiter. A
                # cross-stream release was tried: correct, but it re-coupled the
                # stage refills to the slower stream and cost measurably more.
                consumer_group=dequant_stream_group,
                tx_count=code_stage_bytes,
                barrier_storage=codek_pipeline_ptr,
                defer_sync=True,
            ).make_participants()
            codev_producer, codev_consumer = pipeline.PipelineTmaAsync.create(
                num_stages=self.code_stages_v,
                producer_group=tma_group,
                consumer_group=dequant_stream_group,  # as the K ring
                tx_count=code_stage_bytes,
                barrier_storage=codev_pipeline_ptr,
                defer_sync=True,
            ).make_participants()

            # Scale ring: per-page f32 [64, 4] scale blocks, produced by the
            # TMA-QK warp alongside K, consumed by the softmax WG, whose
            # prefetch waits a tile's stages one tile ahead of that tile's
            # S-wait, copies the quads to registers and releases them at once
            # (the K-fold and the P-quant of tile i share that register copy,
            # not a stage). Issuance and consumption must always land
            # together, because a produced-never-consumed ring deadlocks the
            # producer at stage scale_stages+1.
            # FOLD-OFF (fold_scale_zero=False): the CONSUMERS are the two
            # dequant streams instead, the warps that apply (s, z) there, so
            # the release group follows (releases arrive from lane 0 of each
            # consumer warp, and every dequant warp releases every stage:
            # dequant_scale_release_group).
            scale_stages = self.scale_stages
            scale_producer = scale_consumer = None
            if cutlass.const_expr(scale_ring_elems > 0):
                scale_stage_bytes = (
                    scale_ring_elems * self.scale_dtype.width // 8
                )
                scale_producer, scale_consumer = (
                    pipeline.PipelineTmaAsync.create(
                        num_stages=scale_stages,
                        producer_group=tma_group,
                        consumer_group=(
                            dequant_scale_release_group if nofold
                            else softmax_release_group
                        ),
                        tx_count=scale_stage_bytes,
                        barrier_storage=scale_pipeline_ptr,
                        defer_sync=True,
                    ).make_participants()
                )

            # channel-K rings: the TMA-QK warp builds per-group Q' planes
            # (consumed by the QK MMA under the kv-ring UMMA-commit release
            # discipline) and (2^m, bias[h]) entries (consumed by the softmax
            # WG's fold-coefficient reads).
            qp_producer = qp_consumer = None
            qb_producer = qb_consumer = None
            if cutlass.const_expr(k_channel):
                # arrive counts == CALLING threads: the whole CORRECTION WG
                # runs the builder (commits x128); every softmax thread
                # releases the bias ring; the plane ring's consumer release is
                # the UMMA-commit mechanism (count 1).
                qp_producer, qp_consumer = pipeline.PipelineAsyncUmma.create(
                    num_stages=qp_stages,
                    producer_group=warpgroup_cooperative,
                    consumer_group=mma_group,
                    barrier_storage=qp_pipeline_ptr,
                    defer_sync=True,
                ).make_participants()
                qb_producer, qb_consumer = pipeline.PipelineAsync.create(
                    num_stages=qp_stages,
                    producer_group=warpgroup_cooperative,
                    consumer_group=warpgroup_cooperative,
                    barrier_storage=qb_pipeline_ptr,
                    defer_sync=True,
                ).make_participants()

            s_stages = self.s_stages
            s_producer, s_consumer = pipeline.PipelineUmmaAsync.create(
                num_stages=self.s_stages,
                producer_group=mma_group,
                consumer_group=softmax_group,
                barrier_storage=s_pipeline_ptr,
                defer_sync=True,
            ).make_participants()

            p_stages = self.p_stages
            p_producer, p_consumer = pipeline.PipelineAsyncUmma.create(
                num_stages=self.p_stages,
                producer_group=softmax_group,
                consumer_group=mma_group,
                barrier_storage=p_pipeline_ptr,
                defer_sync=True,
            ).make_participants()

            o_stages = self.o_stages
            o_producer, o_consumer = pipeline.PipelineUmmaAsync.create(
                num_stages=o_ring,
                producer_group=mma_group,
                consumer_group=correction_group,
                barrier_storage=o_pipeline_ptr,
                defer_sync=True,
            ).make_participants()

            # channel-V: pre-reduced [colsum | l] smem ring (softmax warps
            # produce their per-warp reductions, the correction WG consumes)
            lc_producer = lc_consumer = None
            if cutlass.const_expr(v_channel):
                # every softmax thread commits; every correction thread
                # releases (AsyncThread arrives are per CALLING thread)
                lc_producer, lc_consumer = pipeline.PipelineAsync.create(
                    num_stages=lc_stages,
                    producer_group=warpgroup_cooperative,
                    consumer_group=warpgroup_cooperative,
                    barrier_storage=lc_pipeline_ptr,
                    defer_sync=True,
                ).make_participants()

            sp_producer = sp_consumer = None
            if cutlass.const_expr(enable_blasst):
                skip_group = thr_cg(correction_threads + (mma_threads + tma_threads) * 2)

                sp_stages = self.sp_stages
                sp_producer, sp_consumer = pipeline.PipelineAsync.create(
                    num_stages=sp_stages,
                    producer_group=softmax_group,
                    consumer_group=skip_group,
                    barrier_storage=sp_pipeline_ptr,
                    defer_sync=True,
                ).make_participants()

                # SP - skip predicates
                sSP_i8 = cute.make_tensor(
                    cute.recast_ptr(sSP_i32.iterator, dtype=Int8),
                    cute.make_layout((warpgroup_warps, sp_stages)),
                )

            ##############################
            # Smem Tensor Allocation
            ##############################
            # Threadblock slice
            thrblk_mma_kq = tiled_mma_kq.get_slice(0)
            thrblk_mma_vp = tiled_mma_vp.get_slice(0)

            # Q, K, V (rings allocated in the smem carve-out above)
            tAsV = cute.make_tensor(
                tAsK.iterator, smem_layout_v_mma.outer
            )  # (MMA, #MMA_M, #MMA_K, kv_stages)
            # The view above shares the ring iterator's swizzle, which is K's 128-byte
            # one, and the V layout is MN_SW128 at every width, so the two agree. The
            # PV MMA descriptor follows from `tAsV`, and the INT4 V deposit, which
            # addresses the ring bytes directly (`deposit_lane_offsets`), asserts that
            # this view's swizzle spans its token row. A V view whose swizzle is not
            # its layout's would feed the tensor core an image the deposits did not
            # write.
            assert (tAsV.iterator.type.swizzle_type
                    == smem_layout_v_mma.inner.type), (
                f"the V view's swizzle {tAsV.iterator.type.swizzle_type} is not the "
                f"V layout's {smem_layout_v_mma.inner.type}")
            # Code ring: flat row-major uint8 page stages, no swizzle, with
            # the K and V views aliasing the same ring bytes.
            # S
            # (MMA_MN, #MMA_M=1, #MMA_N=1, #TILE_SM, s_stages)
            tCtS_shape = tiled_mma_kq.partition_shape_C(
                (mma_tile_m, mma_tile_n, tiles_sm, s_stages)
            )
            tCtS = thrblk_mma_kq.make_fragment_C(tCtS_shape)
            tCtS2 = None
            if cutlass.const_expr(dual_s):
                tCtS2 = thrblk_mma_kq.make_fragment_C(tCtS_shape)

            # P - Treat MN C tile of BMM0 as NM B tile of BMM1
            # (MMA_NK, #MMA_N=1, #MMA_K=TILE_S/MMA_K, p_stages)
            blk_tile_nm = (None, mma_tile_n, mma_tile_m * tiles_sm)
            tBsP_nm_layout = sm100_utils.make_smem_layout_b(
                tiled_mma_vp, blk_tile_nm, mma_dtype, p_stages
            )
            # Tile for NK B tile iteration
            tBsP_nk_tile = thrblk_mma_vp.partition_shape_B(
                (mma_tile_n, mma_tile_k)
            )  # (MMA_NK, #MMA_N=1, #MMA_K=MMA_TILE_K/MMA_K, #TILE_SK=TILE_S/MMA_TILE_K, p_stages)
            tBsP_nk = cute.local_tile(tBsP_nm, tBsP_nk_tile, (0, 0, None, None))

            # Reshape NM B tile of BMM1 to become MN C tile of BMM0
            # (MMA_NK, #MMA_N, #MMA_K=TILE_S/MMA_K, p_stages) ->
            # (MMA_MN, #MMA_M, #MMA_N, #TILE_SM, p_stages)
            tCsP_tile = cute.make_ordered_layout(tCtS_shape, order=((2, 0), 3, 1, 4, 5))
            tCsP = cute.composition(tBsP_nm, tCsP_tile)

            # O
            # Reuse KV smem for O TMA store
            sO_iterator = cute.recast_ptr(tAsK.iterator, smem_layout_o.inner, dtype=o_dtype)
            # (MMA_TILE_M, MMA_TILE_N, #TILE_DM, #TILE_HN)
            sO_mma = cute.make_tensor(sO_iterator, smem_layout_o.outer)
            # (MMA, #MMA_M, #MMA_N, #TILE_DM, #TILE_HN=1)
            tCsO = thrblk_mma_vp.partition_C(sO_mma)
            tCsO = tCsO[mma_dice + (None, 0)]
            # Where `o_direct_for` holds: this item's tile of the output (of a split's partial
            # output in a split build) in global memory (the tile the TMA store would write,
            # on the plain-stride view) and a coordinate tensor of the same shape, both
            # partitioned as sO is, so that the correction knows the global row d, query
            # head h and token p of every element it holds. The bounds are the valid query
            # heads and tokens of the tile; the tile is exactly head_dim wide.
            o_gmem = None
            if cutlass.const_expr(mOraw is not None):
                gO_d = cute.local_tile(
                    mOraw,
                    tiler=(blk_tile_d, blk_tile_hp),
                    coord=(0, coord_hp, (coord_hk, kv_split_idx * batches + coord_b
                                         if do_kernel_red else coord_b)),
                )
                cO_d = cute.make_identity_tensor(gO_d.shape)
                tCgO_d = thrblk_mma_vp.partition_C(cute.flat_divide(gO_d, (mma_tile_m, mma_tile_n)))
                tCcO_d = thrblk_mma_vp.partition_C(cute.flat_divide(cO_d, (mma_tile_m, mma_tile_n)))
                o_gmem = (
                    tCgO_d[mma_dice + (None, 0)],
                    tCcO_d[mma_dice + (None, 0)],
                    (grouped_heads - coord_hg * blk_tile_h, prediction - coord_p * blk_tile_p),
                )
            # (MMA, #MMA_M, #MMA_N, #TILE_DM, o_stages)
            tCtO = thrblk_mma_vp.make_fragment_C((*tCsO.shape, o_ring))

            # PT - Page Table lookup buffer
            pt_stages = self.pt_stages
            sPT_layout = cute.make_layout((pages_s, pt_stages))

            # M - colmax
            sM_layout = cute.make_layout(blk_tile_n)
            lane_store_max = blk_tile_n == warp_threads or lane_idx < blk_tile_n
            if warp_idx_l == init_warp_l:
                if lane_store_max:
                    sM[lane_idx] = -Float32.inf
            init_warp_l += 1

            # `sM`'s producer/consumer window is what serves the correction
            # warpgroup; there is deliberately no separate tile-parity colmax
            # buffer.
            # L - colsum
            sL_layout = cute.make_layout((blk_tile_n, warpgroup_warps))
            if warp_idx_l == init_warp_l:
                for i in cutlass.range_constexpr(0, cute.size(sL), warp_threads):
                    if i + lane_idx < cute.size(sL):
                        sL[i + lane_idx] = Float32(0)
            init_warp_l += 1

            # Fold stats. There is no sVmax running-max cell: pscale is
            # the pool-wide exponent (`_pool_pscale_lp`), so softmax and
            # correction use the SAME constant power-of-2 pscale with no window
            # and no per-tile log2. sC = c_h cross-warp reduction slots for the
            # correction WG's final combine: PER-WARP columns (sL's shape)
            # written with plain stores and summed in fixed warp order after the
            # sL_final nbar. DETERMINISM INVARIANT (kernel-reduction mode is the
            # deterministic path by design): relaxed f32 atomic adds into a
            # g-wide strip flipped roughly 5-16 of 4096 output words by 1 ulp per
            # launch, from warp-order-dependent f32 rounding. Do NOT replace the
            # ordered sum with atomics.
            if cutlass.const_expr(fold_p):
                if warp_idx_l == init_warp_l:
                    if lane_idx < cute.size(sC):
                        sC[lane_idx] = Float32(0)
                init_warp_l += 1

            # NOTE: the channel-K amax-round mbarrier (128-arrive, correction
            # WG's cross-warp max for the Q'-plane lift) was initialized here.
            # The lift is per row and rows are warp-private, so there is no
            # cross-warp round left to name. `init_warp_l` is deliberately NOT
            # advanced for it any more -- the barrier does not exist.

            # per-thread colsum
            # (MMA_MN, #MMA_M=1, #MMA_N=1, o_stages)
            if cutlass.const_expr(not v_channel):
                tCtL_shape = tiled_mma_kq.partition_shape_C(
                    (mma_tile_m, mma_tile_n, o_stages)
                )
                tCtL = thrblk_mma_kq.make_fragment_C(tCtL_shape)

            # R - cluster reduction buffers for colmax + colsum
            if cutlass.const_expr(do_atomic_red):
                sR_layout = cute.make_layout((blk_tile_n, max_reduction_iters, 2))
                assert sR is not None  # allocated in the smem carve-out

            ##############################
            # Sync
            ##############################
            # Ensure visibility of cluster mbarriers
            if cutlass.const_expr(do_atomic_red):
                cute.arch.cluster_wait()

            # The previous item's O store may still be reading sO, which is K-ring
            # stage 0, and the K deposit that follows this barrier overwrites it (where
            # `o_direct_for` holds there is no O store, and so no wait). So
            # the TMA-VO warp, whose lanes committed that store, waits for the read
            # here, where it overlaps the item boundary and the initialization above
            # rather than delaying the boundary. Every thread runs the wait, which
            # has nothing to wait for in a thread that committed no bulk group, so
            # that no warp test is added to the item loop. At the first item nothing
            # is pending.
            if cutlass.const_expr(persistent and tma_atom_o is not None):
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            # Ensure visibility of local mbarrier inits, table offset async load, tmem alloc
            cute.arch.sync_threads()
            assert init_warp_l <= (self.threads_per_cta // warp_threads), (
                f"used {init_warp_l} init warps, {self.threads_per_cta // warp_threads} warps available"
            )

            # Runtime control flow
            if cutlass.const_expr(is_varlen):
                seqlen = seqlen_smem[0]
                page_count = cute.ceil_div(seqlen, page_size)
                table_offset = table_offset_smem[0]
            else:
                seqlen = seqlens_iter
                page_count = cute.ceil_div(seqlen, page_size)
                table_offset = coord_b * page_count
            tiles_s = cute.ceil_div(seqlen, blk_tile_s)
            iters_s = cute.ceil_div(tiles_s - kv_split_idx, kv_splits)
            exit_early = kv_split_idx >= tiles_s
            prefetch_iters = min(2, s_stages - 1)  # MMA KQ iters to hide first softmax
            assert pt_stages > prefetch_iters + 1

            ##############################
            # Tmem tensor allocation
            ##############################
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Int32, 16, tmem_ptr_smem_ptr)
            tmem_offset = 0

            tCtS = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtS.layout
            )
            tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtS)
            # dual-S second accumulator (K token-64 hi chain / K channel-64
            # odd-group chain)
            if cutlass.const_expr(dual_s):
                tCtS2 = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype),
                    tCtS2.layout,
                )
                tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtS2)

            # tL channel: allocated whenever the transport is tmem, which is
            # every configuration except channel-V (pre-reduced smem ring).
            if cutlass.const_expr(not v_channel):
                tCtL = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtL.layout
                )
                tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtL)

            tCtO = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtO.layout
            )
            tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtO)

            assert tmem_offset <= tmem_alloc_cols, (
                f"\t{tmem_offset} tmem cols used, {tmem_alloc_cols} tmem cols allocated"
            )

            ##############################
            # Exit early
            ##############################
            # PERSISTENT ITEM CLOSURE for an item with no tile (a zero-length
            # request). A persistent CTA carries every named barrier's count into
            # its next item, so an empty item must leave each one as a full item
            # does (`_close_empty_item`): it issues the one-time ordering primes
            # when it is the CTA's first item, because every later item relies on
            # them, and the correction warpgroup meets `reduction_none`'s three
            # final barriers (sM_final, sL_final, sM_final) as it does at a full
            # item's end, because the reduction warp runs all three for every item.
            # A one-item CTA ends with the item, so its arm is unchanged. The empty
            # item still stores no output.
            #
            # The two kinds of build compile two separate chains: a persistent CTA the
            # chain with the closure in each role's branch, a one-item CTA the chain
            # without a TMA-VO branch. One chain for both, with an empty TMA-VO branch
            # in one-item builds, makes some channel-wise split builds at 64/8 heads up
            # to 4.6% slower, and moving that branch out of the chain for both kinds of
            # build costs persistent builds up to 0.9%.
            if exit_early:
                if cutlass.const_expr(persistent):
                    if cutlass.const_expr(compact_close):
                        # `compact_close`: the correction first, with its whole closure,
                        # then one first-item test for the two warps that only prime.
                        if warpgroup_idx_l == correction_warpgroup_id:
                            sM_final_nbar.arrive()
                            self._close_empty_item(
                                _item, "correction", mma_order_vp_nbar, tma_order_v_nbar,
                                sM_consumer_nbar, sL_final_nbar, sM_final_nbar,
                                tL_consumer_nbar, v_channel, with_phase)
                        elif _item == 0:
                            if warp_idx_l == mma_vp_warp_id:
                                mma_order_vp_nbar.arrive()
                            elif warp_idx_l == tma_vo_warp_id:
                                tma_order_v_nbar.arrive()
                    else:
                        if warp_idx_l == mma_vp_warp_id:
                            # In a persistent build tmem stays allocated across
                            # items; the after-loop tail owns the single dealloc.
                            self._close_empty_item(
                                _item, "mma_vp", mma_order_vp_nbar, tma_order_v_nbar,
                                sM_consumer_nbar, sL_final_nbar, sM_final_nbar,
                                tL_consumer_nbar, v_channel, with_phase)

                        elif warp_idx_l == tma_vo_warp_id:
                            self._close_empty_item(
                                _item, "tma_vo", mma_order_vp_nbar, tma_order_v_nbar,
                                sM_consumer_nbar, sL_final_nbar, sM_final_nbar,
                                tL_consumer_nbar, v_channel, with_phase)

                        # The final-handshake arrivals come from the WG that owns the
                        # final colmax/colsum, which is the correction warpgroup.
                        elif warpgroup_idx_l == correction_warpgroup_id:
                            sM_final_nbar.arrive()
                            self._close_empty_item(
                                _item, "correction", mma_order_vp_nbar, tma_order_v_nbar,
                                sM_consumer_nbar, sL_final_nbar, sM_final_nbar,
                                tL_consumer_nbar, v_channel, with_phase)
                else:
                    if warp_idx_l == mma_vp_warp_id:
                        cute.arch.relinquish_tmem_alloc_permit()
                        cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)

                    # The final-handshake arrivals come from the WG that owns the
                    # final colmax/colsum, which is the correction warpgroup.
                    elif warpgroup_idx_l == correction_warpgroup_id:
                        sM_final_nbar.arrive()
                        sL_final_nbar.arrive()

            ##############################
            # TMA QK Dispatch
            ##############################
            elif _role_w(warp_idx_l) == tma_qk_warp_id:
                self._tma_qk_warp(
                    blk_tile_d,
                    blk_tile_hp,
                    codek_producer,
                    coord_hb,
                    coord_hk,
                    coord_hp,
                    cpasync_atom,
                    enable_blasst,
                    iters_s,
                    kv_split_idx,
                    kv_splits,
                    lane_idx,
                    mKC,
                    mQ,
                    mSC,
                    mcast_coord,
                    mcast_layout,
                    mma_tile_k,
                    mma_tile_n,
                    page_count,
                    page_table_iter,
                    pages_m,
                    pages_s,
                    prefetch_iters,
                    q_load_mbar,
                    qflex,
                    sCode,
                    sPT,
                    sScale,
                    scale_producer,
                    scale_ring_elems,
                    scale_ring_live,
                    smem_layout_code_k_tma,
                    sp_consumer,
                    tBsQ,
                    table_offset,
                    thrblk_mma_kq,
                    tiles_dk,
                    tiles_sm,
                    tma_atom_kc,
                    tma_atom_q,
                    tma_atom_sc,
                    tma_order_k_nbar,
                    tma_order_v_nbar,
                )

            ##############################
            # TMA VO Dispatch
            ##############################
            elif _role_w(warp_idx_l) == tma_vo_warp_id:
                self._tma_vo_warp(
                    _item,
                    batches,
                    blk_tile_d,
                    blk_tile_hp,
                    codev_producer,
                    coord_b,
                    coord_hk,
                    coord_hp,
                    enable_blasst,
                    iters_s,
                    kv_split_idx,
                    mO,
                    mVC,
                    mcast_coord,
                    mcast_layout,
                    mma_tile_k,
                    mma_tile_m,
                    mma_tile_n,
                    pages_k,
                    prefetch_iters,
                    sCode,
                    sO_final_nbar,
                    sO_mma,
                    sPT,
                    sSP_i32,
                    smem_layout_code_v_tma,
                    sp_consumer,
                    tiles_dm,
                    tiles_sk,
                    tma_atom_o,
                    tma_atom_vc,
                    tma_order_k_nbar,
                    tma_order_v_nbar,
                )

            ##############################
            # MMA KQ (BMM1) Dispatch
            ##############################
            elif _role_w(warp_idx_l) == mma_kq_warp_id:
                tiled_mma_kq = self._mma_kq_warp(
                    enable_blasst,
                    iters_s,
                    k_channel,
                    k_consumer,
                    k_parts,
                    k_stages,
                    k_tok64,
                    kv_stage_bytes,
                    mma_order_kq_nbar,
                    mma_order_vp_nbar,
                    mma_tile_k,
                    mma_tile_m,
                    prefetch_iters,
                    q_load_mbar,
                    qp_consumer,
                    sQP,
                    s_producer,
                    sp_consumer,
                    tAsK,
                    tBsQ,
                    tCtS,
                    tCtS2,
                    thrblk_mma_kq,
                    tiled_mma_kq,
                    tiles_dk,
                    tiles_sm,
                )

            ##############################
            # MMA VP (BMM2) Dispatch
            ##############################
            elif _role_w(warp_idx_l) == mma_vp_warp_id:
                tiled_mma_vp = self._mma_vp_warp(
                    _item,
                    enable_blasst,
                    iters_s,
                    k_stages,
                    kv_stage_bytes,
                    mma_order_kq_nbar,
                    mma_order_vp_nbar,
                    mma_tile_k,
                    mma_tile_m,
                    o_producer,
                    p_consumer,
                    prefetch_iters,
                    sSP_i32,
                    sp_consumer,
                    tAsK,
                    tAsV,
                    tBsP_nk,
                    tCtO,
                    thrblk_mma_vp,
                    tiled_mma_vp,
                    tiles_dm,
                    tiles_sk,
                    tmem_ptr,
                    v_channel,
                    v_consumer,
                    v_parts,
                    v_stages,
                )

            ##############################
            # Softmax Dispatch
            ##############################
            elif warpgroup_idx_l in softmax_warpgroup_ids:
                self._softmax_warpgroup(
                    acc_dtype,
                    blk_tile_d,
                    blk_tile_h,
                    blk_tile_n,
                    blk_tile_p,
                    blk_tile_s,
                    coord_b,
                    coord_hg,
                    coord_hk,
                    coord_p,
                    dual_s,
                    enable_blasst,
                    flex_ring_w,
                    fold_k,
                    fold_p,
                    grouped_heads,
                    iters_s,
                    k_ch64,
                    k_channel,
                    k_is_token,
                    k_parts,
                    k_ring_w,
                    k_tok64,
                    kv_split_idx,
                    kv_splits,
                    lane_idx,
                    lane_store_max,
                    lc_producer,
                    log2_threshold_scale_factor,
                    mDbgS,
                    mQD,
                    mQraw,
                    mma_dtype,
                    mma_tile_m,
                    mma_tile_n,
                    o_consumer,
                    o_dtype,
                    p_producer,
                    pages_m,
                    pages_s,
                    prediction,
                    pscale_lp if sLp is None or not persistent else sLp[0],
                    lp_table,
                    lp_shift,
                    sLp if not persistent else None,
                    qb_consumer,
                    qflex,
                    sC,
                    sL,
                    sLC,
                    sL_final_nbar,
                    sM,
                    sM_consumer_nbar,
                    sM_final_nbar,
                    sM_producer_nbar,
                    sO_final_nbar,
                    sQB,
                    sSP_i32,
                    sSP_i8,
                    sSP_producer_nbar,
                    sScale,
                    s_consumer,
                    scale_consumer,
                    scale_ring_elems,
                    scale_s_log2_e,
                    seqlen,
                    sp_producer,
                    tCsO,
                    tCsP,
                    tCtL,
                    tCtO,
                    tCtS,
                    tCtS2,
                    tL_consumer_nbar,
                    tL_producer_nbar,
                    tiles_dm,
                    tiles_s,
                    tiles_sm,
                    v_channel,
                    v_is_token,
                    v_parts,
                    v_ring_off,
                    v_ring_w,
                    warpgroup_tidx,
                    warpgroup_widx,
                    with_phase,
                )
            ##############################
            # Correction Dispatch
            ##############################
            elif warpgroup_idx_l == correction_warpgroup_id:
                self._correction_warpgroup(
                    _item,
                    acc_dtype,
                    blk_tile_n,
                    coord_hk,
                    correction_regs,
                    enable_blasst,
                    fold_p,
                    iters_s,
                    k_ch_off,
                    k_channel,
                    k_pages,
                    k_parts,
                    kb_ahead,
                    kch_beta,
                    kch_lift_mul,
                    kch_s_mul,
                    kv_split_idx,
                    kv_splits,
                    lane_idx,
                    lane_store_max,
                    lc_consumer,
                    mQ,
                    mSC_flat,
                    mma_dtype,
                    mma_tile_k,
                    mma_tile_n,
                    o_consumer,
                    o_dtype,
                    o_gmem,
                    o_ring,
                    page_count,
                    page_table_iter,
                    pages_s,
                    pscale_lp if sLp is None or not persistent else sLp[0],
                    lp_table if sLp is None else None,
                    lp_shift,
                    sLp if not persistent else None,
                    q_load_mbar,
                    qb_producer,
                    qp_producer,
                    qp_stages,
                    rebalance_regs,
                    sC,
                    sL,
                    sLC,
                    sL_final_nbar,
                    sM,
                    sM_consumer_nbar,
                    sM_final_nbar,
                    sM_producer_nbar,
                    sO_final_nbar,
                    sQB,
                    sQP,
                    sSP_i32,
                    seqlens_iter,
                    sp_consumer,
                    tBsQ,
                    tCsO,
                    tCtL,
                    tCtO,
                    tL_consumer_nbar,
                    tL_producer_nbar,
                    table_offset,
                    tiles_dk,
                    tiles_dm,
                    v_ch_off,
                    v_channel,
                    v_pages,
                    v_parts,
                    vch_beta,
                    vch_s_mul,
                    warpgroup_tidx,
                    warpgroup_widx,
                    with_phase,
                )
            ##############################
            # Dequant Dispatch
            ##############################
            elif warpgroup_idx_l in dequant_warpgroup_ids:
                self._dequant_warpgroup(
                    acc_dtype,
                    code_stage_bytes,
                    codek_consumer,
                    codev_consumer,
                    dequant_regs,
                    dequant_warpgroup_ids,
                    iters_s,
                    k_producer,
                    k_stages,
                    kv_stage_bytes,
                    kv_stages,
                    mma_tile_k,
                    mma_tile_m,
                    nofold,
                    pages_k,
                    pages_m,
                    pages_s,
                    prefetch_iters,
                    rebalance_regs,
                    sCode,
                    sScale,
                    scale_consumer,
                    tAsK,
                    tAsV,
                    tidx,
                    tiles_dk,
                    tiles_dm,
                    tiles_sk,
                    tiles_sm,
                    v_producer,
                    v_stages,
                )
            ##############################
            # Reduction Dispatch
            ##############################
            if warp_idx_l == reduction_warp_id:
                if cutlass.const_expr(do_kernel_red):
                    GqaDecode.reduction_epilogue(
                        blk_tile_hp,
                        coord_hp,
                        coord_hb,
                        kv_split_idx,
                        lane_idx,
                        sM_final_nbar,
                        sL_final_nbar,
                        sM,
                        sL,
                        mM_partial,
                        mL_partial,
                    )
                elif cutlass.const_expr(do_atomic_red):
                    gL = None
                    if cutlass.const_expr(store_lse):
                        gL = cute.local_tile(mL, (blk_tile_hp,), (coord_hp, coord_hb))
                    GqaDecode.reduction_cluster(
                        blk_tile_n,
                        kv_splits,
                        kv_split_idx,
                        lane_idx,
                        sM_final_nbar,
                        sL_final_nbar,
                        reduction_mbars_ptr,
                        sM,
                        sL,
                        sR,
                        gL,
                        scale_o,
                    )
                elif cutlass.const_expr(do_none_red):
                    gL = None
                    lane_store_lse = lane_store_max
                    if cutlass.const_expr(store_lse):
                        gL = cute.local_tile(mL, (blk_tile_hp,), (coord_hp, coord_hb))
                        # The lanes that hold a real query head and token, as
                        # `reduction_epilogue` predicates its stores (row 67).
                        cL = cute.local_tile(cute.make_identity_tensor(mL.shape[0]),
                                             blk_tile_hp, coord_hp)
                        idx_hg, idx_p = cL[lane_idx]
                        lane_store_lse = (lane_store_max & (idx_hg < grouped_heads)
                                          & (idx_p < prediction))
                    GqaDecode.reduction_none(
                        lane_store_max,
                        lane_idx,
                        sM_final_nbar,
                        sL_final_nbar,
                        sM,
                        sL,
                        gL,
                        scale_o,
                        lane_store_lse,
                    )
                # Persistent build: dependents launch once, after the item loop.
                if cutlass.const_expr(not persistent):
                    cute.arch.griddepcontrol_launch_dependents()
            # Item-boundary sync: all rings drained (every consumer
            # released, every producer committed) and all mbarriers
            # quiesced before the next item re-initializes them.
            if cutlass.const_expr(persistent):
                cute.arch.sync_threads()

        ##############################
        # Persistent epilogue (once per CTA)
        ##############################
        if cutlass.const_expr(persistent):
            # The kernel-entry warp index, not a fresh read; the comment at the top of the
            # item loop says why.
            if warp_idx == mma_vp_warp_id:
                tmem_ptr_p = cute.arch.retrieve_tmem_ptr(
                    Int32, 16, tmem_ptr_smem_ptr
                )
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.dealloc_tmem(tmem_ptr_p, tmem_alloc_cols)
            if warp_idx == reduction_warp_id:
                cute.arch.griddepcontrol_launch_dependents()
            # Drain the O-store group the last item left, if it stored one, before
            # the CTA exits. As at the initialization barrier, every thread runs the
            # wait, and only the TMA-VO warp's lanes can have a group to wait on. Where
            # `o_direct_for` holds there is no O store and so nothing to drain.
            if cutlass.const_expr(tma_atom_o is not None):
                cute.arch.cp_async_bulk_wait_group(0, read=True)

        return
