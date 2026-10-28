"""Canonical warp-MMA quantized paged-decode (SM90) — UNIFIED, configurable-granularity baseline.

Multi-warp CTA flash-decode (CuTe DSL, int4/int2 quantized KV, in-register dequant via a CONFLICT-FREE
nibble LUT [q -> bf16(q) from a 16/4-entry SMEM table strided to one bank/entry] + bf16 affine,
split-K + on-chip LSE combine + CombineLSE). The K/V quantization AXIS and GROUP-SIZE are configurable,
so one implementation covers the common KV-quant formats. `WarpMMADecodeMW` is the baseline;
`WarpMMADecode` (bottom of file) is the legacy 1-warp prototype kept for reference.

The LUT offloads the per-element bit-extract from the ALU pipe to the LSU.

Config (WarpMMADecodeMW.__init__):  k_axis, k_group, v_axis, v_group   (axis in {'token','channel'}).

  - token-wise   (scale per token, grouped along head-dim D):     scale tensor [B, S, H_kv, D//group].
        assert group<=D and D%group==0.  element (token n, channel d) -> group (n, d//group).
  - channel-wise (scale per channel, grouped along tokens S):     scale tensor [B, S//group, H_kv, D].
        assert group|N or N|group.  element (n, d) -> (n//group, d). Re-staged only on group crossings.
  - channel-wise SEQ (one scale per channel over the WHOLE sequence): scale tensor [B, H_kv, D].
        Tile- AND warp-invariant -> staged ONCE into shared [D] (zero overhead).

  DEFAULT: K=(token,128) + V=(token,128) — both per-token, group=128.
  (Validated bit-identical in behaviour to the pre-unification hardcoded kernel,
   across the full axis*group*bits matrix and the split-K perf shapes.)

Quant DATA path (repack_K/repack_V from probe_frag_layout) is format-agnostic — only the per-element
scale LOOKUP + SMEM scale staging + host scale-tensor shapes vary with (axis, group).

Harnesses: scripts/test_mw.py (correctness), scripts/test_mw_perf.py (split-K shapes),
  scripts/_smoke_seq.py (SEQ-channel regression).
"""
import os, sys, math
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp
from cutlass import Float32, Int32
from cutedsl_decode._vendor.flash_attn.cute import ampere_helpers as sm80_utils
from cutedsl_decode._vendor.flash_attn.cute import utils
from cutedsl_decode._vendor.flash_attn.cute import copy_utils
from cutedsl_decode._vendor.flash_attn.cute.softmax import Softmax
from cutedsl_decode._vendor.quack import layout_utils

DT = cutlass.BFloat16
_LOG2E = 1.4426950408889634

TOKEN = "token"
CHANNEL = "channel"
SEQ = "seq"        # channel-wise sentinel: ONE scale per channel over the WHOLE sequence (scale [B,H_kv,D]).
                   # Tile- AND warp-invariant -> staged ONCE into shared [D] (== the original kernel's V path).


def _validate_axis_group(name, axis, group, D, N):
    assert axis in (TOKEN, CHANNEL), f"{name}_axis must be 'token' or 'channel', got {axis!r}"
    if axis == TOKEN:
        # grouped along D: a scale per (token, group-of-`group` channels). group=D -> per-token.
        assert group != SEQ, f"{name} token-wise has no whole-sequence (SEQ) mode (grouping is along D)"
        assert group <= D and D % group == 0, (
            f"{name} token-wise requires group<=D and D%group==0 (D={D}, group={group}); "
            f"group>D (e.g. 256 at D=128) is rejected")
        return D // group           # number of d-groups (cols of the staged scale)
    else:
        if group == SEQ:
            return 1                # whole-sequence: one channel scale for all tokens; shared [D], staged once
        # grouped along S: a scale per (group-of-`group` tokens, channel).
        assert N % group == 0 or group % N == 0, (
            f"{name} channel-wise requires group|N or N|group for tile-invariant staging "
            f"(N={N}, group={group})")
        # # of token-groups a single N-tile (base aligned to N) touches -> compile-time constant.
        return (N + group - 1) // group if group < N else 1


class WarpMMADecodeMW:
    """Multi-warp CTA flash-decode with CONFIGURABLE K/V quant granularity (axis + group-size).

    W warps share Q in SMEM, each streams a DISJOINT KV sub-range, then LSE-combine on-chip. The
    K/V scale (s) and zero (z) are staged into SMEM per tile, per warp, with a shape decided by
    (axis, group); the in-register unpack looks up the element's scale-group from its B-frag coord.
    grid=(outer_splits, h_kv, batch), block=W*32."""

    def __init__(self, d, n_tile=64, m_tile=16, bits=4, num_warps=2,
                 k_axis=TOKEN, k_group=None, v_axis=TOKEN, v_group=None):   # default: both K,V token-wise group=128
        self.d = d; self.N = n_tile; self.M = m_tile; self.bits = bits
        self.W = num_warps; self.num_threads = num_warps * 32
        # default group = full axis extent (per-token for token, per-128-block... -> just full D for V)
        self.k_axis = k_axis; self.k_group = k_group if k_group is not None else d
        self.v_axis = v_axis; self.v_group = v_group if v_group is not None else d
        # whole-sequence channel: one scale per channel for ALL tokens -> stage ONCE into shared [D] (== original)
        self.k_seq = (k_axis == CHANNEL and self.k_group == SEQ)
        self.v_seq = (v_axis == CHANNEL and self.v_group == SEQ)
        # ng* = staged scale "rows-or-cols" count that varies WITHIN a tile:
        #   token   -> # d-groups (D//group)        (token index spans the N rows separately)
        #   channel -> # token-groups touched/tile  (channel index spans the D cols separately)
        self.k_ng = _validate_axis_group("k", self.k_axis, self.k_group, d, n_tile)
        self.v_ng = _validate_axis_group("v", self.v_axis, self.v_group, d, n_tile)

    def _mma(self):
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(DT, Float32, (16, 8, 16)), (1, 1, 1), permutation_mnk=(16, 16, 16))

    @cute.kernel
    def kernel(self, mQ, mKqr, mKs, mKz, mVqr, mVs, mVz, mOp, mLSE,
               scale_log2: Float32, softmax_scale: Float32, seqlen: Int32, tiles_per_cta: Int32, g: Int32,
               sQ_layout, sKqr_layout, sVqr_layout, sO_layout, sLSE_layout, sKsc_layout, sVsc_layout):
        outer, hk, b = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        wid = tidx // 32; wlane = tidx % 32
        D, N, M, W = self.d, self.N, self.M, self.W
        bits, vpb, qmask = self.bits, 8 // self.bits, cutlass.Uint8((1 << self.bits) - 1)
        vpc = 4 // self.bits   # quant values per 4-bit chunk: int4=1 (1bf16/LDS), int2=2 (2bf16/LDS.32)
        kg, vg = self.k_group, self.v_group
        k_is_token = cutlass.const_expr(self.k_axis == TOKEN)
        v_is_token = cutlass.const_expr(self.v_axis == TOKEN)
        KTk, perK = D // 16, (N // 8) * 4 * bits // 8
        KTv, perV = N // 16, (D // 8) * 4 * bits // 8
        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(DT, sQ_layout, byte_alignment=128)                 # shared
        sKqr = smem.allocate_tensor(cutlass.Uint8, sKqr_layout, byte_alignment=128)  # [W, rows, per]
        sVqr = smem.allocate_tensor(cutlass.Uint8, sVqr_layout, byte_alignment=128)
        # scale/zero staged per warp: token -> [W, N, ng] (ng d-groups); channel -> [W, ng, D] (ng tok-groups)
        sKs = smem.allocate_tensor(DT, sKsc_layout, byte_alignment=128)
        sKz = smem.allocate_tensor(DT, sKsc_layout, byte_alignment=128)
        sVsc = smem.allocate_tensor(DT, sVsc_layout, byte_alignment=128)
        sVzr = smem.allocate_tensor(DT, sVsc_layout, byte_alignment=128)
        sO = smem.allocate_tensor(Float32, sO_layout, byte_alignment=128)             # combine accum [M,D]
        sLSE = smem.allocate_tensor(Float32, sLSE_layout, byte_alignment=128)         # [W, M] per-warp lse
        sWt = smem.allocate_tensor(Float32, sLSE_layout, byte_alignment=128)          # [W, M] merge weight
        sLUT = smem.allocate_tensor(DT, cute.make_ordered_layout((16, 2), order=(1, 0)), byte_alignment=128)  # 4-bit-chunk LUT: entry=4B (1 bank)=>conflict-free; int4 uses 1 slot, int2 packs 2 int2

        # shared Q (grouped heads -> rows [0,g), pad with 0), loaded once by all warps
        for i in cutlass.range(tidx, M * D, self.num_threads, unroll=1):
            r, c = i // D, i % D
            sQ[(r, c)] = mQ[(b, hk * g + r, c)] if r < g else DT(0.0)
        # conflict-free nibble dequant LUT (shared, built once): sLUT[q] = bf16(q), entry q strided to bank q (4B).
        # A per-element nibble extract (cheap on the now-idle ALU) indexes it -> CONFLICT-FREE/broadcast SMEM load.
        for i in cutlass.range(tidx, 16 * vpc, self.num_threads, unroll=1):
            c = i // vpc; k = i % vpc                       # entry c (4-bit chunk) holds vpc decoded values; k-th is bf16((c>>bits*k)&mask)
            sLUT[(c, k)] = ((cutlass.Uint8(c) >> (cutlass.Uint8(bits) * cutlass.Uint8(k))) & qmask).to(Float32).to(DT)
        cute.arch.barrier()

        mma = self._mma(); thr_mma = mma.get_slice(wlane)
        ldA = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), DT)
        copy_Q = utils.make_tiled_copy_A(ldA, mma).get_slice(wlane)
        tSsQ = copy_Q.partition_S(sQ)
        tBcK = thr_mma.partition_B(cute.make_identity_tensor((N, D)))   # K coord -> (token n, channel d)
        tVcV = thr_mma.partition_B(cute.make_identity_tensor((D, N)))   # V coord -> (channel d, token n)
        tcK = copy_utils.tiled_copy_2d(cutlass.Uint8, perK, 32, is_async=True)
        tcV = copy_utils.tiled_copy_2d(cutlass.Uint8, perV, 32, is_async=True)
        thr_tcK = tcK.get_slice(wlane); thr_tcV = tcV.get_slice(wlane)
        gKqr = mKqr[(b, None, hk, None, None)]; gVqr = mVqr[(b, None, hk, None, None)]
        sKqr_w = sKqr[(wid, None, None)]; sVqr_w = sVqr[(wid, None, None)]
        tKsKq = thr_tcK.partition_D(sKqr_w); tKsVq = thr_tcV.partition_D(sVqr_w)
        if cutlass.const_expr(self.k_seq):
            sKs_w = sKs; sKz_w = sKz                                       # shared [D], indexed by channel
        else:
            sKs_w = sKs[(wid, None, None)]; sKz_w = sKz[(wid, None, None)]
        if cutlass.const_expr(self.v_seq):
            sVsc_w = sVsc; sVzr_w = sVzr
        else:
            sVsc_w = sVsc[(wid, None, None)]; sVzr_w = sVzr[(wid, None, None)]
        # whole-sequence channel scales are tile- AND warp-invariant -> stage ONCE, shared by all warps (== original)
        if cutlass.const_expr(self.k_seq):
            for i in cutlass.range(tidx, D, self.num_threads, unroll=1):
                sKs[i] = mKs[(b, hk, i)].to(DT); sKz[i] = mKz[(b, hk, i)].to(DT)
        if cutlass.const_expr(self.v_seq):
            for i in cutlass.range(tidx, D, self.num_threads, unroll=1):
                sVsc[i] = mVs[(b, hk, i)].to(DT); sVzr[i] = mVz[(b, hk, i)].to(DT)
        if cutlass.const_expr(self.k_seq or self.v_seq):
            cute.arch.barrier()

        acc_O = cute.make_rmem_tensor(thr_mma.partition_shape_C((M, D)), Float32); acc_O.fill(0.0)
        softmax = Softmax.create(scale_log2, num_rows=acc_O.shape[0][0] * acc_O.shape[1], arch=80)
        softmax.reset()

        # this warp's DISJOINT KV sub-range: [base_w, base_w + tiles_per_cta//W tiles)
        tpw = tiles_per_cta // W
        wtok0 = outer * tiles_per_cta * N + wid * tpw * N
        nb = Int32(0)
        while nb < tpw:                          # all W warps run tpw tiles -> block barriers in lockstep
            base = wtok0 + nb * N
            base_tile = base // N
            cute.copy(tcK, thr_tcK.partition_S(gKqr[(base_tile, None, None)]), tKsKq)
            cute.copy(tcV, thr_tcV.partition_S(gVqr[(base_tile, None, None)]), tKsVq)
            # ---- stage this tile's K scale/zero into SMEM (shape decided by k_axis/k_group) ----
            if cutlass.const_expr(self.k_seq):
                pass                                        # whole-seq channel: staged once before the loop
            elif cutlass.const_expr(k_is_token):
                kg0 = base                                  # token row = absolute token; col = d-group
                for i in cutlass.range(wlane, N * self.k_ng, 32, unroll=1):
                    n, dg = i // self.k_ng, i % self.k_ng
                    sKs_w[(n, dg)] = mKs[(b, hk, kg0 + n, dg)].to(DT)   # scale layout [B,HKV,S,ng]: consecutive tokens contiguous -> coalesced
                    sKz_w[(n, dg)] = mKz[(b, hk, kg0 + n, dg)].to(DT)
            elif cutlass.const_expr(self.k_ng == 1):
                # grouped channel, one token-group per tile: re-stage ONLY on a group-boundary crossing
                # (whole-seq would be 1 group -> staged on tile 0 only; grouped-128 -> every group/N tiles)
                kg0 = base // kg
                if (nb == 0) | (kg0 != (base - N) // kg):
                    for i in cutlass.range(wlane, D, 32, unroll=1):
                        sKs_w[(0, i)] = mKs[(b, hk, kg0, i)].to(DT)   # scale layout [B,HKV,S//g,D]
                        sKz_w[(0, i)] = mKz[(b, hk, kg0, i)].to(DT)
            else:
                kg0 = base // kg                            # group<N: multiple token-groups/tile -> per-tile
                for i in cutlass.range(wlane, self.k_ng * D, 32, unroll=1):
                    tg, dch = i // D, i % D
                    sKs_w[(tg, dch)] = mKs[(b, hk, kg0 + tg, dch)].to(DT)
                    sKz_w[(tg, dch)] = mKz[(b, hk, kg0 + tg, dch)].to(DT)
            # ---- stage this tile's V scale/zero into SMEM (shape decided by v_axis/v_group) ----
            if cutlass.const_expr(self.v_seq):
                pass                                        # whole-seq channel: staged once before the loop
            elif cutlass.const_expr(v_is_token):
                vg0 = base
                for i in cutlass.range(wlane, N * self.v_ng, 32, unroll=1):
                    n, dg = i // self.v_ng, i % self.v_ng
                    sVsc_w[(n, dg)] = mVs[(b, hk, vg0 + n, dg)].to(DT)   # scale layout [B,HKV,S,ng]: coalesced
                    sVzr_w[(n, dg)] = mVz[(b, hk, vg0 + n, dg)].to(DT)
            elif cutlass.const_expr(self.v_ng == 1):
                # grouped channel, one token-group per tile: re-stage ONLY on a group-boundary crossing
                vg0 = base // vg
                if (nb == 0) | (vg0 != (base - N) // vg):
                    for i in cutlass.range(wlane, D, 32, unroll=1):
                        sVsc_w[(0, i)] = mVs[(b, hk, vg0, i)].to(DT)   # scale layout [B,HKV,S//g,D]
                        sVzr_w[(0, i)] = mVz[(b, hk, vg0, i)].to(DT)
            else:
                vg0 = base // vg                            # group<N: multiple token-groups/tile -> per-tile
                for i in cutlass.range(wlane, self.v_ng * D, 32, unroll=1):
                    tg, dch = i // D, i % D
                    sVsc_w[(tg, dch)] = mVs[(b, hk, vg0 + tg, dch)].to(DT)
                    sVzr_w[(tg, dch)] = mVz[(b, hk, vg0 + tg, dch)].to(DT)
            cute.arch.cp_async_commit_group(); cute.arch.cp_async_wait_group(0)
            cute.arch.sync_warp()       # warp-local: each warp owns its K/V buffer (no cross-warp sync)

            acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((M, N)), Float32); acc_S.fill(0.0)
            KT = cute.size(tBcK, mode=[2])
            qbuf = cute.make_rmem_tensor(thr_mma.partition_shape_A((M, 32)), DT)
            qbuf_cp = copy_Q.retile(qbuf)
            cute.copy(copy_Q, tSsQ[None, None, 0], qbuf_cp[None, None, 0])
            tSrK_k = cute.make_rmem_tensor(tBcK[None, None, 0].shape, DT)
            qfK = cute.make_rmem_tensor(cute.make_ordered_layout((cute.size(tSrK_k) // 2, 2), order=(1, 0)), DT)  # [pair,2] bf16(q)
            regK = cute.make_rmem_tensor(perK, cutlass.Uint8)
            for kt in cutlass.range_constexpr(KT):
                if cutlass.const_expr(kt < KT - 1):
                    cute.copy(copy_Q, tSsQ[None, None, kt + 1], qbuf_cp[None, None, (kt + 1) % 2])
                cute.autovec_copy(sKqr_w[(kt * 32 + wlane, None)], regK)
                cKk = tBcK[None, None, kt]
                for bi in cutlass.range_constexpr(perK):       # decode by 4-bit chunk: 2 chunks/byte, vpc values/chunk
                    bb = regK[bi]
                    for ci in cutlass.range_constexpr(2):
                        chunk = (bb >> (cutlass.Uint8(4) * cutlass.Uint8(ci))) & cutlass.Uint8(0xF)
                        if cutlass.const_expr(vpc == 1):
                            qfK[(bi, ci)] = sLUT[(Int32(chunk), 0)]                                   # int4: 1 bf16/nibble
                        else:
                            cute.autovec_copy(sLUT[(Int32(chunk), None)], qfK[(bi * 2 + ci, None)])   # int2: 1 LDS.32 -> 2 bf16
                for j in cutlass.range_constexpr(cute.size(tSrK_k)):
                    n, dd = cKk[j]                          # K B-frag coord: (token n, channel d)
                    if cutlass.const_expr(self.k_seq):
                        s = sKs_w[dd]; z = sKz_w[dd]        # whole-seq channel: one scale per channel
                    elif cutlass.const_expr(k_is_token):
                        s = sKs_w[(n, dd // kg)]; z = sKz_w[(n, dd // kg)]
                    else:                                   # grouped channel: local token-group = n//group
                        tgl = n // kg
                        s = sKs_w[(tgl, dd)]; z = sKz_w[(tgl, dd)]
                    tSrK_k[j] = qfK[(j // 2, j % 2)] * s + z   # bf16 affine (LUT gives exact q, no -128)
                cute.gemm(mma, acc_S, qbuf[None, None, kt % 2], tSrK_k, acc_S)

            row_scale = softmax.online_softmax(acc_S, is_first=False)
            softmax.rescale_O(acc_O, row_scale)
            rP = cute.make_rmem_tensor_like(acc_S, DT); rP.store(acc_S.load().to(DT))
            tOrP = layout_utils.reshape_acc_to_frgA(rP)
            KTv2 = cute.size(tVcV, mode=[2])
            tOrVt_k = cute.make_rmem_tensor(tVcV[None, None, 0].shape, DT)
            qfV = cute.make_rmem_tensor(cute.make_ordered_layout((cute.size(tOrVt_k) // 2, 2), order=(1, 0)), DT)  # [pair,2] bf16(q)
            regV = cute.make_rmem_tensor(perV, cutlass.Uint8)
            for kk in cutlass.range_constexpr(KTv2):
                cute.autovec_copy(sVqr_w[(kk * 32 + wlane, None)], regV)
                cVk = tVcV[None, None, kk]
                for bi in cutlass.range_constexpr(perV):       # decode by 4-bit chunk: 2 chunks/byte, vpc values/chunk
                    bb = regV[bi]
                    for ci in cutlass.range_constexpr(2):
                        chunk = (bb >> (cutlass.Uint8(4) * cutlass.Uint8(ci))) & cutlass.Uint8(0xF)
                        if cutlass.const_expr(vpc == 1):
                            qfV[(bi, ci)] = sLUT[(Int32(chunk), 0)]                                   # int4: 1 bf16/nibble
                        else:
                            cute.autovec_copy(sLUT[(Int32(chunk), None)], qfV[(bi * 2 + ci, None)])   # int2: 1 LDS.32 -> 2 bf16
                for j in cutlass.range_constexpr(cute.size(tOrVt_k)):
                    dch, n = cVk[j]                         # V B-frag coord: (channel d, token n)
                    if cutlass.const_expr(self.v_seq):
                        s = sVsc_w[dch]; z = sVzr_w[dch]    # whole-seq channel: one scale per channel
                    elif cutlass.const_expr(v_is_token):
                        s = sVsc_w[(n, dch // vg)]; z = sVzr_w[(n, dch // vg)]
                    else:                                   # grouped channel: local token-group = n//group
                        tgl = n // vg
                        s = sVsc_w[(tgl, dch)]; z = sVzr_w[(tgl, dch)]
                    tOrVt_k[j] = qfV[(j // 2, j % 2)] * s + z   # bf16 affine (LUT gives exact q, no -128)
                cute.gemm(mma, acc_O, tOrP[None, None, kk], tOrVt_k, acc_O)
            cute.arch.sync_warp()       # warp-local: before this warp reloads its own K/V buffer
            nb += 1

        row_scale = softmax.finalize(); softmax.rescale_O(acc_O, row_scale)   # acc_O := O_warp, row_sum := lse
        cId_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cute.make_identity_tensor((M, D))))
        # --- ON-CHIP combine: stage per-warp lse -> per-row gmax/denom -> weight -> seq-add weighted O ---
        for r in cutlass.range_constexpr(cute.size(softmax.row_sum)):
            m = cId_mn[(r, 0)][0]
            if m < g:
                sLSE[(wid, m)] = softmax.row_sum[r]
        cute.arch.barrier()
        for r in cutlass.range_constexpr(cute.size(softmax.row_sum)):
            m = cId_mn[(r, 0)][0]
            if m < g:
                gm = Float32(-1e30)
                for w in cutlass.range_constexpr(W):
                    gm = cutlass.max(gm, sLSE[(w, m)])
                dn = Float32(0.0)
                for w in cutlass.range_constexpr(W):
                    dn += cute.arch.exp2((sLSE[(w, m)] - gm) * Float32(_LOG2E))
                sWt[(wid, m)] = cute.arch.exp2((softmax.row_sum[r] - gm) * Float32(_LOG2E)) / dn
                mLSE[(outer, b, hk * g + m)] = gm + cute.math.log(dn)
        for i in cutlass.range(tidx, M * D, self.num_threads, unroll=1):
            sO[(i // D, i % D)] = Float32(0.0)
        cute.arch.barrier()
        tCcO = thr_mma.partition_C(cute.make_identity_tensor((M, D)))
        for w in cutlass.range_constexpr(W):
            if wid == w:
                for i in cutlass.range_constexpr(cute.size(acc_O)):
                    m, dd = tCcO[i]
                    if m < g:
                        sO[(m, dd)] = sO[(m, dd)] + sWt[(wid, m)] * acc_O[i]
            cute.arch.barrier()
        for i in cutlass.range(tidx, g * D, self.num_threads, unroll=1):
            m, dd = i // D, i % D
            mOp[(outer, b, hk * g + m, dd)] = sO[(m, dd)]

    @cute.jit
    def __call__(self, mQ, mKqr, mKs, mKz, mVqr, mVs, mVz, mOp, mLSE, scale_log2, softmax_scale,
                 seqlen, outer_splits, h_kv, batch, g, stream):
        D, N, M, W = self.d, self.N, self.M, self.W
        sQ_layout = cute.tile_to_shape(sm80_utils.get_smem_layout_atom(DT, D), (M, D), order=(0, 1))
        KTk, perK = D // 16, (N // 8) * 4 * self.bits // 8
        KTv, perV = N // 16, (D // 8) * 4 * self.bits // 8
        sKqr_layout = cute.make_ordered_layout((W, KTk * 32, perK), order=(2, 1, 0))
        sVqr_layout = cute.make_ordered_layout((W, KTv * 32, perV), order=(2, 1, 0))
        # staged scale buffers: whole-seq channel -> shared [D] (staged ONCE); token -> [W,N,ng]; grouped channel -> [W,ng,D]
        if cutlass.const_expr(self.k_seq):
            sKsc_layout = cute.make_layout(D)
        else:
            ksc_rows, ksc_cols = (N, self.k_ng) if self.k_axis == TOKEN else (self.k_ng, D)
            sKsc_layout = cute.make_ordered_layout((W, ksc_rows, ksc_cols), order=(2, 1, 0))
        if cutlass.const_expr(self.v_seq):
            sVsc_layout = cute.make_layout(D)
        else:
            vsc_rows, vsc_cols = (N, self.v_ng) if self.v_axis == TOKEN else (self.v_ng, D)
            sVsc_layout = cute.make_ordered_layout((W, vsc_rows, vsc_cols), order=(2, 1, 0))
        sO_layout = cute.make_ordered_layout((M, D), order=(1, 0))
        sLSE_layout = cute.make_ordered_layout((W, M), order=(1, 0))
        total_tiles = (seqlen + N - 1) // N
        tiles_per_cta = (total_tiles + outer_splits - 1) // outer_splits
        self.kernel(mQ, mKqr, mKs, mKz, mVqr, mVs, mVz, mOp, mLSE, scale_log2, softmax_scale, seqlen,
                    tiles_per_cta, g, sQ_layout, sKqr_layout, sVqr_layout, sO_layout, sLSE_layout,
                    sKsc_layout, sVsc_layout).launch(
            grid=[outer_splits, h_kv, batch], block=[self.num_threads, 1, 1], stream=stream)


class CombineLSE:
    """LSE-merge the split-K partials: grid=(HQ, B), one block per (b,hq), D threads (thread d -> O[d])."""
    def __init__(self, d):
        self.d = d
        self.num_threads = d

    @cute.kernel
    def kernel(self, mOp, mLSE, mO, splits: Int32):
        hq, b, _ = cute.arch.block_idx()
        dd, _, _ = cute.arch.thread_idx()
        m = Float32(-1e30)
        for s in cutlass.range(splits, unroll=1):
            m = cutlass.max(m, mLSE[(s, b, hq)])
        denom = Float32(0.0)
        acc = Float32(0.0)
        for s in cutlass.range(splits, unroll=1):
            w = cute.arch.exp2((mLSE[(s, b, hq)] - m) * Float32(_LOG2E))
            denom += w
            acc += w * mOp[(s, b, hq, dd)]
        mO[(b, hq, dd)] = (acc / denom).to(DT)

    @cute.jit
    def __call__(self, mOp, mLSE, mO, splits, hq, batch, stream):
        self.kernel(mOp, mLSE, mO, splits).launch(
            grid=[hq, batch, 1], block=[self.num_threads, 1, 1], stream=stream)


# ===================== LEGACY: 1-warp prototype =====================
# The earlier prototype (single-warp CTA, HBM split-K partials, no on-chip combine).
# Superseded by the multi-warp WarpMMADecodeMW above; kept for bench_warpmma/prof_warpmma.

class WarpMMADecode:
    """One warp per CTA; grid = (kv_splits, h_kv, batch). M = grouped heads (h_q//h_kv), padded to 16."""
    def __init__(self, d, n_tile=64, m_tile=16, bits=4):
        self.d = d
        self.N = n_tile           # KV tokens per QK tile
        self.M = m_tile           # query rows (grouped heads), padded to mma M=16
        self.bits = bits          # 4 (int4, 2 vals/byte) or 2 (int2, 4 vals/byte)
        self.num_threads = 32

    def _mma(self):
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(DT, Float32, (16, 8, 16)), (1, 1, 1),
            permutation_mnk=(16, 16, 16))

    @cute.kernel
    def kernel(self, mQ, mKqr, mKs, mKz, mVqr, mVs, mVz, mOp, mLSE,
               scale_log2: Float32, softmax_scale: Float32,
               seqlen: Int32, tiles_per_split: Int32, g: Int32,
               sQ_layout, sKqr_layout, sVqr_layout):
        split, hk, b = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        D, N, M = self.d, self.N, self.M
        # repacked (fragment-order) packed tiles: row = kslice*32 + lane; each byte packs vpb=8/bits
        # fragment elems. lane reads its contiguous row -> conflict-free LDS.128 (was 16x/32x scatter).
        bits, vpb, qmask = self.bits, 8 // self.bits, cutlass.Uint8((1 << self.bits) - 1)
        KTk, perK = D // 16, (N // 8) * 4 * bits // 8       # K: 8 kslices, int4=16/int2=8 B/lane/slice
        KTv, perV = N // 16, (D // 8) * 4 * bits // 8       # V: 4 kslices, int4=32/int2=16 B/lane/slice
        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(DT, sQ_layout, byte_alignment=128)
        sKqr = smem.allocate_tensor(cutlass.Uint8, sKqr_layout, byte_alignment=128)
        sKs = smem.allocate_tensor(Float32, cute.make_layout(N), byte_alignment=128)
        sKz = smem.allocate_tensor(Float32, cute.make_layout(N), byte_alignment=128)
        sVqr = smem.allocate_tensor(cutlass.Uint8, sVqr_layout, byte_alignment=128)
        sVsc = smem.allocate_tensor(Float32, cute.make_layout(D), byte_alignment=128)  # per-channel
        sVzr = smem.allocate_tensor(Float32, cute.make_layout(D), byte_alignment=128)

        # this split's token range
        tok0 = split * tiles_per_split * N
        tok1 = cutlass.min((split + 1) * tiles_per_split * N, seqlen)

        # load Q for (b, hk): grouped heads [hk*g : hk*g+g] -> rows [0,g); pad rows [g,M) with 0
        for i in cutlass.range(tidx, M * D, self.num_threads, unroll=1):
            r, c = i // D, i % D
            sQ[(r, c)] = mQ[(b, hk * g + r, c)] if r < g else DT(0.0)
        # stage this CTA's per-channel V scale/zero (D,) once
        for i in cutlass.range(tidx, D, self.num_threads, unroll=1):
            sVsc[i] = mVs[(b, hk, i)]; sVzr[i] = mVz[(b, hk, i)]
        cute.arch.barrier()

        mma = self._mma()
        thr_mma = mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        ldA = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), DT)
        copy_Q = utils.make_tiled_copy_A(ldA, mma).get_slice(tidx)
        tSsQ = copy_Q.partition_S(sQ)
        tBcK = thr_mma.partition_B(cute.make_identity_tensor((N, D)))

        # coalesced cp.async loads of the repacked tiles ([KT*32, per], row-contiguous). seqlen%N==0.
        tcK = copy_utils.tiled_copy_2d(cutlass.Uint8, perK, self.num_threads, is_async=True)
        tcV = copy_utils.tiled_copy_2d(cutlass.Uint8, perV, self.num_threads, is_async=True)
        thr_tcK = tcK.get_slice(tidx); thr_tcV = tcV.get_slice(tidx)
        gKqr = mKqr[(b, None, hk, None, None)]    # [NT, KTk*32, perK]
        gVqr = mVqr[(b, None, hk, None, None)]    # [NT, KTv*32, perV]
        tKsKq = thr_tcK.partition_D(sKqr)
        tKsVq = thr_tcV.partition_D(sVqr)

        acc_O = cute.make_rmem_tensor(thr_mma.partition_shape_C((M, D)), Float32)
        acc_O.fill(0.0)
        softmax = Softmax.create(scale_log2, num_rows=acc_O.shape[0][0] * acc_O.shape[1], arch=80)
        softmax.reset()

        nb = Int32(0)
        ntok = tok1 - tok0
        # is_first=False for ALL tiles: with reset() (row_max=-inf, row_sum=0) the first tile's
        # rescale-by-0 of the zero acc_O is harmless -> identical result, no tile-0-separate needed.
        # (single-buffer: double-buffering tested and REGRESSED -- bound is too-few-warps not latency.)
        while nb * N < ntok:
            base = tok0 + nb * N
            base_tile = base // N
            cute.copy(tcK, thr_tcK.partition_S(gKqr[(base_tile, None, None)]), tKsKq)
            cute.copy(tcV, thr_tcV.partition_S(gVqr[(base_tile, None, None)]), tKsVq)
            for i in cutlass.range(tidx, N, self.num_threads, unroll=1):
                sKs[i] = mKs[(b, base + i, hk)]; sKz[i] = mKz[(b, base + i, hk)]
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            # QK: per-k-slice fused unpack -- only ONE K-slice live in regs (was the whole 256-elem
            # B-frag -> 128 regs -> spill). Mirrors sm80_utils.gemm: ldmatrix Q slice, unpack K slice,
            # mma; prefetch next Q slice. Within a slice the ~32 nibble-unpacks are independent (ILP).
            acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((M, N)), Float32)
            acc_S.fill(0.0)
            tSrQ_cp = copy_Q.retile(tSrQ)
            cute.copy(copy_Q, tSsQ[None, None, 0], tSrQ_cp[None, None, 0])
            KT = cute.size(tBcK, mode=[2])
            tSrK_k = cute.make_rmem_tensor(tBcK[None, None, 0].shape, DT)   # reused per k-slice
            regK = cute.make_rmem_tensor(perK, cutlass.Uint8)              # lane's 16 repacked bytes / slice
            for kt in cutlass.range_constexpr(KT):
                if cutlass.const_expr(kt < KT - 1):
                    cute.copy(copy_Q, tSsQ[None, None, kt + 1], tSrQ_cp[None, None, kt + 1])
                cute.autovec_copy(sKqr[(kt * 32 + tidx, None)], regK)  # 1 conflict-free LDS.128
                cKk = tBcK[None, None, kt]
                for j in cutlass.range_constexpr(cute.size(tSrK_k)):
                    n, _ = cKk[j]                                       # token index for the scale
                    byte = regK[j // vpb]
                    q = (byte >> (cutlass.Uint8(bits) * (j % vpb))) & qmask
                    tSrK_k[j] = (q.to(Float32) * sKs[n] + sKz[n]).to(DT)
                cute.gemm(mma, acc_S, tSrQ[None, None, kt], tSrK_k, acc_S)

            row_scale = softmax.online_softmax(acc_S, is_first=False)
            softmax.rescale_O(acc_O, row_scale)
            rP = cute.make_rmem_tensor_like(acc_S, DT)
            rP.store(acc_S.load().to(DT))
            tOrP = layout_utils.reshape_acc_to_frgA(rP)
            # PV: per-k-slice fused V unpack -- one V-slice live (was 256-elem -> 128 regs). V B-frag
            # is [d, tokens]; per (d,token) coord unpack of packed V + per-channel s_d/z_d; mma.
            tVcV = thr_mma.partition_B(cute.make_identity_tensor((D, N)))
            KTv = cute.size(tVcV, mode=[2])
            tOrVt_k = cute.make_rmem_tensor(tVcV[None, None, 0].shape, DT)   # reused per k-slice
            regV = cute.make_rmem_tensor(perV, cutlass.Uint8)               # lane's 32 repacked bytes / slice
            for kk in cutlass.range_constexpr(KTv):
                cute.autovec_copy(sVqr[(kk * 32 + tidx, None)], regV)    # conflict-free LDS.128 x2
                cVk = tVcV[None, None, kk]
                for j in cutlass.range_constexpr(cute.size(tOrVt_k)):
                    dch, _ = cVk[j]                                      # channel index for the scale
                    bv = regV[j // vpb]
                    qv = (bv >> (cutlass.Uint8(bits) * (j % vpb))) & qmask
                    tOrVt_k[j] = (qv.to(Float32) * sVsc[dch] + sVzr[dch]).to(DT)
                cute.gemm(mma, acc_O, tOrP[None, None, kk], tOrVt_k, acc_O)
            cute.arch.barrier()
            nb += 1

        row_scale = softmax.finalize()        # normalizes; acc_O := O (final, this split)
        softmax.rescale_O(acc_O, row_scale)
        # write partial O[split,b,head,:] (final O when kv_splits==1)
        cO = cute.make_identity_tensor((M, D))
        tCcO = thr_mma.partition_C(cO)
        for i in cutlass.range_constexpr(cute.size(acc_O)):
            m, dd = tCcO[i]
            if m < g:
                mOp[(split, b, hk * g + m, dd)] = acc_O[i]
        # LSE per head row: finalize() OVERWROTE softmax.row_sum[r] with the natural-log LSE.
        cId_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cute.make_identity_tensor((M, D))))
        for r in cutlass.range_constexpr(cute.size(softmax.row_sum)):
            m = cId_mn[(r, 0)][0]
            if m < g:
                mLSE[(split, b, hk * g + m)] = softmax.row_sum[r]

    @cute.jit
    def __call__(self, mQ, mKqr, mKs, mKz, mVqr, mVs, mVz, mOp, mLSE, scale_log2, softmax_scale,
                 seqlen, kv_splits, h_kv, batch, g, stream):
        D, N, M = self.d, self.N, self.M
        sQ_layout = cute.tile_to_shape(sm80_utils.get_smem_layout_atom(DT, D), (M, D), order=(0, 1))
        KTk, perK = D // 16, (N // 8) * 4 * self.bits // 8
        KTv, perV = N // 16, (D // 8) * 4 * self.bits // 8
        sKqr_layout = cute.make_ordered_layout((KTk * 32, perK), order=(1, 0))   # per-byte row contiguous
        sVqr_layout = cute.make_ordered_layout((KTv * 32, perV), order=(1, 0))
        total_tiles = (seqlen + N - 1) // N
        tiles_per_split = (total_tiles + kv_splits - 1) // kv_splits
        self.kernel(mQ, mKqr, mKs, mKz, mVqr, mVs, mVz, mOp, mLSE, scale_log2, softmax_scale, seqlen,
                    tiles_per_split, g, sQ_layout, sKqr_layout, sVqr_layout).launch(
            grid=[kv_splits, h_kv, batch], block=[self.num_threads, 1, 1], stream=stream)
