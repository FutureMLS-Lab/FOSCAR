# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Literal, Type, Tuple, cast
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
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
    Float32,
    BFloat16,
    Float16,
    Int32,
    Int64,
    Optional,
)

# Kernel invariants
mma_modes = (0, 1, 2)
mma_dice = (None, None, None)  # (MMA, #MMA_M, #MMA_K)
warp_threads = 32
warpgroup_warps = 4
warpgroup_threads = 128
max_reduction_iters = 4  # log2(16)

# Math helpers
log2_e = math.log2(math.e)  # change exponential base
exp2 = partial(cute.math.exp2, fastmath=True)
warp_fmax = partial(cute.arch.warp_redux_sync, kind="fmax", nan=True)
smem_fmax = partial(cute.arch.atomic_fmax, sem="relaxed", scope="cta")
gmem_fmax = partial(cute.arch.atomic_fmax, sem="relaxed", scope="gpu")


class GroupedQueryAttentionDecode:
    # The unquantized kernel this class used to carry -- __init__, the @cute.jit
    # __call__, the @cute.kernel decode body, and the example run()/__main__
    # harness -- went with the rest of the dense decode path. What remains is
    # deliberately NOT a kernel: it is the split-K reduction library that the
    # quantized kernel imports, which is why this file survived at all.
    #
    # int_fold_decode.py takes 11 module-level names from here, and uses this
    # CLASS as the namespace for the reduction members below, so neither the
    # module name nor the class name may change.
    # Two dependencies are invisible to a grep of those import lists:
    #   * `launch_reduction` reaches the kernel through a class-qualified
    #     `GroupedQueryAttentionDecode.reduction_kernel(...)` call.
    #   * `reduction_epilogue` resolves `gmem_fmax` as a MODULE GLOBAL rather
    #     than taking it as an argument.
    # Pruning either one fails at TRACE time, not at import time.
    #
    # `can_implement` is called UNBOUND with a foreign `self` -- the fold kernels
    # pass their own instance -- which is why it stays a plain method on a class
    # that is now never instantiated.
    ##############################
    # Launch helpers
    ##############################
    # Runtime implementable check
    def can_implement(
        self, kv_splits, qo_shape_bshd, kv_shape_bshd, qkv_dtype, o_dtype
    ):
        b_k, s_q, h_q, d_q = qo_shape_bshd
        b_q, s_k, h_k, d_k = kv_shape_bshd

        if qkv_dtype is cutlass.Float8E4M3:
            raise TypeError("use Float8E4M3FN instead of Float8E4M3")

        if not (d_q == d_k == self.head_dim):
            raise ValueError(
                f"headdim_q({d_q}), headdim_k({d_k}) must be {self.head_dim}"
            )

        if h_q % h_k != 0:
            raise ValueError(f"heads_q({h_q}) must be a multiple of heads_k({h_k})")

        if 0 < s_k < s_q:
            raise ValueError(
                f"non-zero seqlen({s_k}) must be at least prediction({s_q})"
            )

        # Only causal mask up to two tiles
        if not self.tma_mask and s_q > self.sequence_tile:
            raise ValueError(
                f"prediction({s_q}) with causal mask can be at most {self.sequence_tile}"
            )

        if self.tma_mask and 0 < s_k < 8:
            raise ValueError(
                f"non-zero seqlen({s_k}) with TMA masking must be at least 8"
            )

        if b_k != b_q:
            raise ValueError(f"batches_k({b_k}) and batches_q({b_q}) mismatch")

        if self.do_atomic_red:
            if kv_splits not in (1, 2, 4, 8, 16):
                raise ValueError(
                    f"atomic reduction requires kv_splits po2 <= 16, got {kv_splits}"
                )

            if o_dtype not in (Float32, BFloat16, Float16):
                raise TypeError(
                    f"atomic reduction requires (Float32, BFloat16, Float16) o_dtype, got {o_dtype}"
                )

        if self.do_none_red and kv_splits != 1:
            raise ValueError("KV splits must be 1 if flash decoding is disabled")

    # Pack grouped heads with predicted tokens (s_q)
    @staticmethod
    def gqa_pack(t_bshd: cute.Tensor, h_k: int):
        d, h_q, s_q, b = tuple(reversed(t_bshd.shape))[:4]
        stride_d, stride_h, stride_s, stride_b = tuple(reversed(t_bshd.stride))[:4]
        # Batch + partial stride must be coalescible
        # to get 5 independent TMA modes (TMA limitation)
        has_partial = cute.rank(t_bshd) == 5
        b_partial = b * t_bshd.shape[0] if has_partial else b
        h_g = h_q // h_k  # grouped heads
        gqa_shape = (b_partial, (h_g, s_q), h_k, d)
        gqa_stride = (stride_b, (stride_h, stride_s), stride_h * h_g, stride_d)
        gqa_layout = cute.make_layout(gqa_shape, stride=gqa_stride)
        return cute.make_tensor(t_bshd.iterator, gqa_layout)

    # Reorder and group modes for GEMM
    @staticmethod
    def gemm_view(t_bshd: cute.Tensor, s_first: bool):
        sdhb = (1, 3, 2, 0)  # GEMM1 MKL
        dshb = (3, 1, 2, 0)  # GEMM2 MKL
        reorder = sdhb if s_first else dshb
        mT_layout = cute.select(t_bshd.layout, reorder)
        mT_layout = cute.group_modes(mT_layout, 2, 4)
        return cute.make_tensor(t_bshd.iterator, mT_layout)

    # Pack, reorder, and group modes for GEMM workspace
    @staticmethod
    def gemm_view_bsh(t_bsh: cute.Tensor, h_k: int):
        h_q, s_q, b = tuple(reversed(t_bsh.shape))[:3]
        stride_h, stride_s, stride_b = tuple(reversed(t_bsh.stride))[:3]
        h_g = h_q // h_k
        mT_shape = ((h_g, s_q), (h_k, b))
        mT_stride = ((stride_h, stride_s), (stride_h * h_g, stride_b))
        has_partial = cute.rank(t_bsh) == 4
        mT_shape += (t_bsh.shape[0],) if has_partial else ()
        mT_stride += (t_bsh.stride[0],) if has_partial else ()
        mT_layout = cute.make_layout(mT_shape, stride=mT_stride)
        return cute.make_tensor(t_bsh.iterator, mT_layout)

    @staticmethod
    @cute.jit
    def reduction_none(
        lane_store_max: bool,
        lane_idx: Int32,
        sM_final_nbar: nbar,
        sL_final_nbar: nbar,
        sM: cute.Tensor,
        sL: cute.Tensor,
        gL: Optional[cute.Tensor],
        scale_o: Float32,
    ):
        store_lse = gL is not None
        colmax = Float32(0)
        colsum = Float32(0)
        sM_final_nbar.arrive_and_wait()
        if cutlass.const_expr(store_lse):
            if lane_store_max:
                colmax = sM[lane_idx]
        sL_final_nbar.arrive_and_wait()
        if lane_store_max:
            sL_lane = sL[lane_idx, None]
            colsum = sL_lane[0] + sL_lane[1] + sL_lane[2] + sL_lane[3]
            normalization = cute.arch.rcp_approx(colsum) * scale_o
            sM[lane_idx] = normalization
        sM_final_nbar.arrive()
        if cutlass.const_expr(store_lse):
            if lane_store_max:
                gL[lane_idx] = colmax + cute.math.log2(colsum)

    @staticmethod
    @cute.jit
    def reduction_epilogue(
        blk_tile_hp: Tuple[int, int],
        coord_hp: Tuple[Int32, Int32],
        coord_hb: Tuple[Int32, Int32],
        kv_split_idx: Int32,
        lane_idx: Int32,
        sM_final_nbar: nbar,
        sL_final_nbar: nbar,
        sM: cute.Tensor,
        sL: cute.Tensor,
        mM: cute.Tensor,
        mM_partial: cute.Tensor,
        mL_partial: cute.Tensor,
    ):
        # get gmem colmax + colsum to store to
        coord_h = (coord_hp, coord_hb, kv_split_idx)
        gM = cute.local_tile(mM, (blk_tile_hp,), coord_h[:-1])
        gM_partial = cute.local_tile(mM_partial, (blk_tile_hp,), coord_h)
        gL_partial = cute.local_tile(mL_partial, (blk_tile_hp,), coord_h)

        # tile predication
        blk_tile_h, blk_tile_p = blk_tile_hp
        blk_tile_n = blk_tile_h * blk_tile_p
        lane_store_max = blk_tile_n == warp_threads or lane_idx < blk_tile_n

        # gmem predication
        grouped_heads, prediction = mM.shape[0]
        cM = cute.make_identity_tensor(mM.shape[0])
        cM = cute.local_tile(cM, blk_tile_hp, coord_hp)
        idx_hg, idx_p = cM[lane_idx]
        lane_store_max &= idx_hg < grouped_heads
        lane_store_max &= idx_p < prediction

        # Load partial colmax and reduce
        cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after barrier
        sM_final_nbar.arrive_and_wait()
        if lane_store_max:
            sM_lane = sM[lane_idx]
            gM_partial[lane_idx] = sM_lane
            gmem_fmax(gM.iterator + gM.layout(lane_idx), sM_lane)

        # Load partial colsum and reduce
        sL_final_nbar.arrive_and_wait()
        if lane_store_max:
            sL_lane_wg = sL[lane_idx, None]
            sL_lane = sL_lane_wg[0] + sL_lane_wg[1] + sL_lane_wg[2] + sL_lane_wg[3]
            gL_partial[lane_idx] = sL_lane

    @staticmethod
    @cute.jit
    def reduction_cluster(
        blk_tile_n: int,
        kv_splits: Int32,
        kv_split_idx: Int32,
        lane_idx: Int32,
        sM_final_nbar: nbar,
        sL_final_nbar: nbar,
        reduction_mbars_ptr: cute.Pointer,
        sM: cute.Tensor,
        sL: cute.Tensor,
        sR: cute.Tensor,
        gL: Optional[cute.Tensor],
        scale_o: Float32,
    ):
        acc_dtype = sM.dtype
        colmax_bits = blk_tile_n * acc_dtype.width
        copy_vec_bits = min(colmax_bits, 128)
        dsmem_store_threads = colmax_bits // copy_vec_bits
        dsmem_store_values = copy_vec_bits // acc_dtype.width
        dsmem_store_atom_r = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyDsmemStoreOp(),
            acc_dtype,
            num_bits_per_copy=copy_vec_bits,
        )
        dsmem_store_r = cute.make_tiled_copy(
            dsmem_store_atom_r,
            cute.make_ordered_layout(
                (dsmem_store_threads, dsmem_store_values), order=(1, 0)
            ),
            (blk_tile_n,),
        )
        thr_store_r = dsmem_store_r.get_slice(lane_idx)
        tRsM = thr_store_r.partition_S(sM)  # (CPY, #CPY)
        tRsL = thr_store_r.partition_S(sL)  # (CPY, #CPY, warpgroup_warps)
        tRsR = thr_store_r.partition_S(sR)  # (CPY, #CPY, max_red_iters, 2)

        tRrM_shape = thr_store_r.partition_D(sM).shape
        tRrM_final = cute.make_rmem_tensor(tRrM_shape, acc_dtype)
        tRrM_prev = cute.make_rmem_tensor(tRrM_shape, acc_dtype)
        tRrL_final = cute.make_rmem_tensor(tRrM_shape, acc_dtype)

        # Wait for last colmax
        cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after barrier
        sM_final_nbar.arrive_and_wait()
        is_reduction_lane = lane_idx < dsmem_store_threads
        if is_reduction_lane:
            tRrM_prev.store(tRsM.load())
            tRrM_final.store(tRrM_prev.load())

            # Cluster butterfly reduction
            for i in cutlass.range_constexpr(max_reduction_iters):
                xor_mask = 0x01 << i
                if xor_mask < kv_splits:
                    peer_idx = kv_split_idx ^ xor_mask
                    tRsR_local = tRsR[None, None, i, 0]
                    tRsR_peer = cute.make_tensor(
                        cute.arch.map_dsmem_ptr(tRsR_local.iterator, peer_idx),
                        tRsR_local.layout,
                    )
                    local_mbar = reduction_mbars_ptr + i
                    peer_mbar = cute.arch.map_dsmem_ptr(local_mbar, peer_idx)
                    cute.copy(
                        dsmem_store_atom_r, tRrM_final, tRsR_peer, mbar_ptr=peer_mbar
                    )
                    cute.arch.fence_acq_rel_cta()  # dont reorder dsmem store after wait
                    cute.arch.mbarrier_wait(local_mbar, phase=0)
                    tRrR = tRsR_local.load()
                    for j in cutlass.range_constexpr(cute.size(tRrM_final)):
                        tRrM_final[j] = cute.arch.fmax(tRrM_final[j], tRrR[j])

        # Wait for last colsum
        sL_final_nbar.arrive_and_wait()
        if is_reduction_lane:
            # Warpgroup reduction
            colsum = tRsL[None, None, 0].load()
            for i in cutlass.range_constexpr(1, warpgroup_warps, 1):
                colsum += tRsL[None, None, i].load()

            # Compute final correction and correct local colsum
            correction = exp2(tRrM_prev.load() - tRrM_final.load())
            correction = correction.reshape(colsum.shape)
            colsum *= correction

            # Cluster butterfly reduction
            for i in cutlass.range_constexpr(max_reduction_iters):
                xor_mask = 0x01 << i
                if xor_mask < kv_splits:
                    peer_idx = kv_split_idx ^ xor_mask
                    tRrL_local = cute.make_rmem_tensor(tRrM_shape, acc_dtype)
                    tRrL_local.store(colsum)
                    tRsR_local = tRsR[None, None, i, 1]
                    tRsR_peer = cute.make_tensor(
                        cute.arch.map_dsmem_ptr(tRsR_local.iterator, peer_idx),
                        tRsR_local.layout,
                    )
                    local_mbar = reduction_mbars_ptr + max_reduction_iters + i
                    peer_mbar = cute.arch.map_dsmem_ptr(local_mbar, peer_idx)
                    cute.copy(
                        dsmem_store_atom_r, tRrL_local, tRsR_peer, mbar_ptr=peer_mbar
                    )
                    cute.arch.fence_acq_rel_cta()  # dont reorder dsmem store after wait
                    cute.arch.mbarrier_wait(local_mbar, phase=0)
                    colsum += tRsR_local.load()

            # Divide by final colsum and store
            rcp_colsum = cute.make_rmem_tensor(colsum.shape, acc_dtype)
            for i in cutlass.range(cute.size(colsum.shape)):
                rcp_colsum[i] = cute.arch.rcp_approx(colsum[i])
            tRsM.store(correction * rcp_colsum.load() * scale_o)

            # Save final colsum for LSE
            if cutlass.const_expr(gL is not None):
                tRrL_final.store(colsum)

        # Notify for final correction
        sM_final_nbar.arrive()

        # Compute and store LSE
        if cutlass.const_expr(gL is not None):
            tRgL = thr_store_r.partition_D(gL)  # (CPY, #CPY=1)
            if kv_split_idx == 0 and is_reduction_lane:
                lse = tRrM_final.load() + cute.math.log2(tRrL_final.load())
                tRgL.store(lse)

    ##############################
    # Reduction Kernel launch
    ##############################
    @staticmethod
    @cute.jit
    def launch_reduction(
        d_per_blk: int,
        o_bshd: cute.Tensor,
        l_bsh: Optional[cute.Tensor],  # LSE output (log2 base) or None
        m_bsh: cute.Tensor,  # colmax_s, already computed
        o_partial_bshd: cute.Tensor,  # partial O per kv split
        l_partial_bsh: cute.Tensor,  # partial colsum_p per kv split
        m_partial_bsh: cute.Tensor,  # partial colmax_s per kv split
        scale_o: Float32,
        stream: cuda.CUstream,
        enable_pdl: bool = True,
    ):
        splits, b, s_q, h_q, d = o_partial_bshd.shape

        # Tile in head_dim first
        def reverse(t: cute.Tensor):
            modes = tuple(reversed(range(cute.rank(t))))
            layout = cute.select(t.layout, modes)
            return cute.make_tensor(t.iterator, layout)

        o_dhsb = reverse(o_bshd)
        l_hsb = reverse(l_bsh) if l_bsh is not None else None
        m_hsb = reverse(m_bsh)
        o_partial_dhsb = reverse(o_partial_bshd)
        l_partial_hsb = reverse(l_partial_bsh)
        m_partial_hsb = reverse(m_partial_bsh)

        d_per_thr = 32 // o_bshd.dtype.width
        thr_per_blk = d_per_blk // d_per_thr
        d_blks = cute.ceil_div(d, d_per_blk)
        smem_bytes = (splits * 2 + 1) * Float32.width // 8

        GroupedQueryAttentionDecode.reduction_kernel(
            (thr_per_blk, d_per_thr, d_per_blk),
            o_dhsb,
            l_hsb,
            m_hsb,
            o_partial_dhsb,
            l_partial_hsb,
            m_partial_hsb,
            scale_o,
        ).launch(
            grid=[d_blks, h_q * s_q, b],
            block=[thr_per_blk, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            smem=smem_bytes,
            min_blocks_per_mp=1,
            use_pdl=enable_pdl,
        )

    @staticmethod
    @cute.kernel
    def reduction_kernel(
        tile_d: cute.Tile,
        o_dhsb: cute.Tensor,
        l_hsb: Optional[cute.Tensor],
        m_hsb: cute.Tensor,
        o_partial_dhsb: cute.Tensor,
        l_partial_hsb: cute.Tensor,
        m_partial_hsb: cute.Tensor,
        scale_o: Float32,
    ):
        thr_per_blk, d_per_thr, d_per_blk = tile_d
        d, h_q, s_q, b, splits = o_partial_dhsb.shape
        d_blk_idx, coord_hs, coord_b = cute.arch.block_idx()
        coord_h, coord_s = cute.idx2crd(coord_hs, (h_q, s_q))
        tidx, _, _ = cute.arch.thread_idx()

        not_oob_d = True
        if d % d_per_blk != 0:
            not_oob_d = d_blk_idx * d_per_blk + tidx * d_per_thr < d

        coord_o = (d_blk_idx, coord_h, coord_s, coord_b, None)
        gO = cute.local_tile(o_dhsb, (d_per_blk,), coord_o[:-1])
        gO_partial = cute.local_tile(o_partial_dhsb, (d_per_blk,), coord_o)

        gM = cute.local_tile(m_hsb, (1,), coord_o[1:-1])
        gM_partial = cute.local_tile(m_partial_hsb, (1,), coord_o[1:])
        gL_partial = cute.local_tile(l_partial_hsb, (1,), coord_o[1:])
        gM_partial_0 = gM_partial[None, tidx]
        gL_partial_0 = gL_partial[None, tidx]

        smem_ptr = cute.arch.get_dyn_smem(Float32)
        partial_layout = cute.make_layout((1, splits))
        sM_partial = cute.make_tensor(smem_ptr, partial_layout)
        sL_partial = cute.make_tensor(smem_ptr + splits, partial_layout)
        sL_partial_0 = sL_partial[None, tidx]
        sM_partial_0 = sM_partial[None, tidx]

        sM_layout = cute.make_layout(1)
        sM = cute.make_tensor(smem_ptr + splits * 2, sM_layout)

        cpasync_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), Float32, num_bits_per_copy=32
        )

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32)
        tv_layout = cute.make_ordered_layout((thr_per_blk, d_per_thr), order=(1, 0))
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, (d_per_blk,))
        thr_copy = tiled_copy.get_slice(tidx)

        tCgO_partial = thr_copy.partition_S(gO_partial)  # (CPY, #CPY=1, splits)
        tCgO_partial = tCgO_partial[None, 0, None]  # (CPY, splits)
        tCgO = thr_copy.partition_D(gO)  # (CPY, #CPY=1)
        tCgO = tCgO[None, 0]  # (CPY)
        tCrO_final = cute.zeros_like(tCgO, Float32)

        cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after PDL wait
        cute.arch.griddepcontrol_wait()

        if tidx == 0:
            cute.copy(cpasync_atom, gM, sM)

        if tidx < splits:
            cute.copy(cpasync_atom, gL_partial_0, sL_partial_0)
            cute.copy(cpasync_atom, gM_partial_0, sM_partial_0)

        for split_idx in cutlass.range(thr_per_blk + tidx, splits, thr_per_blk):
            gL_partial_n = gL_partial[None, split_idx]
            sL_partial_n = sL_partial[None, split_idx]
            cute.copy(cpasync_atom, gL_partial_n, sL_partial_n)

            gM_partial_n = gM_partial[None, split_idx]
            sM_partial_n = sM_partial[None, split_idx]
            cute.copy(cpasync_atom, gM_partial_n, sM_partial_n)

        cute.arch.cp_async_commit_group()
        # The first `o_pf` splits' partial outputs are loaded here, before the wait for the
        # partial maxima and sums, so the two loads overlap instead of following each
        # other. Each load reads an existing split (the index is clamped), so it is in
        # range whether or not that split produced a row; the loop below accumulates a
        # split's output only where it did before, in the same order, so the result is
        # unchanged. Without whole d blocks (`not_oob_d` dynamic) nothing is prefetched.
        o_pf = 16 if cutlass.const_expr(d % d_per_blk == 0) else 0
        rO_pf = [
            tCgO_partial[None, cutlass.min(Int32(j), splits - 1)].load()
            for j in range(o_pf)
        ]
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        max_final = sM[0]
        sum_final = Float32(0)
        if max_final > -Float32.inf and not_oob_d:
            for j in cutlass.range_constexpr(o_pf):
                if j < splits:
                    max_partial = sM_partial[0, j]
                    if max_partial > -Float32.inf:
                        correction = exp2(max_partial - max_final)
                        sum_final += correction * sL_partial[0, j]
                        tCrO_final += correction * rO_pf[j]
            for split_idx in cutlass.range(o_pf, splits, unroll=8):
                max_partial = sM_partial[0, split_idx]
                if max_partial > -Float32.inf:
                    correction = exp2(max_partial - max_final)
                    sum_final += correction * sL_partial[0, split_idx]
                    tCrO_final += correction * tCgO_partial[None, split_idx].load()
            tCrO_final *= cute.arch.rcp_approx(sum_final) * scale_o

        cute.arch.griddepcontrol_launch_dependents()

        if not_oob_d:
            tCgO.store(tCrO_final.to(o_dhsb.dtype))

        if cutlass.const_expr(l_hsb is not None):
            if d_blk_idx == 0 and tidx == 0:
                l_hsb[coord_h, coord_s, coord_b] = max_final + cute.math.log2(sum_final)

        return

