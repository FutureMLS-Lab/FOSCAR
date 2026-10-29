# Copyright 2026 Donglin Zhuang
# SPDX-License-Identifier: Apache-2.0
"""IntAppend: fused token-wise quantize and fragment-order scatter append into a ServingCache.

This module is the device write path for serving_cache.ServingCache. Given new K/V token
vectors, it quantizes them token-wise and writes the resulting values and their (scale, zero)
pairs directly into the global paged fragment-order pool that IntPingPongDecodeSm90, the SM90
decode kernel, reads. The write matches the host path that builds the same pool from
chunk16-ordered thread rows (`serving_cache.make_serving_cache`), value for value.

CONTRACT (writes serving_cache.ServingCache, exactly):
  codes_k uint8 [NP, HKV, per_k/16, 128, 16]  /  codes_v uint8 [NP, HKV, per_v/16, 128, 16]
        in the reader's code-pool order, IntPingPongDecodeSm90.code_pool_order ("chunk16"): chunk
        c of thread row t = bytes [16c, 16c+16) of that row (per = 32B int4 / 16B int2 at D=128,
        64B / 32B at D=256, `serving_cache.code_row_bytes`; row = warpgroup thread; "per" =
        plane-word A-frag bytes).  V is in V^T orientation.
  scales  <sdtype> [NP, HKV, N, 4]      last dim = (s_k, z_k, s_v, z_v); N = 64 (page = one tile).

INPUTS to __call__:
  mK, mV  bf16 [num_new, HKV, D]        new token vectors (K and V) to append, D in {128, 256}.
  mSlot   int32 [num_new, 2]            per new token: (global_page, intra-tile pos t in [0, N)).
  codes_k, codes_v, scales              the ServingCache tensors to write INTO (must be ZEROED at
                                        allocation: the scatter is an RMW-OR, so codes accumulate).
The scatter map of the head dimension (`serving_cache.scatter_map_file`: `frag_scatter_k16.pt` at
D=128, `frag_scatter_k16_d256.pt` at D=256) is loaded on the host by `_ensure_scatter` before
compile; `__call__` reads it as gmem int32 [N, D, 3] tensors.

QUANT (token-local over the D channels, no cross-token dependency):  z = min_d x ;  s = (max_d x
- min_d x)/qmax clamp 1e-8 ;  q = round((x - z)/s).clamp(0, qmax).  qmax = 2^bits - 1.  round =
round-to-nearest-even (cvt.rni.s32.f32, which matches torch.round's tie rule, so the written
values equal the host's).

BF16 ARITHMETIC: the `_bf()` helper rounds each intermediate arithmetic result through bf16
(Float32 -> BFloat16 -> Float32) because the host quantizer keeps all arithmetic in bf16: the
K/V inputs are bf16 and are never up-cast to f32.  Matching that rounding is what makes the
written values equal the host's.  If the host quantizer ever moves to f32 arithmetic, the
`_bf()` rounding in `_quant` and in `kernel` must move with it; changing one side alone breaks
the match even though individual numbers may look close.

SCATTER (shared read-modify-write, disjoint bit-fields):
a single token's D channel codes land in plane-words shared with OTHER tokens of the same tile,
but no two tokens ever claim the same (row, word, field_bit).  So the append RMW-ORs each code's
bits into the destination u32 word (preserving the bits other tokens already wrote).  Concurrent
writers within ONE CTA (different channels) hit DISJOINT fields of possibly-SHARED words ->
atomic_or makes that race-free; concurrent CTAs (different new tokens) either target different pages
or different t within a page, and within a page write disjoint fields, so atomic_or is correct
across the whole launch.  The scatter map of the head dimension:
  scatter_k/v[bits] int32 [N=64 token, D chan, 3] = (thread_row, word_index, field_bit) ;
  to place code c of (token t, channel ch): OR (c << field_bit) into u32 `word_index` of dest row.
  The map names a word of a thread ROW.  Where that word sits in the pool is the reader's
  code-pool order, and `serving_cache.code_word_offset` gives it: word w of row t is at word
  t*thread + (w // 4)*chunk + w % 4 of the (page, kv-head) block, which for chunk16 is
  (chunk, row, word-in-chunk) = (w // 4, t, w % 4).  That is the one place the order enters.
The V^T orientation and the D/64 stacked output halves are already baked into scatter_v (an
absolute word index over the per_v row), so the V path follows scatter_v with no extra transpose.

GRID / THREAD DESIGN:
  grid  = [num_new, 1, 1]                one CTA per new token (covers single-token decode append AND
                                         N-aligned bulk = num_new CTAs in flight).
  block = [32 * HKV, 1, 1]               one WARP per kv-head; warp w = head hk=w.
  Per warp the D channels of (token i, head hk) map across the 32 lanes as D/32 elems/lane:
  lane L owns channels L + 32*e, e in [0, D/32).  min/max is a lane-local reduce over the owned
  channels then a warp butterfly (offsets 1,2,4,8,16); s,z are then warp-uniform.  Each lane
  quantizes its channels and atomic_or-scatters them; lane 0 writes the (s_k,z_k,s_v,z_v) vector
  store to scales[page, hk, t, :].  At D=128 the body holds each lane's four owned values in four
  scalars; any other D holds them in two register tensors of D/32 elements, with the same reduce,
  butterfly and scatter.
"""
import os

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, BFloat16
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from ..quant_layout import SM90_HEAD_DIMS
from .int_decode_pingpong import IntPingPongDecodeSm90
from .serving_cache import code_row_bytes, code_word_offset, require_code_pool, scatter_map_file

_S_CLAMP = 1.0e-8
_SCRIPTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts"))


@dsl_user_op
def _cvt_rni_s32(x, *, loc=None, ip=None):
    """f32 -> s32 with round-to-nearest-even (cvt.rni.s32.f32); matches torch.round tie rule."""
    return Int32(llvm.inline_asm(
        T.i32(), [Float32(x).ir_value(loc=loc, ip=ip)],
        "cvt.rni.s32.f32 $0, $1;", "=r,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


def _fmin(a, b):
    return -cute.arch.fmax(-a, -b)


@cute.jit
def _bf(x):
    """Round an f32 through bf16 and back, modeling the host's bf16 arithmetic.  The host
    quantizer computes s, z and q in bf16 arithmetic (the K/V inputs are bf16 and are never cast
    to f32), so the device rounds each arithmetic step to bf16 to match it:
    s = bf(bf(mx-mn)/qmax), q = round(bf(bf(x-z)/s))."""
    return Float32(BFloat16(x))


class IntAppend:
    def __init__(self, head_dim, page_size=64, bits=4, sdtype=Float32):
        # The head dimensions the reader decodes, each of which has its own scatter map
        # (`serving_cache.scatter_map_file`). A raise, not an assert: `python -O` removes an
        # assert, and a head dimension without its map would write codes where no reader looks.
        if head_dim not in SM90_HEAD_DIMS:
            raise ValueError(f"IntAppend writes head_dim in {SM90_HEAD_DIMS}, the head "
                             f"dimensions IntPingPongDecodeSm90 decodes; got {head_dim}")
        assert page_size == 64, "page size N is fixed at 64 (the m64 wgmma tile)"
        assert bits in (4, 2)
        self.head_dim = head_dim
        self.page_size = page_size
        self.bits = bits
        self.qmax = (1 << bits) - 1
        self.sdtype = sdtype
        # THE POOL IS THE READER'S, asked of it rather than restated: the append follows
        # the decode kernel's code-pool order. The thread-row bytes come from the pool
        # geometry the reader uses too (`serving_cache`, CODE POOL GEOMETRY), and from both,
        # `code_word_offset` places each row word in the pool, which is all `kernel`'s
        # addressing uses.
        self.code_pool_order = IntPingPongDecodeSm90.code_pool_order
        self.k_code_bytes_per_thread, self.v_code_bytes_per_thread = code_row_bytes(
            head_dim, page_size, bits)
        # scatter maps loaded + uploaded lazily on first __call__ (device tensors, int32 [N,D,3]).
        self._scat_k = None
        self._scat_v = None

    def _ensure_scatter(self, device):
        if self._scat_k is not None:
            return
        d = torch.load(os.path.join(_SCRIPTS_DIR, scatter_map_file(self.head_dim)))
        self._scat_k = d["scatter_K"][self.bits].to(device=device, dtype=torch.int32).contiguous()
        self._scat_v = d["scatter_V"][self.bits].to(device=device, dtype=torch.int32).contiguous()

    @cute.kernel
    def kernel(self, mK, mV, mSlot, mCodesK, mCodesV, mScale, mScatK, mScatV):
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        # 64-BIT INDEXING: the token index and the page are widened BEFORE either is
        # multiplied into an offset. In 32 bits at HKV = 8, mK passes 2^31 elements at 2^21
        # new tokens, a code pool passes 2^31 words at page 262,144 (int4, 8 GiB) and the scale
        # pool passes 2^31 elements at page 2^20, and past those the product wraps, so the
        # write lands outside its page (8 GiB before the pool at int4 page 262,144; exactly on
        # page 0 at int2 page 2^20). Every other index (lane, head, t, the scatter map's row
        # and word) is bounded by the page geometry and stays 32-bit.
        i = Int64(cta)                          # this CTA's new token
        qmax = Float32(self.qmax)
        lane = tidx % 32
        hk = tidx // 32                         # warp index == kv-head

        # this token's slot: (global page, intra-tile position t)
        page = mSlot[(i, 0)]
        t = mSlot[(i, 1)]

        # The D=128 body holds each lane's four owned values in four scalars; any other head
        # dimension holds them in register tensors of D/32 elements (the module docstring).
        if cutlass.const_expr(self.head_dim == 128):
            # ---- load the 4 channels this lane owns (channel = lane + 32*e) ----
            x_k0 = Float32(mK[(i, hk, lane + 0)]);   x_v0 = Float32(mV[(i, hk, lane + 0)])
            x_k1 = Float32(mK[(i, hk, lane + 32)]);  x_v1 = Float32(mV[(i, hk, lane + 32)])
            x_k2 = Float32(mK[(i, hk, lane + 64)]);  x_v2 = Float32(mV[(i, hk, lane + 64)])
            x_k3 = Float32(mK[(i, hk, lane + 96)]);  x_v3 = Float32(mV[(i, hk, lane + 96)])

            # lane-local min/max over the 4 owned channels
            mxk = cute.arch.fmax(cute.arch.fmax(x_k0, x_k1), cute.arch.fmax(x_k2, x_k3))
            mnk = _fmin(_fmin(x_k0, x_k1), _fmin(x_k2, x_k3))
            mxv = cute.arch.fmax(cute.arch.fmax(x_v0, x_v1), cute.arch.fmax(x_v2, x_v3))
            mnv = _fmin(_fmin(x_v0, x_v1), _fmin(x_v2, x_v3))

            # warp butterfly reduce (offsets 1,2,4,8,16 -> all 32 lanes agree)
            for off in cutlass.range_constexpr(5):
                o = 1 << off
                mxk = cute.arch.fmax(mxk, cute.arch.shuffle_sync_bfly(mxk, offset=o))
                mxv = cute.arch.fmax(mxv, cute.arch.shuffle_sync_bfly(mxv, offset=o))
                mnk = _fmin(mnk, cute.arch.shuffle_sync_bfly(mnk, offset=o))
                mnv = _fmin(mnv, cute.arch.shuffle_sync_bfly(mnv, offset=o))

            # s = bf(bf(max-min) clamp 1e-8 / qmax) ; z = min  (bf16 arithmetic, the host
            # quantizer's rule; mx,mn are exact bf16 — comparisons don't round — so only the
            # sub/div need bf16 rounding).
            sk = _bf(cute.arch.fmax(_bf(mxk - mnk), Float32(_S_CLAMP)) / qmax)
            sv = _bf(cute.arch.fmax(_bf(mxv - mnv), Float32(_S_CLAMP)) / qmax)
            zk = mnk
            zv = mnv

            # u32 views of the code pools (atomic_or into plane-words). The pool is dense
            # (`__call__`), and `code_word_offset` gives the word of (page, kv-head, row, word) in
            # the reader's order, with the page widened to 64 bits before it is multiplied.
            cuK = cute.recast_tensor(mCodesK, Int32)
            cuV = cute.recast_tensor(mCodesV, Int32)
            hkv_k, hkv_v = mCodesK.shape[1], mCodesV.shape[1]
            order, per_k, per_v = self.code_pool_order, self.k_code_bytes_per_thread, self.v_code_bytes_per_thread

            # ---- quantize + scatter each of the 4 owned channels ----
            for e in cutlass.range_constexpr(4):
                ch = lane + 32 * e
                xk = (x_k0 if e == 0 else (x_k1 if e == 1 else (x_k2 if e == 2 else x_k3)))
                xv = (x_v0 if e == 0 else (x_v1 if e == 1 else (x_v2 if e == 2 else x_v3)))
                qk = self._quant(xk, zk, sk, qmax)
                qv = self._quant(xv, zv, sv, qmax)
                # K scatter: scatter_k[t, ch] = (row, word, bit)
                rowk = mScatK[(t, ch, 0)]
                wordk = mScatK[(t, ch, 1)]
                bitk = mScatK[(t, ch, 2)]
                offk = code_word_offset(order, per_k, hkv_k, page, hk, rowk, wordk, wide=Int64)
                cute.arch.atomic_or(cuK.iterator + offk, qk << bitk)
                # V scatter: scatter_v[t, ch] = (row, word, bit)  (V^T orientation baked in)
                rowv = mScatV[(t, ch, 0)]
                wordv = mScatV[(t, ch, 1)]
                bitv = mScatV[(t, ch, 2)]
                offv = code_word_offset(order, per_v, hkv_v, page, hk, rowv, wordv, wide=Int64)
                cute.arch.atomic_or(cuV.iterator + offv, qv << bitv)

        else:
            # ---- load the D/32 channels this lane owns (channel = lane + 32*e) ----
            owned = self.head_dim // 32
            rXk = cute.make_rmem_tensor(owned, Float32)
            rXv = cute.make_rmem_tensor(owned, Float32)
            for e in cutlass.range_constexpr(owned):
                rXk[e] = Float32(mK[(i, hk, lane + 32 * e)])
                rXv[e] = Float32(mV[(i, hk, lane + 32 * e)])

            # lane-local min/max over the owned channels (exact, so the order does not matter)
            mxk = rXk[0]; mnk = rXk[0]; mxv = rXv[0]; mnv = rXv[0]
            for e in cutlass.range_constexpr(owned - 1):
                mxk = cute.arch.fmax(mxk, rXk[e + 1]); mnk = _fmin(mnk, rXk[e + 1])
                mxv = cute.arch.fmax(mxv, rXv[e + 1]); mnv = _fmin(mnv, rXv[e + 1])

            # warp butterfly reduce (offsets 1,2,4,8,16 -> all 32 lanes agree)
            for off in cutlass.range_constexpr(5):
                o = 1 << off
                mxk = cute.arch.fmax(mxk, cute.arch.shuffle_sync_bfly(mxk, offset=o))
                mxv = cute.arch.fmax(mxv, cute.arch.shuffle_sync_bfly(mxv, offset=o))
                mnk = _fmin(mnk, cute.arch.shuffle_sync_bfly(mnk, offset=o))
                mnv = _fmin(mnv, cute.arch.shuffle_sync_bfly(mnv, offset=o))

            # s and z exactly as the D=128 body computes them
            sk = _bf(cute.arch.fmax(_bf(mxk - mnk), Float32(_S_CLAMP)) / qmax)
            sv = _bf(cute.arch.fmax(_bf(mxv - mnv), Float32(_S_CLAMP)) / qmax)
            zk = mnk
            zv = mnv

            # the pool views and offsets of the D=128 body
            cuK = cute.recast_tensor(mCodesK, Int32)
            cuV = cute.recast_tensor(mCodesV, Int32)
            hkv_k, hkv_v = mCodesK.shape[1], mCodesV.shape[1]
            order, per_k, per_v = self.code_pool_order, self.k_code_bytes_per_thread, self.v_code_bytes_per_thread

            # ---- quantize + scatter each owned channel ----
            for e in cutlass.range_constexpr(owned):
                ch = lane + 32 * e
                qk = self._quant(rXk[e], zk, sk, qmax)
                qv = self._quant(rXv[e], zv, sv, qmax)
                rowk = mScatK[(t, ch, 0)]
                wordk = mScatK[(t, ch, 1)]
                bitk = mScatK[(t, ch, 2)]
                offk = code_word_offset(order, per_k, hkv_k, page, hk, rowk, wordk, wide=Int64)
                cute.arch.atomic_or(cuK.iterator + offk, qk << bitk)
                rowv = mScatV[(t, ch, 0)]
                wordv = mScatV[(t, ch, 1)]
                bitv = mScatV[(t, ch, 2)]
                offv = code_word_offset(order, per_v, hkv_v, page, hk, rowv, wordv, wide=Int64)
                cute.arch.atomic_or(cuV.iterator + offv, qv << bitv)

        # ---- scale vector store (lane 0 of each warp): scales[page, hk, t, :] = (sk,zk,sv,zv) ----
        # (the page coordinate is 64-bit, so the page stride is multiplied in 64 bits)
        page64 = Int64(page)
        if lane == 0:
            ty = self.sdtype
            mScale[(page64, hk, t, 0)] = ty(sk)
            mScale[(page64, hk, t, 1)] = ty(zk)
            mScale[(page64, hk, t, 2)] = ty(sv)
            mScale[(page64, hk, t, 3)] = ty(zv)

    @cute.jit
    def _quant(self, x, z, s, qmax):
        """q = clamp(round(bf(bf(x - z) / s)), 0, qmax) as Int32, round-to-nearest-even via
        cvt.rni, with the sub and div bf16-rounded to match the host's bf16 arithmetic exactly.
        Rounding then clamping equals clamping the f32 ratio then rounding, because the bounds
        are integers; so the bf16 ratio is clamped in float and then cvt.rni'd, which is monotone
        and gives the identical integer result."""
        ratio = _bf(_bf(x - z) / s)                       # bf16(bf16(x-z)/s) — the host's pre-round value
        ratio = cute.arch.fmax(Float32(0.0), _fmin(ratio, qmax))
        return _cvt_rni_s32(ratio)

    @cute.jit
    def __call__(self, mK, mV, mSlot, codes_k, codes_v, scales, mScatK, mScatV, stream):
        # The reader's order and C-contiguity, because `kernel` addresses each block as dense.
        require_code_pool("codes_k", codes_k.shape, self.code_pool_order, self.k_code_bytes_per_thread,
                          stride=codes_k.stride)
        require_code_pool("codes_v", codes_v.shape, self.code_pool_order, self.v_code_bytes_per_thread,
                          stride=codes_v.stride)
        num_new = mK.shape[0]
        HKV = mK.shape[1]
        self.kernel(mK, mV, mSlot, codes_k, codes_v, scales, mScatK, mScatV).launch(
            grid=[num_new, 1, 1], block=[32 * HKV, 1, 1], stream=stream)
