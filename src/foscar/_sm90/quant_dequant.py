# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: MIT
"""QuantDequantSm90: the paged dequantize of the cache `IntPingPongDecodeSm90` reads, on Hopper.

One launch per layer reads K and V of the pages a call names back into bf16 rows
(`QuantKVCache.dequantize`). The request search, the scale records, the arithmetic and the traps
are `_dequant_core`'s; this module reads the Hopper code pools.

THE CODE POOLS hold, per (page, KV head), one row per thread of the decode's warpgroup: the
quantized values that thread feeds to its tensor-core multiplies, V transposed, in the reader's
chunk16 order (`serving_cache`, CODE POOL ORDERS). A row is not a token row: a K row holds two
tokens, and a V row two channels of every token, so the values reach token rows through shared
memory.

STAGE, THEN GATHER. The CTA's first warpgroup copies the (page, head) block of K into shared
memory and the second the block of V, each thread its own row, with the 16-byte loads of the
decode and one 16-byte store per chunk (`serving_cache.dequant_stage_word`, a swizzle of the
16-byte slots). After one barrier the CTA works like the Blackwell kernel: each thread takes
output units of 8 channels of one token, consecutive threads consecutive units of a token row,
and for each unit gathers its K and V values from the staged blocks
(`serving_cache.dequant_unit_gathers`: four words of four consecutive rows for K, whose channel
pairs share a word, and eight words for V), converts them with the token's scale record and
writes K's and V's 16 bytes. The swizzle puts every gather and every staging store at one word
per bank (`tests/test_sm90_dequant_banks_cpu.py`). The CTA loads the scale records of all its
units and the code rows before the request search and converts only the units of tokens below
the page's length, so a request's last page costs its tokens.

WHY THE CTA STAGES THE BLOCK INSTEAD OF TRANSPOSING IT. The alternative converts each thread's
row in registers, values in their fragment order, and transposes the bf16 values through a
swizzled [64, D] tile. Staging the compressed block moves an eighth (INT2) or a quarter (INT4) of
the shared-memory bytes that tile moves, and a unit skips its gathers outright past the page's
length. Over the benchmark's cases this form measured faster or equal at every size and shape
tried, most at small sizes, on nearly empty last pages and at head dimension 256, and equal at
the largest sizes at head dimension 128.
"""
import functools

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from .._append_core import PAGE, bf16x2, f32_from_bits, layout_key, ld_nc_b32x2, pool_fake, st_b32
from .._dequant_core import (
    MAGIC, MAGIC_BITS, UNIT_CHANNELS, QuantDequantBase, code_value, compile_dequant, fma_rn,
    launch_dequant, ld_nc_b32x4)
from .serving_cache import (
    code_block_shape, code_pool_shape, code_position, code_row_bytes, code_word_offset,
    dequant_stage_word, dequant_unit_gathers, require_code_pool)

#: One warpgroup stages each tensor's block, and all 256 threads convert. Against 128 threads
#: (which stage both blocks each) and 512, this count is the fastest or as fast at every size
#: tried: 128 threads are slower at head dimension 256 and small sizes, and 512 slower on ragged
#: pages.
THREADS = 256


class QuantDequantSm90(QuantDequantBase):
    """Constructed from a `QuantKVLayout` and the out_indptr dtype; raises outside the cache
    writer's envelope (`_append_core.envelope_refusal`), whose layouts are the ones it reads."""

    ARCH = "sm90"

    def __init__(self, layout, out_indptr_64):
        super().__init__(layout, out_indptr_64)
        self.code_pool_order = layout.code_pool_order
        self.k_row_bytes, self.v_row_bytes = code_row_bytes(self.head_dim, PAGE, self.bits)
        code_block_shape(self.code_pool_order, self.k_row_bytes)    # refuses an unknown order
        self.units = self.head_dim // UNIT_CHANNELS
        self.tokens_per_pass = THREADS // self.units
        self.passes = PAGE // self.tokens_per_pass
        self.k_gathers = dequant_unit_gathers("K", self.bits, self.head_dim)
        self.v_gathers = dequant_unit_gathers("V", self.bits, self.head_dim)

    @cute.jit
    def _stage(self, tensor: cutlass.Constexpr, mPool, sStage, page, head, row):
        """Copy thread row `row` of the (page, head) block of `tensor` into shared memory, one
        16-byte load and one 16-byte store per chunk; V's words go into their chunk in the
        order `dequant_stage_word` permutes them to, which is a choice of registers."""
        per = self.k_row_bytes if tensor == "K" else self.v_row_bytes
        for chunk in cutlass.range_constexpr(max(per // 16, 1)):
            offset = code_word_offset(self.code_pool_order, per, self.num_kv_heads, page, head,
                                      row, 4 * chunk, wide=Int64)
            if cutlass.const_expr(per == 8):
                words = ld_nc_b32x2(mPool.iterator.toint() + offset * Int64(4)) + [Int32(0), Int32(0)]
            else:
                words = ld_nc_b32x4(mPool.iterator.toint() + offset * Int64(4))
            # Word 4 * chunk + i lands at slot word i ^ swap of the chunk's 16 bytes.
            swap = dequant_stage_word(tensor, 0, 4 * chunk) % 4
            staged = cute.make_rmem_tensor(4, Int32)
            for i in cutlass.range_constexpr(4):
                staged[i] = words[i ^ swap]
            cute.autovec_copy(staged, cute.make_tensor(
                (sStage.iterator + (dequant_stage_word(tensor, row, 4 * chunk) - swap)).align(16),
                cute.make_layout(4)))

    @cute.jit
    def _unit(self, tensor: cutlass.Constexpr, sStage, token, unit, s, z):
        """The four bf16 pairs of output unit `unit` of `token` of `tensor`, gathered from the
        staged block: in K the word of channels 2i and 2i + 1 holds them 16 bits apart, in V
        each channel is a word's field of its own."""
        bits = self.bits
        qmax = (1 << bits) - 1
        row, word, field = code_position(tensor, bits, token, UNIT_CHANNELS * unit)
        pairs = []
        if cutlass.const_expr(tensor == "K"):
            for i in cutlass.range_constexpr(UNIT_CHANNELS // 2):
                dr, dw, df = self.k_gathers[2 * i]
                x = sStage[dequant_stage_word("K", row + dr, word + dw)]
                lanes = (x >> (Int32(bits) * (field + df))) & Int32(qmax | (qmax << 16))
                pairs.append(bf16x2(fma_rn(s, code_value(lanes, 0), z),
                                    fma_rn(s, code_value(lanes, 2), z)))
        else:
            x = []
            for i in cutlass.range_constexpr(UNIT_CHANNELS):
                dr, dw, df = self.v_gathers[i]
                w = sStage[dequant_stage_word("V", row + dr, word + dw)]
                q = ((w >> (Int32(bits) * (field + df))) & Int32(qmax)) | Int32(MAGIC_BITS)
                x.append(fma_rn(s, f32_from_bits(q) - Float32(MAGIC), z))
            for i in cutlass.range_constexpr(UNIT_CHANNELS // 2):
                pairs.append(bf16x2(x[2 * i], x[2 * i + 1]))
        return pairs

    @cute.jit
    def _store(self, mOut, pairs, out_row, head, unit):
        st_b32(mOut.iterator.toint()
               + (out_row * Int64(mOut.layout.stride[0]) + Int64(head) * Int64(self.head_dim)
                  + Int64(UNIT_CHANNELS) * Int64(unit)) * Int64(2), pairs)

    @cute.kernel
    def kernel(self, mKq, mVq, mScale, mKo, mVo, mPageIndptr, mPageIndices, mSeqLens,
               mOutIndptr):
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        row = tidx % 128
        p = cta // Int32(self.num_kv_heads)
        head = cta - p * Int32(self.num_kv_heads)
        num_pages = Int32(mKq.shape[0])
        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(Int32, cute.make_layout(max(self.k_row_bytes, 16) * 32),
                                  byte_alignment=16)
        sV = smem.allocate_tensor(Int32, cute.make_layout(max(self.v_row_bytes, 16) * 32),
                                  byte_alignment=16)
        if p < Int32(mPageIndices.shape[0]):
            page, safe = self._page_of(mPageIndices, num_pages, p)
            unit = tidx % Int32(self.units)
            sub = tidx // Int32(self.units)
            # Request-independent work (`_dequant_core`, GRID): the staging of both blocks and
            # the loads of every unit's scale record.
            if tidx < 128:
                self._stage("K", mKq, sK, safe, head, row)
            else:
                self._stage("V", mVq, sV, safe, head, row)
            k_group = unit // Int32(self.units // 2) if self.k_half else 0
            records = []
            for i in cutlass.range_constexpr(self.passes):
                records.append(self._token_record(
                    mScale, safe, head, Int32(i * self.tokens_per_pass) + sub, k_group))
            active, n, row0 = self._locate(mPageIndptr, mSeqLens, mOutIndptr, mKo, mVo, p, lane)
            cute.arch.barrier()
            if active != 0:
                self._require_page(page, num_pages)
                for i in cutlass.range_constexpr(self.passes):
                    token = Int32(i * self.tokens_per_pass) + sub
                    if token < n:
                        (sk, zk), (sv, zv) = self._record_pairs(records[i])
                        self._store(mKo, self._unit("K", sK, token, unit, sk, zk),
                                    row0 + Int64(token), head, unit)
                        self._store(mVo, self._unit("V", sV, token, unit, sv, zv),
                                    row0 + Int64(token), head, unit)
        else:
            self._validate(mPageIndptr, mPageIndices, mSeqLens, cta, tidx, THREADS)

    @cute.jit
    def __call__(self, mKq, mVq, mScale, mKo, mVo, mPageIndptr, mPageIndices, mSeqLens,
                 mOutIndptr, num_ctas: Int32, stream):
        # The reader's order and C-contiguity, refused at trace time, because the loads address
        # each (page, head) block as one dense run in that order.
        require_code_pool("k_codes", mKq.shape, self.code_pool_order, self.k_row_bytes,
                          stride=mKq.stride)
        require_code_pool("v_codes", mVq.shape, self.code_pool_order, self.v_row_bytes,
                          stride=mVq.stride)
        self.kernel(mKq, mVq, mScale, mKo, mVo, mPageIndptr, mPageIndices, mSeqLens,
                    mOutIndptr).launch(grid=[num_ctas, 1, 1], block=[THREADS, 1, 1],
                                       stream=stream)


@functools.cache
def _compile(key, out_indptr_64):
    from types import SimpleNamespace
    layout = SimpleNamespace(**dict(key))
    kern = QuantDequantSm90(layout, out_indptr_64)
    num_pages = cute.sym_int()
    pools = [pool_fake(code_pool_shape(kern.code_pool_order, num_pages, kern.num_kv_heads, per))
             for per in (kern.k_row_bytes, kern.v_row_bytes)]
    return compile_dequant(kern, __file__, layout, (*pools, num_pages))


def compile_quant_dequant(layout, out_indptr_64):
    """The compiled Hopper dequantize for `layout` and the out_indptr dtype (int64 when True),
    memoized and taken from the persistent kernel cache when an identical trace exists."""
    return _compile(layout_key(layout), bool(out_indptr_64))


#: Launches a compiled dequantize with this kernel's thread count, which sizes its checking CTAs
#: (`_dequant_core`, GRID).
launch_quant_dequant = functools.partial(launch_dequant, threads=THREADS)
