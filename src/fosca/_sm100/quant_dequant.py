# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""QuantDequantSm100: the paged dequantize of the cache `IntDecodeSm100` reads, on Blackwell.

One launch per layer reads K and V of the pages a call names back into bf16 rows
(`QuantKVCache.dequantize`). The request search, the scale records, the arithmetic and the traps
are `_dequant_core`'s; this module reads the Blackwell code pools.

THE CODE POOLS hold one row per token, D * bits / 8 bytes packing its D channels, K and V alike
(`_sm100/code_layout.py`). Channel e sits in word e // W, W = 32 // bits, so the eight channels
8u to 8u + 7 of output unit u are half a word at INT2 (word u // 2, the fields of parity u % 2)
and a whole word at INT4 (word u). `code_layout.code_field` is the authority; the constructor
checks the byte-lane extraction below against it for every channel.

THE WORK. A thread takes one unit of one token at a time, for K and V together: lanes
consecutive in the thread index take consecutive units of one token row, so a warp writes whole
2 * D-byte head rows of k_out and v_out, one 16-byte store per unit and tensor. Each unit reads
its K and V code words and its token's scale record straight from global memory with
`ld.global.nc`; the lanes of one token read the same record and the same few words, which the L1
serves once. The CTA covers the page's 64 tokens in `passes` passes of `tokens_per_pass` tokens
and issues the loads of every pass, 8 passes of 6 words at the shipped layout, before the
request search, then converts the units below the page's length. On B200, issuing every pass up
front is faster in nearly every case than a software pipeline that loads 1, 2 or 4 passes ahead
of the conversion: shallower pipelines lose most at small sizes. The same holds against 256 and
512 threads per CTA: 256 is faster only marginally at the smallest sizes and slower at large
sizes and on nearly empty last pages. It holds 48 words of loads per thread: the builds use 48
to 66 registers, 7 to 10 CTAs per SM, and `ptxas` spills one 8-byte slot (one store and one load
per thread) only in INT2 with K token-wise over D/2 and an int32 out_indptr.
"""
import functools

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from .._append_core import PAGE, bf16x2, layout_key, pool_fake, st_b32
from .._dequant_core import (
    UNIT_CHANNELS, QuantDequantBase, byte_lanes, code_value, compile_dequant, field_lanes, fma_rn,
    launch_dequant, ld_nc_b32)
from .code_layout import code_field, code_row_index

#: Threads per CTA: 8 tokens per pass at head dimension 128, so the page is 8 passes.
THREADS = 128


def unit_fields(bits, head_dim, unit):
    """(word, ((lane word k, byte), ...) for the unit's 8 channels in order): where the eight
    channels of output unit `unit` sit in a token row of `head_dim` channels, read from
    `code_layout.code_field` and put in `_dequant_core.field_lanes` form."""
    per_word = 32 // bits
    where = {code_field(bits, w, f): (w, f) for w in range(head_dim // per_word)
             for f in range(per_word)}
    words = {where[UNIT_CHANNELS * unit + i][0] for i in range(UNIT_CHANNELS)}
    if len(words) != 1:
        raise AssertionError(f"unit {unit} spans words {sorted(words)} at int{bits}")
    return words.pop(), tuple(field_lanes(bits, where[UNIT_CHANNELS * unit + i][1])
                              for i in range(UNIT_CHANNELS))


class QuantDequantSm100(QuantDequantBase):
    """Constructed from a `QuantKVLayout` and the out_indptr dtype; raises outside the cache
    writer's envelope (`_append_core.envelope_refusal`), whose layouts are the ones it reads."""

    ARCH = "sm100"

    def __init__(self, layout, out_indptr_64):
        super().__init__(layout, out_indptr_64)
        D = self.head_dim
        self.row_bytes = D * self.bits // 8
        self.units = D // UNIT_CHANNELS
        self.tokens_per_pass = THREADS // self.units
        self.passes = PAGE // self.tokens_per_pass
        # The extraction `_unit_values` hard-codes, checked against the pool's placement for
        # every unit: at INT4 lane word k of word u holds channels 8u + 4k + byte; at INT2 lane
        # word k of the unit's word holds channel 8u + 2 * byte + k // 2 for parity k % 2.
        for u in range(self.units):
            word, lanes = unit_fields(self.bits, D, u)
            if self.bits == 4:
                want = (u, tuple((i // 4, i % 4) for i in range(UNIT_CHANNELS)))
            else:
                want = (u // 2, tuple((2 * (i % 2) + u % 2, i // 2)
                                      for i in range(UNIT_CHANNELS)))
            if (word, lanes) != want:
                raise AssertionError(f"int{self.bits} unit {u}: the pool holds it at "
                                     f"{(word, lanes)}, the extraction reads {want}")

    @cute.jit
    def _unit_values(self, word, unit):
        """The f32 codes of the unit's eight channels, in channel order, from its code word."""
        out = []
        if cutlass.const_expr(self.bits == 4):
            lanes = (byte_lanes(word, 4, 0), byte_lanes(word, 4, 1))
            for i in cutlass.range_constexpr(UNIT_CHANNELS):
                out.append(code_value(lanes[i // 4], i % 4))
        else:
            # INT2: the unit's fields have the parity of the unit; shifting by it puts them at
            # the even fields, whose lane words 0 and 2 hold channels 2 * byte and 2 * byte + 1.
            half = word >> (Int32(2) * (unit % Int32(2)))
            lanes = (byte_lanes(half, 2, 0), byte_lanes(half, 2, 2))
            for i in cutlass.range_constexpr(UNIT_CHANNELS):
                out.append(code_value(lanes[i % 2], i // 2))
        return out

    @cute.jit
    def _load_pass(self, mKq, mVq, mScale, page, head, token, unit):
        """Issue the loads of one unit of K and V: the two code words and the scale words."""
        word = unit // Int32(2) if self.bits == 2 else unit
        row = code_row_index(PAGE, self.num_kv_heads, page, head, token, wide=Int64)
        offset = row * Int64(self.row_bytes) + Int64(4) * Int64(word)
        kw = ld_nc_b32(mKq.iterator.toint() + offset)
        vw = ld_nc_b32(mVq.iterator.toint() + offset)
        group = unit // Int32(self.units // 2) if self.k_half else 0
        scale = self._token_record(mScale, page, head, token, group)
        return kw, vw, scale

    @cute.jit
    def _store_unit(self, mOut, word, s, z, unit, out_row, head):
        """Convert one unit and store it: eight fma.rn, four bf16 pairs, one 16-byte store."""
        q = self._unit_values(word, unit)
        pairs = []
        for i in cutlass.range_constexpr(UNIT_CHANNELS // 2):
            pairs.append(bf16x2(fma_rn(s, q[2 * i], z), fma_rn(s, q[2 * i + 1], z)))
        st_b32(mOut.iterator.toint()
               + (out_row * Int64(mOut.layout.stride[0]) + Int64(head) * Int64(self.head_dim)
                  + Int64(UNIT_CHANNELS) * Int64(unit)) * Int64(2), pairs)

    @cute.kernel
    def kernel(self, mKq, mVq, mScale, mKo, mVo, mPageIndptr, mPageIndices, mSeqLens,
               mOutIndptr):
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        p = cta // Int32(self.num_kv_heads)
        head = cta - p * Int32(self.num_kv_heads)
        num_pages = Int32(mKq.shape[0])
        if p < Int32(mPageIndices.shape[0]):
            unit = tidx % Int32(self.units)
            sub = tidx // Int32(self.units)
            page, safe = self._page_of(mPageIndices, num_pages, p)
            # Every pass's loads, in flight before the request is known (module docstring).
            loads = []
            for i in cutlass.range_constexpr(self.passes):
                loads.append(self._load_pass(mKq, mVq, mScale, safe, head,
                                             Int32(i * self.tokens_per_pass) + sub, unit))
            active, n, row0 = self._locate(mPageIndptr, mSeqLens, mOutIndptr, mKo, mVo, p, lane)
            if active != 0:
                self._require_page(page, num_pages)
                for i in cutlass.range_constexpr(self.passes):
                    token = Int32(i * self.tokens_per_pass) + sub
                    if token < n:
                        kw, vw, scale = loads[i]
                        (sk, zk), (sv, zv) = self._record_pairs(scale)
                        self._store_unit(mKo, kw, sk, zk, unit, row0 + Int64(token), head)
                        self._store_unit(mVo, vw, sv, zv, unit, row0 + Int64(token), head)
        else:
            self._validate(mPageIndptr, mPageIndices, mSeqLens, cta, tidx, THREADS)

    @cute.jit
    def __call__(self, mKq, mVq, mScale, mKo, mVo, mPageIndptr, mPageIndices, mSeqLens,
                 mOutIndptr, num_ctas: Int32, stream):
        self.kernel(mKq, mVq, mScale, mKo, mVo, mPageIndptr, mPageIndices, mSeqLens,
                    mOutIndptr).launch(grid=[num_ctas, 1, 1], block=[THREADS, 1, 1],
                                       stream=stream)


@functools.cache
def _compile(key, out_indptr_64):
    from types import SimpleNamespace
    layout = SimpleNamespace(**dict(key))
    kern = QuantDequantSm100(layout, out_indptr_64)
    num_pages = cute.sym_int()
    pools = [pool_fake((num_pages, kern.num_kv_heads, PAGE, kern.row_bytes)) for _ in range(2)]
    return compile_dequant(kern, __file__, layout, (*pools, num_pages))


def compile_quant_dequant(layout, out_indptr_64):
    """The compiled Blackwell dequantize for `layout` and the out_indptr dtype (int64 when
    True), memoized and taken from the persistent kernel cache when an identical trace exists."""
    return _compile(layout_key(layout), bool(out_indptr_64))


#: Launches a compiled dequantize with this kernel's thread count, which sizes its checking CTAs
#: (`_dequant_core`, GRID).
launch_quant_dequant = functools.partial(launch_dequant, threads=THREADS)
