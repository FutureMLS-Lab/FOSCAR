# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: MIT
"""QuantAppendSm90: the serving cache write for `IntPingPongDecodeSm90`, on Hopper.

One launch per layer quantizes new K and V tokens and writes them into the global-memory pools
the Hopper decode reads. The page runs, the
quantization, the scale records and the tail clear are `_append_core.QuantAppendBase`'s; this
module adds the Hopper code pools.

THE CODE POOLS. K and V are uint8 [NP, HKV, per/16, 128, 16] in the reader's code-pool order,
`IntPingPongDecodeSm90.code_pool_order` ("chunk16"), which the layout carries: chunk c of
warpgroup thread row r holds row bytes [16c, 16c + 16). A row holds plane words of RS wgmma
A-fragments, V transposed, and `serving_cache.code_field` names the (token, channel) of every
field of every row word. A K word holds two tokens, t and t XOR 8, and a V word four tokens at
int4 or eight at int2, so a single token shares every word it touches with other tokens.

THE STORES (step 3 of the core). Thread r owns row r of both pools and visits its row one
16-byte chunk at a time. For each of a chunk's four words it marks the fields whose token lies
in the run and, when there is any, builds their values from the staged bytes; the core's
`_store_word_group` then issues one 16-byte store when the run owns the whole chunk, and
otherwise a plain store per owned word and red.and plus red.or per shared one. Every address
comes from `serving_cache.code_word_offset`, which widens the page to 64 bits before it is
multiplied, so the pool order enters in exactly one place. A whole-page run (a prefill page)
therefore issues only plain 16-byte stores, and a warp's stores of one chunk cover 512
contiguous bytes.

`QuantAppendSm90(layout, prequantized=True)` is the writer of `QuantKVCache.append_quantized`,
which lays out codes and scales the engine quantized (the core's THE PREQUANTIZED WRITE); its
stores are these.
"""
import functools

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from .._append_core import (
    PAGE, THREADS, QuantAppendBase, check_clip_indices, compile_writer, given_fakes, in_run,
    input_fakes, launch_writer, layout_key, pool_fake, resident_ctas_per_sm, s32, scale_fake,
    select)
from .serving_cache import (
    code_block_shape, code_field, code_pool_shape, code_row_bytes, code_word_offset,
    require_code_pool)


@functools.cache
def _token_deltas(tensor, bits, word):
    """The token of each field of row word `word`, less the token of its field 0: the same for
    every thread row (checked here, at trace time), which lets `_store_rows` test a whole word
    against a run with two compares."""
    deltas = [code_field(tensor, bits, 0, word, f)[0] - code_field(tensor, bits, 0, word, 0)[0]
              for f in range(32 // bits)]
    for row in range(128):
        base = code_field(tensor, bits, row, word, 0)[0]
        if [code_field(tensor, bits, row, word, f)[0] - base for f in range(32 // bits)] != deltas:
            raise AssertionError(f"the field tokens of {tensor} word {word} depend on the row")
    return tuple(deltas)


class QuantAppendSm90(QuantAppendBase):
    """Constructed from a `QuantKVLayout` alone; raises outside the version-1 envelope
    (`_append_core.envelope_refusal`)."""

    ARCH = "sm90"

    def __init__(self, layout, prequantized=False, clip=(-1, -1)):
        super().__init__(layout, prequantized, clip)
        self.code_pool_order = layout.code_pool_order
        self.k_row_bytes, self.v_row_bytes = code_row_bytes(self.head_dim, PAGE, self.bits)
        code_block_shape(self.code_pool_order, self.k_row_bytes)    # refuses an unknown order

    @cute.jit
    def _store_rows(self, tensor: cutlass.Constexpr, sQ, mPool, per: cutlass.Constexpr,
                    page, head, t0, n, row):
        """Build and store every word of thread row `row` of one code pool that the run
        touches, one 16-byte chunk at a time."""
        bits = self.bits
        fields = 32 // bits
        pool = mPool.iterator.toint()
        chunk_words = min(per, 16) // 4
        for chunk in cutlass.range_constexpr(max(per // 16, 1)):
            masks = []
            values = []
            for u in cutlass.range_constexpr(chunk_words):
                word = 4 * chunk + u
                # The word's tokens are its field 0's plus constants, so one pair of compares
                # tells whether the run touches it at all; most words of a short run it does not.
                deltas = _token_deltas(tensor, bits, word)
                first, _ = code_field(tensor, bits, row, word, 0)
                mask = Int32(0)
                value = Int32(0)
                if first + min(deltas) >= t0 and first + max(deltas) < t0 + n:
                    # The run owns every field: a whole page's words, all of them.
                    mask = Int32(-1)
                    for f in cutlass.range_constexpr(fields):
                        token, channel = code_field(tensor, bits, row, word, f)
                        value = value | (Int32(sQ[(token, channel)]) << (bits * f))
                else:
                    if first + max(deltas) >= t0 and first + min(deltas) < t0 + n:
                        for f in cutlass.range_constexpr(fields):
                            token, channel = code_field(tensor, bits, row, word, f)
                            owned = in_run(token, t0, n)
                            mask = mask | select(owned, s32(self.qmax << (bits * f)), 0)
                            value = value | select(owned, Int32(sQ[(token, channel)]) << (bits * f),
                                                   0)
                masks.append(mask)
                values.append(value)
            offset = code_word_offset(self.code_pool_order, per, self.num_kv_heads, page, head,
                                      row, 4 * chunk, wide=Int64)
            self._store_word_group(pool + offset * Int64(4), masks, values)

    @cute.jit
    def _store_run(self, sQk, sQv, mKq, mVq, page, head, t0, n, tidx):
        self._store_rows("K", sQk, mKq, self.k_row_bytes, page, head, t0, n, tidx)
        self._store_rows("V", sQv, mVq, self.v_row_bytes, page, head, t0, n, tidx)

    @cute.jit
    def __call__(self, mK, mV, mSlot, mKq, mVq, mScale, num_slices: Int32, stream):
        # The reader's order and C-contiguity, refused at trace time, because the stores address
        # each (page, head) block as one dense run in that order.
        require_code_pool("k_codes", mKq.shape, self.code_pool_order, self.k_row_bytes,
                          stride=mKq.stride)
        require_code_pool("v_codes", mVq.shape, self.code_pool_order, self.v_row_bytes,
                          stride=mVq.stride)
        self.kernel(mK, mV, mSlot, mKq, mVq, mScale, None, None, num_slices).launch(
            grid=[self.num_kv_heads, num_slices, 1], block=[THREADS, 1, 1], stream=stream,
            min_blocks_per_mp=resident_ctas_per_sm(self.head_dim))

    @cute.jit
    def call_prequantized(self, mKc, mKs, mKz, mVc, mVs, mVz, mSlot, mKq, mVq, mScale,
                          num_slices: Int32, stream):
        """The prequantized write's entry (`prequantized=True`): uint8 codes [T, HKV, D] and f32
        scales and zero points, then the arguments of `__call__`."""
        require_code_pool("k_codes", mKq.shape, self.code_pool_order, self.k_row_bytes,
                          stride=mKq.stride)
        require_code_pool("v_codes", mVq.shape, self.code_pool_order, self.v_row_bytes,
                          stride=mVq.stride)
        self.kernel_prequantized(mKc, mKs, mKz, mVc, mVs, mVz, mSlot, mKq, mVq, mScale, None,
                                 None, num_slices).launch(
            grid=[self.num_kv_heads, num_slices, 1], block=[THREADS, 1, 1], stream=stream,
            min_blocks_per_mp=resident_ctas_per_sm(self.head_dim))


@functools.cache
def _compile(key, prequantized, clip):
    from types import SimpleNamespace
    layout = SimpleNamespace(**dict(key))
    kern = QuantAppendSm90(layout, prequantized, clip)
    if prequantized:
        inputs, num_pages = given_fakes(kern)
    else:
        k, v, slots, num_pages = input_fakes(kern)
        inputs = (k, v, slots)
    pools = [pool_fake(code_pool_shape(kern.code_pool_order, num_pages, kern.num_kv_heads, per))
             for per in (kern.k_row_bytes, kern.v_row_bytes)]
    return compile_writer(kern, __file__, layout,
                          (*inputs, pools[0], pools[1], scale_fake(kern, num_pages)))


def compile_quant_append(layout, prequantized=False, clip=(-1, -1)):
    """The compiled Hopper writer for `layout`, the prequantized write when `prequantized`, with
    the (K, V) clip indices `clip` (`kv_cache.clip_index`, -1 for no clip), memoized per layout,
    form and clip and taken from the persistent kernel cache (`kernel_cache.compile_or_load`)
    when an identical trace exists. The indices are checked before the memoized compile, whose
    key would otherwise equate 122.0 with 122."""
    return _compile(layout_key(layout), bool(prequantized),
                    check_clip_indices(clip, int(layout.head_dim)))


def launch_quant_append(compiled, *, inputs, slot_mapping, k_codes, v_codes, scales,
                        v_page_max=None, v_max_hist=None):
    """Write `inputs` at `slot_mapping` (int64 [T]) into the pools, on torch's current stream:
    (k, v), bf16 [T, HKV, D], for the quantizing writer, or (k codes, k scales, k zero points,
    v codes, v scales, v zero points) for the prequantized one. Hopper keeps no page maximum, so
    `v_page_max` and `v_max_hist` must be None."""
    if v_page_max is not None or v_max_hist is not None:
        raise ValueError("the Hopper writer keeps no v_page_max or v_max_hist; pass None")
    launch_writer(compiled, (*inputs, slot_mapping, k_codes, v_codes, scales))
