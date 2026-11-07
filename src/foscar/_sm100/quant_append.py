# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: MIT
"""QuantAppendSm100: the serving cache write for `IntDecodeSm100`, on Blackwell.

One launch per layer quantizes new K and V tokens and writes them into the global-memory pools
the Blackwell decode reads, and keeps `v_page_max` with its count table `v_max_hist`
(docs/SERVING_READINESS.md, sections 3, 4 and 6.2). The page runs, the quantization, the scale
records, the tail clear and the page maximum are `_append_core.QuantAppendBase`'s; this module
adds the Blackwell code pools.

THE CODE POOLS, in the production layout the reader's predicates name
(`IntDecodeSm100.k_pool_is_permuted` and `v_pool_token_rows`; `_sm100/code_layout.py`). V is
stored exactly like K (docs/SM100_ALIGNED_KV.md), so the two pools have one shape and one
within-word order:
    K  uint8 [NP, HKV, 64, D * bits / 8]   one row per token, packing its D channels
    V  uint8 [NP, HKV, 64, D * bits / 8]   the same
A row belongs to one token alone, so a run owns every word of every row it touches, and no
store of one run can meet another run's fields.

THE STORES (step 3 of the core). For each pool the threads take (token row, 16-byte chunk) items
in row-major order, so consecutive threads store consecutive 16 bytes of the pool. An item whose
token is in the run builds its four words from the staged bytes and stores them with one 16-byte
store. Every address widens the page to 64 bits before it is multiplied
(`code_layout.code_row_index`).

`QuantAppendSm100(layout, prequantized=True)` is the writer of `QuantKVCache.append_quantized`,
which lays out codes and scales the engine quantized (the core's THE PREQUANTIZED WRITE); its
stores, page maxima and count moves are these.
"""
import functools

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from .._append_core import (
    PAGE, THREADS, QuantAppendBase, check_clip_indices, compile_writer, given_fakes, in_run,
    input_fakes, launch_writer, layout_key, pool_fake, resident_ctas_per_sm, scale_fake, st_b32)
from .code_layout import code_field, code_row_index
from .fold_runtime import VMAX_BINS


class QuantAppendSm100(QuantAppendBase):
    """Constructed from a `QuantKVLayout` alone; raises outside the version-1 envelope
    (`_append_core.envelope_refusal`)."""

    ARCH = "sm100"
    WRITES_V_PAGE_MAX = True

    def __init__(self, layout, prequantized=False, clip=(-1, -1)):
        super().__init__(layout, prequantized, clip)
        self.row_bytes = self.head_dim * self.bits // 8

    @cute.jit
    def _store_run(self, sQk, sQv, mKq, mVq, page, head, t0, n, tidx):
        self._store_token_rows(sQk, mKq, page, head, t0, n, tidx)
        self._store_token_rows(sQv, mVq, page, head, t0, n, tidx)

    @cute.jit
    def _store_token_rows(self, sQ, mQ, page, head, t0, n, tidx):
        """Store the run's token rows of one pool (K or V) from its staged bytes `sQ`: one row
        per token, owned by that token alone, so every store is a plain 16-byte one."""
        bits = self.bits
        fields = 32 // bits
        chunks = self.row_bytes // 16
        items = PAGE * chunks
        base = mQ.iterator.toint()
        for i in cutlass.range_constexpr((items + THREADS - 1) // THREADS):
            item = tidx + THREADS * i
            token = item // chunks
            chunk = item % chunks
            if item < items and in_run(token, t0, n):
                values = []
                for u in cutlass.range_constexpr(4):
                    value = Int32(0)
                    for f in cutlass.range_constexpr(fields):
                        channel = code_field(bits, 4 * chunk + u, f)
                        value = value | (Int32(sQ[(token, channel)]) << (bits * f))
                    values.append(value)
                row = code_row_index(PAGE, self.num_kv_heads, page, head, token, wide=Int64)
                st_b32(base + row * Int64(self.row_bytes) + Int64(16 * chunk), values)

    @cute.jit
    def __call__(self, mK, mV, mSlot, mKq, mVq, mScale, mVPageMax, mVMaxHist, num_slices: Int32,
                 stream):
        self.kernel(mK, mV, mSlot, mKq, mVq, mScale, mVPageMax, mVMaxHist, num_slices).launch(
            grid=[self.num_kv_heads, num_slices, 1], block=[THREADS, 1, 1], stream=stream,
            min_blocks_per_mp=resident_ctas_per_sm(self.head_dim))

    @cute.jit
    def call_prequantized(self, mKc, mKs, mKz, mVc, mVs, mVz, mSlot, mKq, mVq, mScale,
                          mVPageMax, mVMaxHist, num_slices: Int32, stream):
        """The prequantized write's entry (`prequantized=True`): uint8 codes [T, HKV, D] and f32
        scales and zero points, then the arguments of `__call__`."""
        self.kernel_prequantized(mKc, mKs, mKz, mVc, mVs, mVz, mSlot, mKq, mVq, mScale,
                                 mVPageMax, mVMaxHist, num_slices).launch(
            grid=[self.num_kv_heads, num_slices, 1], block=[THREADS, 1, 1], stream=stream,
            min_blocks_per_mp=resident_ctas_per_sm(self.head_dim))


@functools.cache
def _compile(key, prequantized, clip):
    from types import SimpleNamespace
    layout = SimpleNamespace(**dict(key))
    kern = QuantAppendSm100(layout, prequantized, clip)
    if prequantized:
        inputs, num_pages = given_fakes(kern)
    else:
        k, v, slots, num_pages = input_fakes(kern)
        inputs = (k, v, slots)
    k_pool = pool_fake((num_pages, kern.num_kv_heads, PAGE, kern.row_bytes))
    v_pool = pool_fake((num_pages, kern.num_kv_heads, PAGE, kern.row_bytes))
    page_max = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint32, (num_pages, kern.num_kv_heads), stride_order=(1, 0), assumed_align=4)
    hist = cute.runtime.make_fake_compact_tensor(cutlass.Uint32, (VMAX_BINS,), assumed_align=16)
    return compile_writer(kern, __file__, layout,
                          (*inputs, k_pool, v_pool, scale_fake(kern, num_pages), page_max,
                           hist))


def compile_quant_append(layout, prequantized=False, clip=(-1, -1)):
    """The compiled Blackwell writer for `layout`, the prequantized write when `prequantized`, with
    the (K, V) clip indices `clip` (`kv_cache.clip_index`, -1 for no clip), memoized per layout,
    form and clip and taken from the persistent kernel cache (`kernel_cache.compile_or_load`)
    when an identical trace exists. The indices are checked before the memoized compile, whose
    key would otherwise equate 122.0 with 122."""
    return _compile(layout_key(layout), bool(prequantized),
                    check_clip_indices(clip, int(layout.head_dim)))


def launch_quant_append(compiled, *, inputs, slot_mapping, k_codes, v_codes, scales,
                        v_page_max=None, v_max_hist=None):
    """Write `inputs` at `slot_mapping` (int64 [T]) into the pools and keep `v_page_max` (uint32
    [NP, HKV]) and `v_max_hist` (uint32 [VMAX_BINS]), on torch's current stream: (k, v), bf16
    [T, HKV, D], for the quantizing writer, or (k codes, k scales, k zero points, v codes,
    v scales, v zero points) for the prequantized one. Every layout this writer accepts has
    token-wise V, so both page-maximum tensors are required."""
    if v_page_max is None or v_max_hist is None:
        raise ValueError("the Blackwell writer keeps v_page_max and v_max_hist for every layout "
                         "it writes; pass the cache's tensors")
    launch_writer(compiled, (*inputs, slot_mapping, k_codes, v_codes, scales, v_page_max,
                             v_max_hist))
