# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""The graph-on planner: what a CUDA-graph bucket launches, and its work list per step.

Graph mode (`PagedDecodeWrapper(..., use_cuda_graph=True)`) runs beside eager mode, which keeps
its own planners and kernels unchanged. A captured graph freezes every launch argument, so a
split count taken from the grid, as eager's kernels take it, would freeze too. Graph mode instead
plans on the host once per step, as FlashInfer does under CUDA graphs: the host writes a work
list into pinned memory and one asynchronous copy moves it into a fixed device buffer, which the
captured decode reads. This module is that planner, for both GPUs, and it imports neither torch
nor the DSL, so the CPU tests load it from its file.

The planner follows FlashInfer's tensor-core decode plan, `PrefillPlan` with
`enable_cuda_graph` and its `PrefillBinarySearchKVChunkSize`. It does not consult eager's
planners, so a graph step can split a request more or less finely than eager would.

ONCE PER BUCKET (`Bucket`). A bucket is one batch size B the engine captures, with the engine's
largest context `max_kv_len`. Its wave W is the split build's resident CTAs per SM times the SM
count: 2 x 132 = 264 on H100 (FlashInfer's tensor-core path also takes two blocks per SM) and
1 x 148 = 148 on B200. The bucket is SPLIT-CAPABLE iff B * HKV < W (FlashInfer splits nowhere
once `batch_size * num_kv_heads >= max_grid_size`): it then launches the graph split build, a
fixed grid of W CTAs that read the work list, and its merge, at every step. Otherwise it
launches exactly eager's unsplit build for B, with no work list and no merge: FlashInfer would
copy each request through its merge instead, which is the same work and a launch more.
`s_tight` is the most chunks one request can get, which sizes the partial buffers:
min(floor(W / HKV) - (B - 1), ceil(T_cap / c_min)), with T_cap the tile count of `max_kv_len`.

ONCE PER STEP (`chunks`, `write`), from the host lengths L_b >= 1, each T_b = ceil(L_b / t)
tiles of t tokens: the chunk c, in tiles, is the smallest of at least c_min (FlashInfer's
128-token minimum chunk, rounded up to one tile) with sum_b ceil(T_b / c) <= floor(W / HKV)
(FlashInfer's `max_batch_size_if_split`), found by a binary search, and request b gets
k_b = ceil(T_b / c) chunks. A bucket that can split always splits this way, also at a step
whose every k_b is 1 (FlashInfer's `split_kv = enable_cuda_graph || ...`), and its merge always
runs. Each architecture partitions a request's T_b tiles among its k_b pieces its own way:
Hopper gives piece j the contiguous range [j * ceil(T/k), (j + 1) * ceil(T/k)), Blackwell the
strided tiles j, j + k, j + 2k, ...; under either no piece is empty, because ceil(T/k) <= c and
so the first k - 1 contiguous ranges hold at most (k - 1) * c < T tiles, and k <= T.

THE WORK LIST (int32, `words(W, B)` words, one per device buffer). Slot j < W is four words,
16-byte aligned: (item = b * HKV + h, split, k_b, L_b), in request-major, head, split-fastest
order (eager's grid order, so an item's pieces start together); every slot past the last piece
holds item = -1, so a CTA needs one 16-byte load and no count. Then k_b for b < B, at word 4W,
which the merges read. The host writes all W slots every step.
"""
import numpy as np

#: FlashInfer's smallest chunk, `min_kv_chunk_size = max(128 / page_size, 1)` pages.
MIN_CHUNK_TOKENS = 128
#: Words per work-list slot: (item, split, k_b, L_b), one 16-byte load.
ENTRY_WORDS = 4
#: The item of a slot no CTA works on.
IDLE_ITEM = -1
#: Resident CTAs per SM of each architecture's split build. Hopper's build is compiled for two
#: (`min_blocks_per_sm`, the 32/104 setmaxnreg split); Blackwell's shared-memory plan holds one.
SPLIT_CTAS_PER_SM = {"sm90": 2, "sm100": 1}


def min_chunk_tiles(tile):
    """The smallest chunk, in tiles of `tile` tokens: 128 tokens, and at least one tile."""
    return max(-(-MIN_CHUNK_TOKENS // tile), 1)


class Bucket:
    """One captured batch size's planning constants, decided once at `plan()`."""

    __slots__ = ("batch", "num_kv_heads", "wave", "tile", "max_kv_len", "split_capable",
                 "s_tight")

    def __init__(self, batch, num_kv_heads, *, wave, tile, max_kv_len):
        self.batch, self.num_kv_heads = int(batch), int(num_kv_heads)
        self.wave, self.tile = int(wave), int(tile)
        self.max_kv_len = int(max_kv_len)
        self.split_capable = self.batch * self.num_kv_heads < self.wave
        #: The most chunks one request can get: every other request takes at least one of the
        #: floor(W / HKV) a step may hand out, and no chunk is below c_min tiles.
        max_tiles = -(-self.max_kv_len // self.tile)
        self.s_tight = (min(self.wave // self.num_kv_heads - (self.batch - 1),
                            -(-max_tiles // min_chunk_tiles(self.tile)))
                        if self.split_capable else 1)

    def chunks(self, lengths):
        """k_b, int32 [B], for host lengths `lengths` (int64 [B], each in [1, max_kv_len]) of a
        split-capable bucket."""
        lengths = np.asarray(lengths, dtype=np.int64)
        tiles = -(-lengths // self.tile)
        # The smallest chunk c whose pieces fit: sum_b ceil(T_b / c) <= cap. That sum lies in
        # [sum T / c, sum T / c + B), so no c below ceil(sum T / cap) fits and, when cap > B,
        # every c from ceil(sum T / (cap - B)) does; the longest request's tile count always
        # fits (one piece each, and B <= cap in a split-capable bucket). A binary search between
        # those bounds takes a few steps.
        cap = self.wave // self.num_kv_heads
        total, longest = int(tiles.sum()), int(tiles.max())
        lo = max(min_chunk_tiles(self.tile), -(-total // cap))
        hi = longest if cap <= self.batch else min(longest, -(-total // (cap - self.batch)))
        hi = max(lo, hi)
        while lo < hi:
            mid = (lo + hi) // 2
            if int((-(-tiles // mid)).sum()) <= cap:
                hi = mid
            else:
                lo = mid + 1
        return (-(-tiles // lo)).astype(np.int32)

    def write(self, out, lengths, k):
        """Write the work list of `lengths` (int64 [B]) chunked `k` (int32 [B]) into `out`, an
        int32 array of `words(wave, batch)` words: every slot and every k_b."""
        hkv, wave = self.num_kv_heads, self.wave
        slots = out[:wave * ENTRY_WORDS].reshape(wave, ENTRY_WORDS)
        per_item = np.repeat(k, hkv)
        pieces = int(per_item.sum())
        if pieces > wave:
            raise ValueError(f"{pieces} pieces do not fit the wave of {wave} CTAs")
        first = np.cumsum(per_item) - per_item
        slots[:pieces, 0] = np.repeat(np.arange(self.batch * hkv, dtype=np.int32), per_item)
        slots[:pieces, 1] = np.arange(pieces, dtype=np.int32) - np.repeat(first, per_item)
        slots[:pieces, 2] = np.repeat(k, hkv * k)
        slots[:pieces, 3] = np.repeat(lengths, hkv * k)
        slots[pieces:] = 0
        slots[pieces:, 0] = IDLE_ITEM
        tail = out[wave * ENTRY_WORDS:]
        tail[:] = 0
        tail[:self.batch] = k

    def write_lengths(self, out, lengths, k):
        """`write` into a work list that `write` last filled with the same `k`: only the lengths
        differ, so only the length word of each piece is rewritten."""
        slots = out[:self.wave * ENTRY_WORDS].reshape(self.wave, ENTRY_WORDS)
        pieces = self.num_kv_heads * k
        slots[:int(pieces.sum()), 3] = np.repeat(lengths, pieces)


def words(wave, batch):
    """The work list's length in int32 words: W slots of four words, then k_b for b < B, padded
    to a multiple of four so that consecutive buffers keep 16-byte alignment."""
    return wave * ENTRY_WORDS + -(-batch // 4) * 4


def k_offset(wave):
    """The word at which k_b starts, after the W slots (16-byte aligned)."""
    return wave * ENTRY_WORDS

