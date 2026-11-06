# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""The graph-on planner (`graph_plan.py`): bucket classes, per-step chunk counts, the work list.

`graph_plan.py` and the Hopper eager planner import nothing but numpy and math, so they are
loaded from their files and need neither torch nor the DSL. The kernels' tile partitions are
modelled here (`_ranges`): Hopper gives split j the contiguous range [j * ceil(T/k), (j + 1) *
ceil(T/k)), Blackwell the strided tiles j, j + k, ... (`_sm90/int_decode_pingpong.py`, the tile
loops' `split_idx * tiles_per_cta`; `_sm100/int_fold_decode.py`, `t * kv_splits + kv_split_idx`).
"""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
HKV = 8
#: H100 and B200 as measured: SM count, and the split build's wave.
H100_SMS, B200_SMS = 132, 148
H100_WAVE, B200_WAVE = 2 * H100_SMS, B200_SMS


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gp = _load("_graph_plan_under_test", "src/foscar/graph_plan.py")
sch90 = _load("_sm90_scheduler_for_graph_plan", "src/foscar/_sm90/scheduler.py")

CONTIGUOUS, STRIDED = "contiguous", "strided"
#: (arch, kernel tile, the kernel's partition): every tile the graph builds take.
ARCHES = [("sm90", 64, CONTIGUOUS), ("sm100", 256, STRIDED), ("sm100", 128, STRIDED)]
IDS = [f"{a}-t{t}" for a, t, _ in ARCHES]


def _bucket(arch, batch, tile, max_kv_len, hkv=HKV):
    assert gp.SPLIT_CTAS_PER_SM[arch] * (H100_SMS if arch == "sm90" else B200_SMS) == \
        (H100_WAVE if arch == "sm90" else B200_WAVE)
    return gp.Bucket(batch, hkv, wave=H100_WAVE if arch == "sm90" else B200_WAVE, tile=tile,
                     max_kv_len=max_kv_len)


def _ranges(tiles, k, partition):
    if partition == CONTIGUOUS:
        per = -(-tiles // k)
        return [range(j * per, min(tiles, (j + 1) * per)) for j in range(k)]
    return [range(j, tiles, k) for j in range(k)]


def _flashinfer_chunk(kv_tiles, max_batch_size_if_split, min_kv_chunk_size):
    """FlashInfer's `PrefillBinarySearchKVChunkSize`, for one query row per request,
    transcribed with lengths in kernel tiles: the smallest chunk in
    [min_kv_chunk_size, max length] whose pieces sum to at most `max_batch_size_if_split`."""
    low, high = min_kv_chunk_size, max(max(kv_tiles), 1)
    while low < high:
        mid = (low + high) // 2
        if sum(-(-max(t, 1) // mid) for t in kv_tiles) > max_batch_size_if_split:
            low = mid + 1
        else:
            high = mid
    return low


def test_a_bucket_splits_exactly_when_its_items_leave_part_of_the_wave_empty():
    """Split-capable iff B * HKV < W, as FlashInfer splits iff batch * KV heads is below its
    grid: on H100 every bucket up to 32 at 8 KV heads (256 < 264), on B200 up to 18 (144 < 148),
    whatever the context; a bucket that fills the wave takes one split."""
    for arch, last in (("sm90", 32), ("sm100", 18)):
        for batch in (1, 2, 16, 24, last, last + 1, 64):
            for max_kv_len in (1, 1023, 131072):
                b = _bucket(arch, batch, 64 if arch == "sm90" else 256, max_kv_len)
                assert b.split_capable is (batch <= last), (arch, batch, max_kv_len)
                assert b.split_capable or b.s_tight == 1


def test_s_tight_is_the_wave_left_by_the_other_requests_or_the_longest_context_in_min_chunks():
    # H100, 8 KV heads: floor(264 / 8) = 33 pieces a step, of which B - 1 go to the others.
    assert [_bucket("sm90", b, 64, 131072).s_tight for b in (1, 2, 16, 24, 32)] == \
        [33, 32, 18, 10, 2]
    # A short context caps it at its tiles in 128-token chunks: 1023 tokens are 16 tiles of 64,
    # 8 chunks of 2 tiles.
    assert _bucket("sm90", 1, 64, 1023).s_tight == 8
    # B200: floor(148 / 8) = 18; at 256-token tiles the minimum chunk is one tile.
    assert [_bucket("sm100", b, 256, 131072).s_tight for b in (1, 2, 18)] == [18, 17, 1]
    assert _bucket("sm100", 1, 256, 1000).s_tight == 4


def _check_step(b, lengths, partition):
    k = b.chunks(lengths)
    tiles = -(-lengths // b.tile)
    assert k.dtype == np.int32 and k.shape == (b.batch,)
    assert int(k.sum()) <= b.wave // b.num_kv_heads
    assert (k >= 1).all() and (k <= b.s_tight).all(), (k, b.s_tight)
    for t, kb in zip(tiles.tolist(), k.tolist()):
        ranges = _ranges(t, kb, partition)
        assert all(len(r) for r in ranges), (t, kb)            # no empty piece
        assert sorted(x for r in ranges for x in r) == list(range(t))   # each tile once
    return k


@pytest.mark.parametrize("arch,tile,partition", ARCHES, ids=IDS)
def test_ragged_steps_cover_every_tile_once_within_the_wave_and_capacity(arch, tile, partition):
    rng = np.random.default_rng(7)
    for batch in (1, 2, 3, 4, 8, 12, 16, 18, 24, 32):
        for cap in (4096, 32768, 131072):
            b = _bucket(arch, batch, tile, cap)
            if not b.split_capable:
                continue
            for _ in range(40):
                shape = rng.integers(0, 4)
                if shape == 0:                       # anything
                    lengths = rng.integers(1, cap + 1, batch)
                elif shape == 1:                     # one long request among short ones
                    lengths = rng.integers(1, 600, batch)
                    lengths[rng.integers(0, batch)] = cap
                elif shape == 2:                     # tile and 128-token edges
                    lengths = rng.choice([1, 64, 65, 127, 128, 129, 256, 257, 10241, cap], batch)
                else:                                # uniform
                    lengths = np.full(batch, rng.integers(1, cap + 1))
                _check_step(b, np.minimum(lengths, cap).astype(np.int64), partition)


@pytest.mark.parametrize("arch,tile,partition", ARCHES, ids=IDS)
def test_the_chunk_is_flashinfers_and_the_smallest_that_fits(arch, tile, partition):
    """`chunks` against FlashInfer's binary search and against a scan of every chunk from the
    128-token minimum: the smallest c (tiles) with sum_b ceil(T_b / c) <= floor(W / HKV), and
    k_b = ceil(T_b / c)."""
    rng = np.random.default_rng(11)
    for batch in (1, 2, 3, 8, 16, 18, 32):
        for hkv in (1, 2, 8):
            b = _bucket(arch, batch, tile, 65536, hkv=hkv)
            if not b.split_capable:
                continue
            for _ in range(30):
                lengths = rng.integers(1, rng.choice([300, 5000, 65537]), batch).astype(np.int64)
                tiles = -(-lengths // tile)
                cap = b.wave // hkv
                c = next(c for c in range(gp.min_chunk_tiles(tile), int(tiles.max()) + 2)
                         if int((-(-tiles // c)).sum()) <= cap)
                fi = _flashinfer_chunk(tiles.tolist(), cap, max(-(-128 // tile), 1))
                want = (-(-tiles // c)).tolist()
                assert (-(-tiles // fi)).tolist() == want
                assert b.chunks(lengths).tolist() == want, (batch, hkv, lengths.tolist())


@pytest.mark.parametrize("length", [8192, 32768])
@pytest.mark.parametrize("bits", [2, 4])
def test_one_long_request_takes_the_wave_where_eager_splits_16(bits, length):
    """docs/integration.md: at B = 1 on H100 the planner splits an 8K to 32K request 32 ways,
    where eager's planner splits it 16 ways."""
    b = _bucket("sm90", 1, 64, 131072)
    assert b.chunks(np.array([length], dtype=np.int64)).tolist() == [32]
    assert sch90.plan_splits(1, HKV, length, bits=bits, head_dim=128, page_size=64,
                             sm_count=H100_SMS)[0] == 16


def test_s_tight_is_reached_by_one_long_request_beside_short_ones():
    """The capacity bound is tight: one request of s_tight minimum chunks (or at max_kv_len,
    if that is shorter) beside B - 1 one-token requests takes every chunk the others leave, or
    every minimum chunk it holds."""
    for arch, tile, partition in ARCHES:
        # 1025 and 4097 tokens end in a partial minimum chunk, which s_tight counts whole
        for batch, cap in ((1, 131072), (2, 32768), (4, 131072), (16, 131072), (1, 1023),
                           (1, 1025), (2, 4097)):
            b = _bucket(arch, batch, tile, cap)
            if not b.split_capable:
                continue
            lengths = np.ones(batch, dtype=np.int64)
            lengths[0] = min(cap, b.s_tight * gp.min_chunk_tiles(tile) * tile)
            assert _check_step(b, lengths, partition)[0] == b.s_tight, (arch, tile, batch, cap)
            lengths[0] = cap
            assert _check_step(b, lengths, partition)[0] <= b.s_tight


def _decode(b, buf):
    """(entries [W, 4], k [B]) read back from a work list."""
    slots = buf[:gp.k_offset(b.wave)].reshape(b.wave, gp.ENTRY_WORDS)
    return slots, buf[gp.k_offset(b.wave):gp.k_offset(b.wave) + b.batch]


@pytest.mark.parametrize("arch,tile,partition", ARCHES, ids=IDS)
def test_the_work_list_holds_every_piece_in_eager_order_then_idle_slots(arch, tile, partition):
    rng = np.random.default_rng(3)
    for batch in (1, 3, 8, 16):
        b = _bucket(arch, batch, tile, 65536)
        buf = np.full(gp.words(b.wave, batch), 12345, dtype=np.int32)
        assert gp.words(b.wave, batch) % 4 == 0 and gp.k_offset(b.wave) % 4 == 0
        for _ in range(10):
            lengths = rng.integers(1, 65537, batch).astype(np.int64)
            k = b.chunks(lengths)
            b.write(buf, lengths, k)
            slots, kb = _decode(b, buf)
            assert kb.tolist() == k.tolist()
            assert (buf[gp.k_offset(b.wave) + batch:] == 0).all()
            want = [(bb * HKV + h, s, int(k[bb]), int(lengths[bb]))
                    for bb in range(batch) for h in range(HKV) for s in range(int(k[bb]))]
            n = len(want)
            assert [tuple(x) for x in slots[:n].tolist()] == want
            assert (slots[n:, 0] == gp.IDLE_ITEM).all() and (slots[n:, 1:] == 0).all()


def test_more_pieces_than_the_wave_are_refused():
    b = _bucket("sm90", 2, 64, 65536)
    buf = np.zeros(gp.words(b.wave, 2), dtype=np.int32)
    with pytest.raises(ValueError, match="do not fit the wave"):
        b.write(buf, np.array([65536, 65536], dtype=np.int64), np.array([17, 17], np.int32))


def test_rewriting_only_the_lengths_gives_the_work_list_a_full_write_gives():
    rng = np.random.default_rng(5)
    b = _bucket("sm90", 4, 64, 65536)
    first = rng.integers(1, 65537, 4).astype(np.int64)
    k = b.chunks(first)
    buf = np.zeros(gp.words(b.wave, 4), dtype=np.int32)
    b.write(buf, first, k)
    for _ in range(20):
        # another length in each request's tile, short of its last token, where the tile count
        # and so the chunks do not move
        lengths = (first - 1) // 64 * 64 + 1 + rng.integers(0, 63, 4)
        assert (b.chunks(lengths) == k).all()
        want = np.zeros_like(buf)
        b.write(want, lengths, k)
        b.write_lengths(buf, lengths, k)
        assert (buf == want).all()


def test_the_initial_plan_is_one_chunk_per_request_at_length_one():
    b = _bucket("sm90", 4, 64, 32768)
    ones = np.ones(4, dtype=np.int64)
    k = b.chunks(ones)
    assert k.tolist() == [1, 1, 1, 1]
    buf = np.zeros(gp.words(b.wave, 4), dtype=np.int32)
    b.write(buf, ones, k)
    slots, _ = _decode(b, buf)
    assert [tuple(x) for x in slots[:4 * HKV].tolist()] == [(i, 0, 1, 1) for i in range(4 * HKV)]
    assert (slots[4 * HKV:, 0] == gp.IDLE_ITEM).all()
