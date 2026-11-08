# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""The shared-memory geometry of the Hopper dequantize (`_sm90/quant_dequant.py`), on a CPU.

The kernel copies each (page, KV head) code block into shared memory, each warpgroup thread its
own row with 16-byte stores, and then gathers the eight values of each output unit (8 channels of
one token) with 4-byte loads, consecutive lanes taking consecutive units of a token row. Three
plain-integer functions of `serving_cache` say where everything is, and the kernel traces them:
`code_position` (where a (token, channel) value sits in a block), `dequant_unit_gathers` (the
rows, words and fields of a unit's channels relative to its first) and `dequant_stage_word` (the
swizzled shared-memory word of a row word). This file checks that

  * `code_position` inverts `code_field` for every field of every row word, both tensors, both
    widths and both head dimensions, so the kernel reads the values the packers wrote;
  * the gather structure holds for every token and unit, and in K the two channels of a pair
    share a word 16 bits apart, which is how the kernel converts them;
  * the staged block is a permutation of the global block in which each 16-byte chunk stays one
    aligned 16-byte slot, so a thread stores each chunk with one 16-byte store;
  * every staging store and every gather is at its ideal wavefront count under the suite's bank
    model (`tests/_smem_banks.py`), for any warp of 32 consecutive units in token-row order.

Without the swizzle the gathers are 2-way (K at INT4) to 16-way (V at INT4 and head dimension
256) conflicted; the last test pins that, so the swizzle is shown to be what removes them.

Run: pytest tests/test_sm90_dequant_banks_cpu.py
"""
import importlib.util
import pathlib
import sys
import types

import pytest

from ._smem_banks import wavefronts

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = 64
UNIT = 8
CASES = [(tensor, bits, head_dim) for tensor in ("K", "V") for bits in (2, 4)
         for head_dim in (64, 128, 256)]


def _load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sc(monkeypatch):
    # Importing the package imports the DSL, so serving_cache is loaded on its own.
    for pkg in ("foscar", "foscar._sm90"):
        stub = types.ModuleType(pkg)
        stub.__path__ = []
        monkeypatch.setitem(sys.modules, pkg, stub)
    _load(monkeypatch, "foscar.quant_layout", ROOT / "src/foscar/quant_layout.py")
    return _load(monkeypatch, "foscar._sm90.serving_cache",
                 ROOT / "src/foscar/_sm90/serving_cache.py")


def _row_words(sc, tensor, bits, head_dim):
    per_k, per_v = sc.code_row_bytes(head_dim, PAGE, bits)
    return (per_k if tensor == "K" else per_v) // 4


@pytest.mark.parametrize("tensor,bits,head_dim", CASES)
def test_code_position_inverts_code_field(sc, tensor, bits, head_dim):
    seen = set()
    for row in range(128):
        for word in range(_row_words(sc, tensor, bits, head_dim)):
            for field in range(32 // bits):
                token, channel = sc.code_field(tensor, bits, row, word, field)
                assert sc.code_position(tensor, bits, token, channel) == (row, word, field)
                seen.add((token, channel))
    assert seen == {(t, c) for t in range(PAGE) for c in range(head_dim)}


@pytest.mark.parametrize("tensor,bits,head_dim", CASES)
def test_unit_gathers(sc, tensor, bits, head_dim):
    gathers = sc.dequant_unit_gathers(tensor, bits, head_dim)
    assert len(gathers) == UNIT and gathers[0] == (0, 0, 0)
    if tensor == "K":
        for i in range(0, UNIT, 2):
            dr, dw, df = gathers[i]
            assert gathers[i + 1] == (dr, dw, df + 16 // bits)


@pytest.mark.parametrize("tensor,bits,head_dim", CASES)
def test_stage_is_a_slot_permutation(sc, tensor, bits, head_dim):
    words = max(_row_words(sc, tensor, bits, head_dim), 4)
    placed = {}
    for row in range(128):
        for chunk in range(words // 4):
            slots = {sc.dequant_stage_word(tensor, row, 4 * chunk + i) // 4 for i in range(4)}
            assert len(slots) == 1
            for i in range(4):
                placed[sc.dequant_stage_word(tensor, row, 4 * chunk + i)] = (row, 4 * chunk + i)
    assert sorted(placed) == list(range(128 * words))


def _stage_stores(sc, tensor, bits, head_dim, stage_word):
    """Every staging store, as (lane, byte address) per warp-wide 16-byte store: thread r of the
    warpgroup stores chunk c of row r at the slot of its first word."""
    for chunk in range(max(_row_words(sc, tensor, bits, head_dim) // 4, 1)):
        for warp in range(4):
            yield [(lane, 16 * (stage_word(tensor, 32 * warp + lane, 4 * chunk) // 4))
                   for lane in range(32)]


def _gathers(sc, tensor, bits, head_dim, stage_word):
    """Every gather, as (lane, byte address) per warp-wide 4-byte load: units numbered in token-row
    order, a warp taking 32 consecutive ones, and gather g of each unit."""
    units = head_dim // UNIT
    gathers = sc.dequant_unit_gathers(tensor, bits, head_dim)
    for first in range(0, PAGE * units, 32):
        for g in range(0, UNIT, 2 if tensor == "K" else 1):
            dr, dw, _ = gathers[g]
            accesses = []
            for lane in range(32):
                token, unit = divmod(first + lane, units)
                row, word, _ = sc.code_position(tensor, bits, token, UNIT * unit)
                accesses.append((lane, 4 * stage_word(tensor, row + dr, word + dw)))
            yield accesses


@pytest.mark.parametrize("tensor,bits,head_dim", CASES)
def test_stores_and_gathers_at_ideal(sc, tensor, bits, head_dim):
    for accesses in _stage_stores(sc, tensor, bits, head_dim, sc.dequant_stage_word):
        actual, ideal = wavefronts(accesses, 16, "st")
        assert actual == ideal, (accesses, actual, ideal)
    for accesses in _gathers(sc, tensor, bits, head_dim, sc.dequant_stage_word):
        actual, ideal = wavefronts(accesses, 4, "ld")
        assert actual == ideal, (accesses, actual, ideal)


def test_unswizzled_gathers_conflict(sc):
    """The chunk16 placement itself, unswizzled, conflicts on the gathers this kernel issues, so
    the swizzle is what makes the previous test pass."""
    def plain(tensor, thread, word):
        return 4 * thread + 512 * (word // 4) + word % 4

    worst = {}
    for tensor, bits, head_dim in CASES:
        worst[(tensor, bits, head_dim)] = max(
            wavefronts(a, 4, "ld")[0]
            for a in _gathers(sc, tensor, bits, head_dim, plain))
    assert worst[("K", 4, 128)] == 2 and worst[("V", 4, 128)] == 8
    assert worst[("V", 4, 256)] == 16 and worst[("V", 2, 128)] == 4
