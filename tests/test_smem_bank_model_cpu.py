# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""The suite's shared-memory bank model, `tests/_smem_banks.py`, against measured lane patterns.

Every CPU test that costs a shared-memory lane map uses that one model, so this file pins a
handful of the lane patterns it is fitted to, with their measured wavefronts. The model's
docstring describes the lane-pattern microbenchmark they come from, which runs each pattern as
one warp-wide load and one warp-wide store under ncu on B200 and on H100 and gives the same
counts on both GPUs. The fixtures cover every part of the rule, and their negative controls are
the conflicted 16-byte patterns: the operand deposit lane map at start row 0 and the 32-byte
lane stride, both at 8 wavefronts against an ideal of 4.

Nothing here runs on a GPU, and nothing here reads kernel source.
"""
from __future__ import annotations

import pytest

from ._smem_banks import wavefronts


def _accesses(offsets, active):
    return [(lane, offsets[lane]) for lane in range(32) if active[lane]]


@pytest.mark.parametrize("accesses,width", [
    ([(0.5, 0)], 8),                 # a fractional lane would be served by no group
    ([(0, 0), (0, 16)], 16),         # a lane twice
    ([(32, 0)], 4),                  # a lane outside the warp
    ([(0, 8)], 16),                  # an address not aligned to the width
    ([(0, 16.0)], 16),               # a fractional address
])
def test_the_model_refuses_an_access_no_warp_can_make(accesses, width):
    for op in ("ld", "st"):
        with pytest.raises(AssertionError):
            wavefronts(accesses, width, op)


# ---------------------------------------------------------------------------------------------
# Pinned patterns
# ---------------------------------------------------------------------------------------------

def _sw128(a):
    """The 128-byte swizzle of the Blackwell kernel's operand ring: address bits 4-6 XOR 7-9."""
    return a ^ ((a >> 3) & 0x70)


def _deposit(start_row):
    """The operand deposit lane map: lane l writes the 16-byte unit at byte 32 (l % 2) of
    128-byte row start_row + l // 2 from a 1024-byte-aligned base, through the 128-byte
    swizzle."""
    return lambda lane: _sw128(128 * (start_row + lane // 2) + 32 * (lane % 2))


def _all(lane):
    return True


#: (pattern name in the microbenchmark, bytes per lane, lane -> address, lane -> active,
#: measured load, measured store), each measurement (actual, ideal) wavefronts per execution,
#: the same on both GPUs.
PINNED = [
    # 16 bytes: one 8-lane quarter warp at a time
    ("c16 contiguous", 16, lambda l: 16 * l, _all, (4, 4), (4, 4)),
    # negative control: every quarter is 2-way; 16 lanes at a time would give 4/4
    ("c16 F1 lane map, start row 0", 16, _deposit(0), _all, (8, 4), (8, 4)),
    ("c16 F1 lane map, start row 6", 16, _deposit(6), _all, (4, 4), (4, 4)),
    # negative control: 32-byte lane stride, 2-way in every quarter
    ("c16 32-byte lane stride (F4 map)", 16, lambda l: 32 * l, _all, (8, 4), (8, 4)),
    # a load merges a half warp that fits one wavefront on distinct banks; a store never does
    ("c16 broadcast, one 16-byte unit", 16, lambda l: 0, _all, (2, 2), (4, 4)),
    ("c16 one unit per quarter, 4 distinct bank groups", 16, lambda l: 16 * (l // 8), _all,
     (2, 2), (4, 4)),
    # a partial warp: quarters 0 and 2 alone, each still 2-way
    ("c16 F1 row 0, lanes 0-7 and 16-23 only", 16, _deposit(0), lambda l: l % 16 < 8,
     (4, 2), (4, 2)),
    # 8 bytes: one 16-lane half warp at a time, by lane index and not by active lanes
    ("c8 16-byte lane stride (P-store map)", 8, lambda l: 16 * l, _all, (4, 2), (4, 2)),
    ("c8 lanes 0-7 banks 0-15, lanes 16-23 banks 16-31 of the next row", 8,
     lambda l: 8 * l if l < 16 else 192 + 8 * (l - 16), lambda l: l % 16 < 8, (2, 1), (2, 1)),
    ("c8 broadcast", 8, lambda l: 0, _all, (1, 1), (2, 2)),
    # 4 bytes: the whole warp
    ("c4 halves on disjoint banks of two rows", 4, lambda l: 4 * l if l < 16 else 128 + 4 * l,
     _all, (1, 1), (1, 1)),
    ("c4 8-byte lane stride (F5 map)", 4, lambda l: 8 * l, _all, (2, 1), (2, 1)),
]


def _lanes(address, active):
    return [address(l) for l in range(32)], [int(bool(active(l))) for l in range(32)]


@pytest.mark.parametrize("name,width,address,active,load,store", PINNED, ids=[p[0] for p in PINNED])
def test_the_model_reproduces_the_pinned_patterns(name, width, address, active, load, store):
    offsets, mask = _lanes(address, active)
    assert wavefronts(_accesses(offsets, mask), width, "ld") == load
    assert wavefronts(_accesses(offsets, mask), width, "st") == store

