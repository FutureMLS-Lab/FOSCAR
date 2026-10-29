# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""The shared-memory bank model the CPU tests cost lane maps with.

Shared memory has 32 banks of 4 bytes and serves 128 bytes per wavefront. `wavefronts` turns
the byte addresses of one warp-wide shared-memory instruction into the (actual, ideal)
wavefronts per execution that ncu reports for it as `L1 Wavefronts Shared` and
`L1 Wavefronts Shared Ideal`.

THE SERVICING RULE. It was measured on 2026-09-29 by a lane-pattern microbenchmark on B200
(ssr-b200, sm_100a) and H100 (research-secure-26, sm_90a): 21 lane patterns, each issued by one
warp as one `ld.shared` and one `st.shared` at its own PC, 100 times, under ncu SourceCounters.
Both GPUs gave identical counts for all 42 instructions; the ncu reports are kept on those nodes
under `remote_workspace/bankconf_20260929/`. The warp is served in groups of lanes, formed by
access width and by lane index:

- A 16-byte access, load or store, is served one 8-lane quarter warp at a time. A load also
  serves a 16-lane half warp as one wavefront when that half's lanes touch at most 128 distinct
  bytes on distinct banks, which covers a broadcast. A store never merges.
- An 8-byte access is served one 16-lane half warp at a time. A load in which every active lane
  reads the same 8 bytes is one wavefront.
- An access of 4 bytes or less is served as the whole warp.

A group costs the largest number of distinct 4-byte words that fall in one bank, because lanes
on the same word are one broadcast. A group with no active lane costs nothing.

THE IDEAL is ncu's own count, and the formula below is a fit to it, not a servicing rule:

- a store: the bytes its active lanes write, over 128;
- a load of 8 bytes or less: the bytes of the distinct words the whole warp touches, over 128,
  and 1 for the 8-byte broadcast;
- a 16-byte load: the active lanes in groups of 16 in lane order; a group whose words sit on
  distinct banks within 128 bytes is 1, and otherwise each run of 8 active lanes in it costs
  the bytes of its distinct words over 128.

The measured patterns do not separate whether a 16-byte quarter is the 8 lanes of that index
or the next 8 active lanes, and they do not include a 16-byte load whose half-warp merge spans
two 128-byte rows. The microbenchmark measured 4-, 8- and 16-byte accesses only, so a 1- or
2-byte access is costed by the 4-byte rule. None of these three questions changes a result of
the lane maps the tests cost with this model.

This rule replaces the two that the tests carried from 2026-09-24. "16 active lanes per group"
served 8- and 16-byte accesses in groups of 16 active lanes. The microbenchmark falsified it:
it predicts 4/4 for the operand deposit lane map at start row 0, where both GPUs measure 8/4
for the load and for the store. `test_smem_bank_model_cpu.py` pins the measured patterns that
this model must reproduce, the conflicted ones included.
"""
from __future__ import annotations

import collections

BANKS, BANK_BYTES, WAVEFRONT_BYTES, WARP = 32, 4, 128, 32


def _words(address, width):
    """The 4-byte words that one lane's access touches."""
    return range(address // BANK_BYTES, (address + width - 1) // BANK_BYTES + 1)


def _group(addresses, width):
    """(largest number of distinct words in one bank, bytes of the distinct words) of a group."""
    words = {w for a in addresses for w in _words(a, width)}
    if not words:
        return 0, 0
    return max(collections.Counter(w % BANKS for w in words).values()), BANK_BYTES * len(words)


def _per_128(nbytes):
    return -(-nbytes // WAVEFRONT_BYTES)


def wavefronts(accesses, width, op):
    """(actual, ideal) wavefronts of one warp-wide shared-memory instruction.

    `accesses` is (lane, byte address) for each active lane, `width` the bytes each lane
    accesses and `op` either "ld" or "st". Lanes and addresses are integers, each lane appears
    once, and every address is aligned to the width, as the hardware requires."""
    assert op in ("ld", "st"), op
    assert width in (1, 2, 4, 8, 16), width
    lanes = dict(accesses)
    assert len(lanes) == len(accesses), "a lane appears twice"
    assert all(isinstance(lane, int) and 0 <= lane < WARP for lane in lanes), sorted(lanes)
    assert all(isinstance(a, int) and a >= 0 and a % width == 0 for a in lanes.values()), \
        "an address that is not a non-negative integer aligned to the width"
    active = sorted(lanes)

    def by_index(first, n):
        return [lanes[lane] for lane in range(first, first + n) if lane in lanes]

    if op == "st":
        ideal = _per_128(width * len(active))
    if width == 16:
        actual = 0
        for half in (0, 16):
            peak, nbytes = _group(by_index(half, 16), 16)
            if op == "ld" and peak == 1 and nbytes <= WAVEFRONT_BYTES:
                actual += 1
            else:
                actual += sum(_group(by_index(q, 8), 16)[0] for q in (half, half + 8))
        if op == "ld":
            ideal = 0
            for i in range(0, len(active), 16):
                group = [lanes[lane] for lane in active[i:i + 16]]
                peak, nbytes = _group(group, 16)
                if peak == 1 and nbytes <= WAVEFRONT_BYTES:
                    ideal += 1
                else:
                    ideal += sum(_per_128(_group(group[j:j + 8], 16)[1])
                                 for j in range(0, len(group), 8))
        return actual, ideal
    if width == 8:
        if op == "ld" and len(set(lanes.values())) == 1:
            return 1, 1
        actual = sum(_group(by_index(half, 16), 8)[0] for half in (0, 16))
    else:
        actual = _group(lanes.values(), width)[0]
    if op == "ld":
        ideal = _per_128(_group(lanes.values(), width)[1])
    return actual, ideal
