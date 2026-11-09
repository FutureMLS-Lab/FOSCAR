# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: MIT
"""Bank arithmetic of the lc strip `sLcFactors` of `IntPingPongDecodeSm90`, on a CPU.

Once per tile, after the tile's PV wgmmas (`pv_tile_restart`), the strip is copied from shared
memory into the A registers of the lc wgmma: by lanes 0-7 of warp 0 on the whole-D body, the
threads that hold the rows of l and c, and by every thread on the grouped body (head dimension
256), thread t reading thread t % 8's bytes. The copy by warp 0's lanes 0-7 used to take 16
scalar loads per tile, each at 2 wavefronts against an ideal of 1, because lanes 0-3 and 4-7 read
rows 32 words apart. Its physical order is now `_lc_factor_layout`, in which each 16-token step is
one 128-byte row in the registers' order, so the copy is 4 16-byte loads per warp per tile.

This file proves by lane arithmetic that every load of the copy is at its ideal, that the other
access to the buffer (the lc writer) stays at its ideal, and that the reader and the writer
still agree on every value. It also re-derives, for the earlier token-fastest layout, the
per-instruction counts ncu measured there (refill LDS at 2.00/1.00 on 8 threads and the lc
writer STS.U16 at 1.00/1.00 on 8 threads), so the model is anchored before it is used.

THE MODEL is the suite's one shared-memory bank model, `tests/_smem_banks.py`: the servicing
rule that a lane-pattern microbenchmark measured on B200 and H100, whose measured
patterns `test_smem_bank_model_cpu.py` pins. A 16-byte load is served one 8-lane quarter warp at a time, merged into one
wavefront per 16-lane half warp when that half touches at most 128 bytes on distinct banks, and
a 4-byte access is served as the whole warp. Each group costs the largest number of distinct
4-byte words in one bank. The microbenchmark measured no 2-byte access, so the lc writer's
STS.U16 is costed by the 4-byte rule, and ncu's 1.00/1.00 for it is the measurement
that anchors that choice here.

A buffer's base only rotates the banks every lane sees, so each check runs at every 16-byte
aligned base offset modulo 128; both buffers are allocated at 16-byte alignment.

The layouts are read out of the kernel SOURCE, because importing the module imports the DSL:
the two module-level helpers are plain Python and are executed here on their own. The last
tests pin that the kernel's allocation, refill and writer actually go through them.
"""
from __future__ import annotations

import ast
import math
import pathlib

import pytest

from ._smem_banks import wavefronts

ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNEL = ROOT / "src/foscar/_sm90/int_decode_pingpong.py"
HELPERS = ("_lc_factor_layout", "_lc_factor_words")

PAGE = 64            # the one page size both kernels accept (kernel_contract.validate_problem)
PLANES = 4           # rcp hi, zr hi, rcp lo, zr lo
WARPGROUPS = 2       # the two consumer warpgroups
BASES = range(0, 128, 16)


@pytest.fixture(scope="module")
def kernel():
    """The kernel's two layout helpers, executed from its source without the DSL."""
    tree = ast.parse(KERNEL.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS]
    assert sorted(n.name for n in nodes) == sorted(HELPERS)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(KERNEL), "exec"), namespace)
    return namespace


# ---------------------------------------------------------------------------------------------
# CuTe coordinates, the wgmma fragments and the bank model
# ---------------------------------------------------------------------------------------------

def _size(shape):
    return math.prod(_size(s) for s in shape) if isinstance(shape, tuple) else shape


def _crd2idx(coord, shape, stride):
    """CuTe's coordinate-to-offset map. A flat integer coordinate into a hierarchical mode is
    decomposed colexicographically (first sub-mode fastest), as CuTe does for the writer's
    token subscript."""
    if not isinstance(shape, tuple):
        assert 0 <= coord < shape, (coord, shape)
        return coord * stride
    if isinstance(coord, tuple):
        assert len(coord) == len(shape)
        return sum(_crd2idx(c, s, d) for c, s, d in zip(coord, shape, stride))
    assert 0 <= coord < _size(shape), (coord, shape)
    offset = 0
    for s, d in zip(shape, stride):
        offset += _crd2idx(coord % _size(s), s, d)
        coord //= _size(s)
    return offset


def _c_fragment(warp, lane, j):
    """(row, column) of accumulator value j of `lane` in `warp` of a warpgroup, for the
    m64n8 wgmma C fragment (PTX ISA, wgmma .m64nNk16 D fragments): rows 16w + lane//4 and
    8 below it, columns 2(lane%4) and the next. The kernel's QK and PV coordinate tensors
    (`tQkCoord`, `tPvCoord`) are this map; its merge relies on it where it takes the rows
    with row % 8 == 0 as the rcp rows."""
    return 16 * warp + lane // 4 + 8 * (j // 2), 2 * (lane % 4) + j % 2


def _refill_a_word(lane, kt, e):
    """(plane, token pair) that A word e of refill lane `lane` (below 8) holds at step kt:
    the m64n8k16 A fragment with the planes on rows 0, 1, 8, 9 (rcp hi, zr hi, rcp lo, zr lo).
    This is what `rLcA[4kt + e]` was loaded with in the scalar layout, and what the lc wgmma reads."""
    return lane // 4 + 2 * (e % 2), 8 * kt + lane % 4 + 4 * (e // 2)


def _cost(instructions):
    """Per-instruction (actual, ideal) of a list of (width, op, accesses), where `accesses` is
    (lane, byte address) for each active lane."""
    return [wavefronts(accesses, width, op) for width, op, accesses in instructions]


# ---------------------------------------------------------------------------------------------
# sLcFactors
# ---------------------------------------------------------------------------------------------

def _lc_elem_today(plane, token, wg):
    """The earlier token-fastest allocation, make_ordered_layout((4, 64, 2), order=(1, 0, 2))."""
    return token + PAGE * plane + PLANES * PAGE * wg


def _lc_elem(kernel):
    shape, stride = kernel["_lc_factor_layout"](PAGE)
    return lambda plane, token, wg: _crd2idx((plane, token, wg), shape, stride)


def _lc_word(kernel):
    shape, stride = kernel["_lc_factor_words"](PAGE)
    return lambda e, lane, kt, wg: _crd2idx((e, lane, kt, wg), shape, stride)


def test_the_lc_layout_is_a_bijection_onto_the_same_1024_bytes(kernel):
    shape, stride = kernel["_lc_factor_layout"](PAGE)
    assert _size(shape) == PLANES * PAGE * WARPGROUPS
    elem = _lc_elem(kernel)
    cells = sorted(elem(p, t, g) for p in range(PLANES) for t in range(PAGE) for g in range(WARPGROUPS))
    assert cells == list(range(PLANES * PAGE * WARPGROUPS))
    words_shape, _ = kernel["_lc_factor_words"](PAGE)
    assert _size(words_shape) * 4 == len(cells) * 2


def test_each_refill_load_delivers_the_words_the_wgmma_reads(kernel):
    """For every warpgroup, refill lane, step and word: the 16-byte slice [None, lane, kt, wg]
    of the word view is contiguous and 16-byte aligned, and its word e holds, in its low and
    high halves, the bf16 values the writer stored for (plane, 2*pair) and (plane, 2*pair + 1),
    where (plane, pair) is what the A operand of the lc wgmma (`rLcATile[4kt + e]`, `rLcA` before
    it) has always held. So the reader and the writer describe the same bytes, and the lc
    wgmma's operands are unchanged."""
    elem, word = _lc_elem(kernel), _lc_word(kernel)
    for wg in range(WARPGROUPS):
        for lane in range(8):
            for kt in range(PAGE // 16):
                first = word(0, lane, kt, wg)
                assert (4 * first) % 16 == 0
                for e in range(4):
                    w = word(e, lane, kt, wg)
                    assert w == first + e
                    plane, pair = _refill_a_word(lane, kt, e)
                    assert 4 * w == 2 * elem(plane, 2 * pair, wg)          # low half
                    assert 4 * w + 2 == 2 * elem(plane, 2 * pair + 1, wg)  # high half


def _refill_today(base):
    """16 LDS.32 per warpgroup per tile in the scalar layout: one per (kt, e), lanes 0-7."""
    out = []
    for wg in range(WARPGROUPS):
        for kt in range(PAGE // 16):
            for e in range(4):
                accesses = []
                for lane in range(8):
                    plane, pair = _refill_a_word(lane, kt, e)
                    accesses.append((lane, base + 2 * _lc_elem_today(plane, 2 * pair, wg)))
                out.append((4, "ld", accesses))
    return out


def _refill(kernel, base):
    """4 LDS.128 per warpgroup per tile on the whole-D body: one per kt, warp 0's lanes 0-7."""
    word = _lc_word(kernel)
    return [(16, "ld", [(lane, base + 4 * word(0, lane, kt, wg)) for lane in range(8)])
            for wg in range(WARPGROUPS) for kt in range(PAGE // 16)]


def _tile_copy(kernel, base):
    """4 LDS.128 per warp per tile on the grouped body: one per kt, every lane, lane t at lane
    t % 8's slice."""
    word = _lc_word(kernel)
    return [(16, "ld", [(lane, base + 4 * word(0, lane % 8, kt, wg)) for lane in range(32)])
            for wg in range(WARPGROUPS) for warp in range(4) for kt in range(PAGE // 16)]


def _lc_writer(elem, base):
    """8 STS.U16 per warp per tile, for each warpgroup: planes 0-3 of values j in {0, 2}, by
    the lanes whose column is 0 (lane % 4 == 0), at the token the QK C fragment gives."""
    out = []
    for wg in range(WARPGROUPS):
        for warp in range(4):
            for j in (0, 2):
                for plane in range(PLANES):
                    accesses = []
                    for lane in range(32):
                        token, column = _c_fragment(warp, lane, j)
                        if column == 0:
                            accesses.append((lane, base + 2 * elem(plane, token, wg)))
                    assert len(accesses) == 8
                    out.append((2, "st", accesses))
    return out


@pytest.mark.parametrize("base", BASES)
def test_the_refill_is_one_conflict_free_16_byte_load_per_step(kernel, base):
    today = _cost(_refill_today(base))
    assert today == [(2, 1)] * (16 * WARPGROUPS)          # ncu: 2.00 / 1.00
    whole = _cost(_refill(kernel, base))
    assert whole == [(1, 1)] * (4 * WARPGROUPS)
    # the grouped body, every lane: each half warp reads the step's one 128-byte row, so each
    # load is two wavefronts, its ideal
    grouped = _cost(_tile_copy(kernel, base))
    assert grouped == [(2, 2)] * (4 * 4 * WARPGROUPS)
    # per warpgroup per tile: 32 wavefronts in 16 loads in the scalar layout, 4 in 4 on the
    # whole-D body now, and 32 in 16 on the grouped body
    assert sum(a for a, _ in today) // WARPGROUPS == 32
    assert sum(a for a, _ in whole) // WARPGROUPS == 4
    assert sum(a for a, _ in grouped) // WARPGROUPS == 32


@pytest.mark.parametrize("base", BASES)
def test_the_lc_writer_stays_at_its_ideal(kernel, base):
    """No excess moves to the only other access of the buffer: each of the writer's 2-byte
    stores touches four distinct words on four banks in both layouts (ncu:
    1.00 / 1.00 on 8 threads)."""
    for elem in (_lc_elem_today, _lc_elem(kernel)):
        assert _cost(_lc_writer(elem, base)) == [(1, 1)] * (8 * 4 * WARPGROUPS)


def test_the_lc_writer_address_is_one_lane_base_plus_an_immediate(kernel):
    """The new writer needs no instruction in the tile loop: its address is a lane part that
    does not depend on the tile plus a per-store constant, as it was before. The lane part
    is 2(i%2) + 16(i//2) + 128w bytes (i = lane // 4), and the constants are 64(p%2) + 4(p//2)
    + 8(j//2)."""
    elem = _lc_elem(kernel)
    for warp in range(4):
        for lane in range(0, 32, 4):
            i = lane // 4
            for j in (0, 2):
                token, _ = _c_fragment(warp, lane, j)
                for plane in range(PLANES):
                    lane_part = 2 * (i % 2) + 16 * (i // 2) + 128 * warp
                    constant = 64 * (plane % 2) + 4 * (plane // 2) + 8 * (j // 2)
                    assert 2 * elem(plane, token, 0) == lane_part + constant


# ---------------------------------------------------------------------------------------------
# The kernel goes through these helpers
# ---------------------------------------------------------------------------------------------

def _method(tree, name):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IntPingPongDecodeSm90")
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _calls(node, name):
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]


def _subscripts(node, name):
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == name]


def _binds(target, name):
    """Whether an assignment target binds `name`, alone or anywhere inside a tuple or list."""
    return any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store)
               for n in ast.walk(target))


def _assigns(node, name):
    """Every statement in `node` that binds `name`: a plain, annotated or augmented assignment,
    a walrus, a for-loop or with-statement target, anywhere in the subtree."""
    found = []
    for n in ast.walk(node):
        if isinstance(n, ast.Assign):
            targets = n.targets
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = [n.target]
        elif isinstance(n, (ast.For, ast.comprehension)):
            targets = [n.target]
        elif isinstance(n, ast.withitem) and n.optional_vars is not None:
            targets = [n.optional_vars]
        else:
            continue
        if any(_binds(t, name) for t in targets):
            found.append(n)
    return found


def _enclosing_ifs(root, node):
    """(condition, branch) of every `if` in `root` that encloses `node`, outermost first."""
    return [(ast.unparse(n.test), branch) for n in ast.walk(root) if isinstance(n, ast.If)
            for branch in ("body", "orelse")
            if any(node is m for s in getattr(n, branch) for m in ast.walk(s))]


def _only_binding(node, name):
    binds = _assigns(node, name)
    assert len(binds) == 1, f"{name} is bound {len(binds)} times"
    return ast.unparse(binds[0])


def test_the_kernel_allocates_and_reads_the_lc_strip_through_its_helpers():
    """The bank arithmetic above is computed from the two helpers, so it describes the kernel
    only if the helpers' shape and stride are exactly what the allocation and the refill's word
    view are built from: each name bound once, from the helper called with the page size, and
    passed unchanged to the layout. On each body the copy must also sit under exactly that body's
    `if` conditions (the lc wgmma's, and lanes 0-7 on the whole-D body), in the tile loop and in no
    loop beyond its own k-step loop, after the operand's allocation and before the fence and the
    lc wgmma that read its registers, and no other statement may name the operand."""
    tree = ast.parse(KERNEL.read_text())
    kernel_fn, consumer = _method(tree, "kernel"), _method(tree, "_consumer_wg")
    assert len(_calls(kernel_fn, "_lc_factor_layout")) == 1
    assert len(_calls(consumer, "_lc_factor_words")) == 1
    for name in ("lc_shape", "lc_stride"):
        assert _only_binding(kernel_fn, name) == "lc_shape, lc_stride = _lc_factor_layout(Nkv)"
    assert _only_binding(kernel_fn, "sLcFactors") == \
        "sLcFactors = smem.allocate_tensor(BF16, cute.make_layout(lc_shape, stride=lc_stride), byte_alignment=16)"
    for name in ("lc_words_shape", "lc_words_stride"):
        assert _only_binding(consumer, name) == "lc_words_shape, lc_words_stride = _lc_factor_words(Nkv)"
    assert _only_binding(consumer, "sLcWords") == (
        "sLcWords = cute.make_tensor(cute.recast_ptr(sLcFactors.iterator, dtype=Int32), "
        "cute.make_layout(lc_words_shape, stride=lc_words_stride))")
    # No loop-carried copy of the operand any more: the restart (`pv_tile_restart`) loads it
    # per tile on both bodies.
    assert _assigns(consumer, "rLcA") == []
    reads = _subscripts(consumer, "sLcWords")
    assert len(reads) == 2
    restart = ("cutlass.const_expr(self.pv_tile_restart)", "body")
    whole, grouped = ("cutlass.const_expr(W is None)", "body"), ("cutlass.const_expr(W is None)", "orelse")
    # The whole-D body: lanes 0-7 of warp 0, one 16-byte load per step from the lane's own slice.
    # The whole block is pinned, because any statement inside it (a rebinding of the slice, an
    # offset iterator) could hand a lane another lane's bytes while every name above stays bound
    # once.
    blocks = [n for n in ast.walk(consumer) if isinstance(n, ast.If)
              and ast.unparse(n.test) == "wg_thread_idx < 8"
              and any(r in list(ast.walk(n)) for r in reads)]
    assert len(blocks) == 1
    assert ast.unparse(blocks[0]) == (
        "if wg_thread_idx < 8:\n"
        "    for kt in cutlass.range_constexpr(pv_k_steps):\n"
        "        sLcRow = sLcWords[None, wg_thread_idx, kt, consumer_wg_idx]\n"
        "        cute.autovec_copy(cute.make_tensor(sLcRow.iterator.align(16), sLcRow.layout), rLcATile[None, kt])")
    assert len(_assigns(consumer, "sLcRow")) == 1
    # The grouped body: every thread, at row t % 8.
    (tile_read,) = [r for r in reads if r not in list(ast.walk(blocks[0]))]
    (loop,) = [n for n in ast.walk(consumer) if isinstance(n, ast.For)
               and ast.unparse(n.iter) == "cutlass.range_constexpr(pv_k_steps)"
               and tile_read in list(ast.walk(n))]
    assert ast.unparse(loop) == (
        "for kt in cutlass.range_constexpr(pv_k_steps):\n"
        "    sLcRowTile = sLcWords[None, wg_thread_idx % 8, kt, consumer_wg_idx]\n"
        "    cute.autovec_copy(cute.make_tensor(sLcRowTile.iterator.align(16), sLcRowTile.layout), rLcATile[None, kt])")
    assert len(_assigns(consumer, "sLcRowTile")) == 1
    # One per-tile allocation on each body, and nothing else writes the operand: the rows of
    # threads 8 to 127 are left as they are, because A row r feeds only C row r and the epilogue
    # reads the l and c rows from threads 0 to 7 only.
    assert [ast.unparse(b) for b in _assigns(consumer, "rLcATile")] == [
        "rLcATile = cute.make_rmem_tensor((4, pv_k_steps), Int32)",
        "rLcATile = cute.make_rmem_tensor((4, pv_k_steps), Int32)"]
    tile_stores = [n for n in ast.walk(consumer) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Subscript) and ast.unparse(t.value) == "rLcATile"
                           for t in n.targets)]
    assert tile_stores == []
    # and no statement of any other kind names the operand: every one is an allocation, a copy
    # into it, the fence that passes it or the lc wgmma that reads it, so no fill, augmented store
    # or call can rewrite it between the copy and the wgmma
    parent = {c: n for n in ast.walk(consumer) for c in ast.iter_child_nodes(n)}
    naming = []
    for n in ast.walk(consumer):
        if isinstance(n, ast.Name) and n.id == "rLcATile":
            p = n
            while not isinstance(p, ast.stmt):
                p = parent[p]
            if not any(p is q for q in naming):
                naming.append(p)
    wgmma = ("_wgmma_n8k16_bf16(rLcTile, rLcATile[4 * kt], rLcATile[4 * kt + 1], rLcATile[4 * kt + 2], "
             "rLcATile[4 * kt + 3], smem_b_desc, 0 if kt == 0 else 1, 0)")
    assert sorted(ast.unparse(p) for p in naming) == sorted([
        "rLcATile = cute.make_rmem_tensor((4, pv_k_steps), Int32)",
        "rLcATile = cute.make_rmem_tensor((4, pv_k_steps), Int32)",
        "cute.autovec_copy(cute.make_tensor(sLcRow.iterator.align(16), sLcRow.layout), rLcATile[None, kt])",
        "cute.autovec_copy(cute.make_tensor(sLcRowTile.iterator.align(16), sLcRowTile.layout), rLcATile[None, kt])",
        "_wgmma_fence(rLcATile[None, kt])", "_wgmma_fence(rLcATile[None, kt])", wgmma, wgmma])
    # On each body the copy, the allocation and the one lc wgmma that reads the operand sit under
    # the same guards, so the wgmma never reads an operand another configuration skipped.
    tile_wgmma = [c for c in _calls(consumer, "_wgmma_n8k16_bf16")
                  if any(isinstance(a, ast.Subscript) and ast.unparse(a.value) == "rLcATile"
                         for a in c.args)]
    assert len(tile_wgmma) == 2
    allocs = _assigns(consumer, "rLcATile")
    fences = [c for c in _calls(consumer, "_wgmma_fence")
              if any(isinstance(a, ast.Subscript) and ast.unparse(a.value) == "rLcATile" for a in c.args)]
    tile_loop = next(n for n in ast.walk(consumer) if isinstance(n, ast.While)
                     and ast.unparse(n.test) == "wg_tile_idx < wg_tile_count")

    def loops_in_tile(node):
        """The loops between the tile loop and `node`, innermost last."""
        found = []

        def walk(n, path):
            if n is node:
                found.extend(path)
                return True
            return any(walk(c, path + [n] if isinstance(n, (ast.For, ast.While)) else path)
                       for c in ast.iter_child_nodes(n))

        for child in ast.iter_child_nodes(tile_loop):
            if walk(child, []):
                break
        return [ast.unparse(f.target) if isinstance(f, ast.For) else "while" for f in found]

    grouped_restart = [grouped, ("cutlass.const_expr(self.lc_ones_quad)", "orelse"), restart]
    for guards, copy in (([whole, restart], blocks[0]), (grouped_restart, loop)):
        # the copy runs under exactly the guards of its body (on the whole-D body, and lanes 0-7),
        # in the tile loop and no loop of its own beyond the k-step loop it holds
        assert _enclosing_ifs(consumer, copy) == guards
        (alloc,) = [a for a in allocs if _enclosing_ifs(consumer, a) == guards]
        (call,) = [c for c in tile_wgmma if _enclosing_ifs(consumer, c) == guards]
        (fence,) = [f for f in fences if _enclosing_ifs(consumer, f) == guards]
        assert loops_in_tile(alloc) == [] and loops_in_tile(copy) == []
        assert loops_in_tile(fence) == ["kt"] and loops_in_tile(call) == ["kt"]
        # allocated, then loaded, then fenced and read by the lc wgmma, in that order in the tile
        assert (alloc.lineno, alloc.col_offset) < (copy.lineno, copy.col_offset) \
            < (fence.lineno, fence.col_offset) < (call.lineno, call.col_offset)


def test_the_lc_writer_takes_its_token_from_each_fragment():
    """The writer's four stores per fragment keep the subscript (plane, mj_l, warpgroup), with
    mj_l read from the QK coordinate of the SAME fragment inside the fragment loop. Fragments 0
    and 2 hold tokens 8 apart, so a token index computed once outside the loop and shared by
    both would write both fragments' factors to one token, and the bank arithmetic above,
    which follows the fragment map, would no longer describe the stores."""
    tree = ast.parse(KERNEL.read_text())
    prepare = _method(tree, "_unpack_k_and_prepare")
    stores = _subscripts(prepare, "sLcFactors")
    assert sorted(ast.unparse(s.slice) for s in stores) == \
        [f"({p}, mj_l, consumer_wg_idx)" for p in range(PLANES)]
    loops = [n for n in ast.walk(prepare) if isinstance(n, ast.For)
             and any(s in list(ast.walk(n)) for s in stores)]
    innermost = min(loops, key=lambda n: len(list(ast.walk(n))))
    assert ast.unparse(innermost.target) == "j"
    # all four C fragments, so the tokens of fragments 2 and 3 get their factors too
    assert ast.unparse(innermost.iter) == "cutlass.range_constexpr(4 if not V_CH else 0)"
    # every binding of mj_l in the whole method, not only in the loop, so a later += or an
    # annotated rebinding cannot move the stores to another token
    binds = _assigns(prepare, "mj_l")
    assert [ast.unparse(b) for b in binds] == ["mj_l, nj_l = tQkCoord[j]"]
    assert binds[0] in list(ast.walk(innermost))
