# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: MIT
"""The shared-memory addressing that removed `IntPingPongDecodeSm90`'s register spills in the
configurations the census and its compile matrix checked (1 to 5 and 8 query heads per kv head,
not 6 or 7), on a CPU.

Two roles of the kernel spilled registers to local memory at D = 128 before this
addressing split. The channel-K Q'-plane builder in IO warps 10-11
carried one swizzled address per (head, channel quarter) plane store and the query block across
its group loop; the consumers under channel-wise V carried four copies of the lc wgmma's
constant ones-row operand. The fix is three shared-memory layouts, each a plain-Python helper
in the kernel file:

- `_plane_store_split` writes each plane store as a lane term plus a trace-time immediate, for
  a plane base aligned to the swizzle's period (`_swizzle_period`);
- `_build_q_layout` is `sBuildQ`, where each builder lane stages its query words once per CTA;
- `_lc_ones_layout` is `sLcOnes`, the 32-byte constant each consumer thread loads once.

This file checks the store split for every lane, head and quarter at D = 128 and 256 and at
every GQA ratio the constructor admits, 1 to 8, with negative controls; the bank arithmetic of
every access to the two new buffers, in the suite's one shared-memory bank model,
`tests/_smem_banks.py`; and, on the kernel's syntax tree, that the builder and the consumers use
these paths under the gates that select them, while the token-wise K and V paths keep their
code.

THE PLANE LAYOUT. `__call__` builds the plane layout with
`sm90_utils.make_smem_layout(BF16, LayoutEnum.ROW_MAJOR, (PG, D))`: the 128-byte-swizzle
K-major atom, 8 rows of 64 bf16 channels, tiled D / 64 times along the channels, so channel d of
row h sits at element 64 h + d % 64 + 512 (d // 64), under the byte-address swizzle
Swizzle<3, 4, 3>. `_plane_layout` below is that model. It is anchored to the compiled kernel at
D = 128 in two ways: the builder's unsplit plane-store addresses were
`base + 2 lane + 0x80 h + {0, 0x40, 0x400, 0x440}` followed by the XOR of bits 7-9 into bits
4-6, which is this layout and swizzle;
and the kernel with the split emits `st.shared.b16 [lane_reg + imm]` with exactly the immediates
`test_the_split_reproduces_the_compiled_store_immediates` derives. At D = 256 the model is the
same atom tiled four times; the constructor refuses channel-wise K at D != 128, so no compiled
D = 256 plane exists to compare with.

WHAT THE STRUCTURAL TESTS PIN. Where a test interprets the kernel's code, it binds the operands of
the kernel's own packing helpers (`_cvt2_bf16x2`, `_bf16x2_lo_f32`, `_bf16x2_hi_f32`) and of
`cutlass.range_constexpr` by parameter, positionally or by keyword, so respelling those calls
passes. Everything else a test reads is pinned as the kernel writes it, by position or as source
text. The lc wgmma calls are pinned whole, because the claim is each call's complete operand
list. The copies into and out of the register tensors, `cute.make_rmem_tensor` and the
`.to(Float32)` widenings are DSL calls whose signatures this CPU test cannot import to bind
against. A behaviour-preserving respelling of any pinned call, keyword arguments for example,
therefore fails its test and must update the pin together with the kernel.
"""
from __future__ import annotations

import ast
import itertools
import pathlib
import types

import pytest

from tests import test_sm90_lc_refill_banks_cpu as banks
from tests._smem_banks import wavefronts

ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNEL = ROOT / "src/foscar/_sm90/int_decode_pingpong.py"
HELPERS = ("_swizzle_period", "_swizzle_of", "_plane_store_split", "_build_q_layout", "_lc_ones_layout")

PG = 8                        # q_head_tile, the plane's row count
SWIZZLE_128B = (3, 4, 3)      # Swizzle<3, 4, 3> on byte addresses
BF16_BYTES = 2
HEAD_DIMS = (128, 256)
RATIOS = (1, 2, 3, 4, 5, 6, 7, 8)   # every ratio the constructor admits (heads_per_kv <= q_head_tile)
ONE_PAIR = 0x3F803F80         # bf16x2 (1.0, 1.0)
BASES = range(0, 128, 16)     # both new buffers are allocated at 16-byte alignment


@pytest.fixture(scope="module")
def kernel():
    """The kernel's addressing helpers, executed from its source without the DSL."""
    tree = ast.parse(KERNEL.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS]
    assert sorted(n.name for n in nodes) == sorted(HELPERS)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(KERNEL), "exec"), namespace)
    return namespace


def _plane_layout():
    """Unswizzled element offset of (h, d) in the plane layout `make_smem_layout` builds, at any
    head dimension: the atom's 8 x 64 elements repeat along the channels."""
    return lambda h, d: 64 * h + d % 64 + 512 * (d // 64)


def _swizzle(address, swizzle=SWIZZLE_128B):
    bits, base, shift = swizzle
    return address ^ ((address >> shift) & (((1 << bits) - 1) << base))


# ---------------------------------------------------------------------------------------------
# The plane-store split
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("heads", RATIOS)
def test_every_plane_store_is_a_lane_term_plus_an_immediate(kernel, head_dim, heads):
    """For every head, quarter and lane: the split's base_elem + (lane ^ lane_xor) is the
    element the swizzled plane holds (h, 32 qtr + lane) at, recomputed here independently of
    the helper; base_elem is a multiple of 32, so the immediate never carries into the lane
    term; and the stores that differ only by lane_xor share one lane register, of which there
    are min(heads, 4)."""
    offset = _plane_layout()
    split = kernel["_plane_store_split"](offset, heads, head_dim // 32, SWIZZLE_128B, BF16_BYTES,
                                         kernel["_swizzle_period"](SWIZZLE_128B))
    assert sorted(split) == [(h, q) for h in range(heads) for q in range(head_dim // 32)]
    for (h, qtr), (base_elem, lane_xor) in split.items():
        assert base_elem % 32 == 0 and 0 <= lane_xor < 32
        for lane in range(32):
            expected = _swizzle(BF16_BYTES * offset(h, 32 * qtr + lane)) // BF16_BYTES
            assert base_elem + (lane ^ lane_xor) == expected, (h, qtr, lane)
    assert len({lane_xor for _, lane_xor in split.values()}) == min(heads, 4)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("heads", RATIOS)
def test_the_split_stores_every_channel_once_inside_the_plane(kernel, head_dim, heads):
    """The builder's stores are a bijection from (real head, channel) onto the elements the
    swizzled plane holds those rows at, inside the plane's PG x D elements."""
    split = kernel["_plane_store_split"](_plane_layout(), heads, head_dim // 32,
                                         SWIZZLE_128B, BF16_BYTES, 1024)
    stored = [base + (lane ^ x) for (h, q), (base, x) in split.items() for lane in range(32)]
    assert len(set(stored)) == len(stored) == heads * head_dim
    assert all(0 <= e < PG * head_dim for e in stored)


def test_the_split_reproduces_the_compiled_store_immediates(kernel):
    """At four heads and D = 128 the compiled builder stores through four registers, one per
    head (%r76-%r79 for heads 0-3), with byte immediates 128 h + {0, 64, 1024, 1088} per
    quarter: `st.shared.b16 [%r76]`, `[%r77+128]`, `[%r78+256]`, `[%r79+384]`, `[%r76+64]` and
    so on in the PTX of flex_i4_K-ch128_V-ch128. The split gives exactly those
    immediates and one lane XOR per head, 8 h, so a head's four quarters share one lane term, as
    they share one register there."""
    split = kernel["_plane_store_split"](_plane_layout(), 4, 4, SWIZZLE_128B, BF16_BYTES, 1024)
    quarter_bytes = (0, 64, 1024, 1088)
    assert sorted(split) == [(h, q) for h in range(4) for q in range(4)]
    for (h, qtr), (base_elem, lane_xor) in split.items():
        assert BF16_BYTES * base_elem == 128 * h + quarter_bytes[qtr]
        assert lane_xor == 8 * h


@pytest.mark.parametrize("alignment", (128, 256, 512))
def test_a_plane_below_the_swizzle_period_is_refused(kernel, alignment):
    with pytest.raises(ValueError, match="swizzle's period"):
        kernel["_plane_store_split"](_plane_layout(), 4, 4, SWIZZLE_128B, BF16_BYTES, alignment)


@pytest.mark.parametrize("name, offset, swizzle", [
    # the channels of a quarter are not contiguous: one head per element, channel-major
    ("channel-major", lambda h, d: PG * d + h, SWIZZLE_128B),
    # a swizzle that permutes the lanes of a quarter among themselves, which no single
    # lane XOR describes
    ("in-quarter swizzle", _plane_layout(), (1, 1, 2)),
    # the whole plane shifted by 16 elements, so a quarter's 32 lanes straddle a 32-element
    # boundary
    ("unaligned quarter", lambda h, d: 16 + 64 * h + d % 64 + 512 * (d // 64), SWIZZLE_128B),
])
def test_a_layout_the_split_does_not_describe_is_refused(kernel, name, offset, swizzle):
    with pytest.raises(ValueError, match="not a trace-time part plus a lane XOR"):
        kernel["_plane_store_split"](offset, 4, 4, swizzle, BF16_BYTES, 1024)


def test_the_swizzle_period_of_the_128_byte_swizzle_is_1024(kernel):
    assert kernel["_swizzle_period"](SWIZZLE_128B) == 1024


# ---------------------------------------------------------------------------------------------
# sBuildQ
# ---------------------------------------------------------------------------------------------

def _build_q_word(kernel, heads, head_dim):
    shape, stride = kernel["_build_q_layout"](heads, head_dim)
    return lambda w, lane, qtr, warp: banks._crd2idx((w, lane, qtr, warp), shape, stride)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("heads", RATIOS)
def test_sbuildq_holds_every_head_of_every_lane_once(kernel, head_dim, heads):
    """The word count per (lane, quarter) is the heads' bf16x2 pair count rounded up to a
    power of two, so one vector access moves them; the layout is a bijection onto its
    words; and each lane's words for a quarter are contiguous and aligned to their size."""
    shape, _ = kernel["_build_q_layout"](heads, head_dim)
    words = shape[0]
    pairs = (heads + 1) // 2
    assert words & (words - 1) == 0 and words >= pairs > words // 2
    assert shape[1:] == (32, head_dim // 32, 2)
    word = _build_q_word(kernel, heads, head_dim)
    cells = sorted(word(w, l, q, p) for w in range(words) for l in range(32)
                   for q in range(head_dim // 32) for p in range(2))
    assert cells == list(range(words * 32 * (head_dim // 32) * 2))
    for lane, qtr, warp in itertools.product(range(32), range(head_dim // 32), range(2)):
        first = word(0, lane, qtr, warp)
        assert [word(w, lane, qtr, warp) for w in range(words)] == list(range(first, first + words))
        assert (4 * first) % (4 * words) == 0


def _sbuildq_accesses(kernel, heads, head_dim, base):
    """One warp-wide access per (quarter, builder warp), each lane moving its `words` words:
    the staging store before the group loop and the per-build load inside it are the same
    access."""
    shape, _ = kernel["_build_q_layout"](heads, head_dim)
    words, word = shape[0], _build_q_word(kernel, heads, head_dim)
    return [(4 * words, [(lane, base + 4 * word(0, lane, qtr, warp)) for lane in range(32)])
            for qtr in range(head_dim // 32) for warp in range(2)]


@pytest.mark.parametrize("op", ("st", "ld"))
@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("heads", RATIOS)
def test_every_sbuildq_access_is_at_its_ideal(kernel, heads, head_dim, base, op):
    """The staging store and the per-build load alike: every warp-wide 4-, 8- or 16-byte
    access of a lane's words takes the ideal number of wavefronts, one per 128 bytes the warp
    moves."""
    accesses = _sbuildq_accesses(kernel, heads, head_dim, base)
    words = kernel["_build_q_layout"](heads, head_dim)[0][0]
    for actual, ideal in (wavefronts(lanes, width, op) for width, lanes in accesses):
        assert actual == ideal == words


# ---------------------------------------------------------------------------------------------
# sLcOnes
# ---------------------------------------------------------------------------------------------

def _lc_ones_word(kernel):
    shape, stride = kernel["_lc_ones_layout"]()
    return lambda e, row: banks._crd2idx((e, row), shape, stride)


def test_slcones_is_two_16_byte_rows(kernel):
    shape, _ = kernel["_lc_ones_layout"]()
    assert shape == (4, 2)
    word = _lc_ones_word(kernel)
    assert sorted(word(e, r) for e in range(4) for r in range(2)) == list(range(8))
    for row in range(2):
        assert [word(e, row) for e in range(4)] == list(range(4 * row, 4 * row + 4))


@pytest.mark.parametrize("base", BASES)
def test_the_slcones_load_is_at_its_ideal(kernel, base):
    """Each consumer warp loads once per CTA, each thread its row: warpgroup threads 0-3 row 0,
    every other thread row 1. Lanes on the same row are a broadcast, so every warp's load is
    at its ideal."""
    word = _lc_ones_word(kernel)
    for warp in range(4):
        accesses = [(lane, base + 4 * word(0, 0 if 32 * warp + lane < 4 else 1)) for lane in range(32)]
        for actual, ideal in [wavefronts(accesses, 16, "ld")]:
            assert actual == ideal


# ---------------------------------------------------------------------------------------------
# The kernel goes through these helpers, under the right gates
# ---------------------------------------------------------------------------------------------

_method, _calls, _subscripts, _only_binding = (
    banks._method, banks._calls, banks._subscripts, banks._only_binding)


def _calls_attr(node, dotted):
    """Every call in `node` whose callee is the dotted name `dotted`, in source order."""
    return sorted((n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == dotted),
                  key=lambda n: (n.lineno, n.col_offset))


@pytest.fixture(scope="module")
def tree():
    return ast.parse(KERNEL.read_text())


def _enclosing_ifs(root, target):
    """The test of every `if` (compile-time or runtime) and every conditional expression that
    encloses `target`, outermost first, with the branch it sits in."""
    found = []

    def walk(node, path):
        if node is target:
            found.extend(path)
            return True
        for field, value in ast.iter_fields(node):
            for child in (value if isinstance(value, list) else [value]):
                if not isinstance(child, ast.AST):
                    continue
                step = path
                if isinstance(node, (ast.If, ast.IfExp)) and field in ("body", "orelse"):
                    step = path + [(ast.unparse(node.test), field)]
                if walk(child, step):
                    return True
        return False

    walk(root, [])
    return found


def _enclosing_fors(root, target):
    """Every `for` loop under `root` that encloses `target`, outermost first."""
    found = []

    def walk(node, path):
        if node is target:
            found.extend(path)
            return True
        for child in ast.iter_child_nodes(node):
            if walk(child, path + [node] if isinstance(node, ast.For) else path):
                return True
        return False

    walk(root, [])
    return found


def _statements_naming(root, names):
    """Every innermost statement under `root` that names one of `names`, each once, in source
    order. A write of any kind (an assignment to an element, an augmented assignment, a method
    call on the tensor, passing it to a function) is such a statement."""
    parent = {c: n for n in ast.walk(root) for c in ast.iter_child_nodes(n)}
    found = []
    for n in ast.walk(root):
        if isinstance(n, ast.Name) and n.id in names:
            p = n
            while not isinstance(p, ast.stmt):
                p = parent[p]
            if not any(p is f for f in found):
                found.append(p)
    return sorted(found, key=lambda n: (n.lineno, n.col_offset))


def _may_run(guards, variants):
    """Whether some variant (a dict of names) satisfies every compile-time guard in `guards`. A
    guard that is not `cutlass.const_expr(...)`, or whose value a variant cannot decide, may hold."""
    for env in variants:
        ok = True
        for test, branch in guards:
            node = ast.parse(test, mode="eval").body
            if not (isinstance(node, ast.Call) and ast.unparse(node.func) == "cutlass.const_expr"
                    and len(node.args) == 1):
                continue
            try:
                value = bool(_eval(node.args[0], **env))
            except Exception:
                continue
            if value != (branch == "body"):
                ok = False
                break
        if ok:
            return True
    return False


def _range_of(call, **names):
    """The Python range a `cutlass.range_constexpr(...)` call denotes, `range`'s rule for its
    arguments: one positional argument is the stop; start, stop and step may each be positional
    or a keyword; evaluated with `names`."""
    assert ast.unparse(call.func) == "cutlass.range_constexpr", ast.unparse(call)
    args = [_eval(a, **names) for a in call.args]
    kw = {k.arg: _eval(k.value, **names) for k in call.keywords}
    assert set(kw) <= {"start", "stop", "step"}, kw
    if len(args) == 1 and not ({"start", "stop"} & set(kw)):
        args = [0] + args
    bound = dict(zip(("start", "stop", "step"), args))
    assert not set(bound) & set(kw), (bound, kw)
    bound.update(kw)
    return range(bound.get("start", 0), bound["stop"], bound.get("step", 1))


def _enclosing_const_expr_tests(root, target):
    """The test of every `if cutlass.const_expr(...)` that encloses `target`, outermost first,
    with the branch it sits in ("body" or "orelse")."""
    found = []

    def walk(node, path):
        if node is target:
            found.extend(path)
            return True
        for field, value in ast.iter_fields(node):
            children = value if isinstance(value, list) else [value]
            for child in children:
                if not isinstance(child, ast.AST):
                    continue
                step = path
                if (isinstance(node, ast.If) and field in ("body", "orelse")
                        and ast.unparse(node.test).startswith("cutlass.const_expr(")):
                    step = path + [(ast.unparse(node.test), field)]
                if walk(child, step):
                    return True
        return False

    walk(root, [])
    return found


def _flat_targets(target):
    if isinstance(target, (ast.Tuple, ast.List)):
        return [t for e in target.elts for t in _flat_targets(e)]
    if isinstance(target, ast.Starred):
        return _flat_targets(target.value)
    return [target]


def _binds_of(root, dotted):
    """Every statement under `root` that binds `dotted` (a name, an attribute such as
    `self.lc_ones_quad`, or a subscript such as `rQStage[w]`), as a whole target or as an
    element of a tuple or list target, in source order."""
    found = []
    for n in ast.walk(root):
        if isinstance(n, ast.Assign):
            targets = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr, ast.For)):
            targets = [n.target]
        else:
            continue
        if any(ast.unparse(t) == dotted for target in targets for t in _flat_targets(target)):
            found.append(n)
    return sorted(found, key=lambda n: (n.lineno, n.col_offset))


def _position(node):
    return (node.lineno, node.col_offset)


def _args_by_param(tree, call, method):
    """Map each parameter of `method` to the source of the argument `call` passes for it,
    positionally or by keyword; every parameter must be bound exactly once."""
    params = [a.arg for a in _method(tree, method).args.args][1:]
    assert len(call.args) <= len(params), method
    bound = dict(zip(params, (ast.unparse(a) for a in call.args)))
    for k in call.keywords:
        assert k.arg in params and k.arg not in bound, (method, k.arg)
        bound[k.arg] = ast.unparse(k.value)
    assert set(bound) == set(params), (method, sorted(set(params) - set(bound)))
    return bound


def _bind_helper(tree, call):
    """Map each positional parameter of the kernel's module-level helper that `call` names to the
    source of the argument passed for it, positionally or by keyword; every such parameter must be
    bound exactly once, and nothing else may be passed."""
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == ast.unparse(call.func)]
    params = [a.arg for a in fn.args.args]
    assert len(call.args) <= len(params), ast.unparse(call)
    bound = dict(zip(params, (ast.unparse(a) for a in call.args)))
    for k in call.keywords:
        assert k.arg in params and k.arg not in bound, (ast.unparse(call), k.arg)
        bound[k.arg] = ast.unparse(k.value)
    assert set(bound) == set(params), (ast.unparse(call), sorted(set(params) - set(bound)))
    return bound


def _eval(expr, **names):
    return eval(compile(ast.Expression(expr), "<kernel>", "eval"), {"__builtins__": {}}, names)


def test_the_ones_quad_gate_is_every_fold_on_channel_wise_v_build(tree):
    """One binding, in the constructor, of every form, and it is true exactly for the fold path
    with channel-wise V, whatever K is."""
    binds = _binds_of(tree, "self.lc_ones_quad")
    assert len(binds) == 1 and isinstance(binds[0], ast.Assign) and len(binds[0].targets) == 1
    assert binds[0] in list(ast.walk(_method(tree, "__init__")))
    for fold, k_axis, v_axis in itertools.product((True, False), ("token", "channel"), ("token", "channel")):
        self = type("Kernel", (), dict(fold=fold, k_axis=k_axis, v_axis=v_axis))()
        assert bool(_eval(binds[0].value, self=self)) == (fold and v_axis == "channel")


def test_the_tile_restart_gate_is_every_fold_on_token_wise_v_build(tree):
    """One binding, in the constructor, and it is true exactly for the fold path with token-wise
    V, whatever K is: only there is the output formed from two long sums, the PV sum over q + beta
    and the lc wgmma's correction sum, which on V data centred near zero are large and cancel."""
    binds = _binds_of(tree, "self.pv_tile_restart")
    assert len(binds) == 1 and isinstance(binds[0], ast.Assign) and len(binds[0].targets) == 1
    assert binds[0] in list(ast.walk(_method(tree, "__init__")))
    for fold, k_axis, v_axis in itertools.product((True, False), ("token", "channel"), ("token", "channel")):
        self = type("Kernel", (), dict(fold=fold, k_axis=k_axis, v_axis=v_axis))()
        assert bool(_eval(binds[0].value, self=self)) == (fold and v_axis == "token")


def test_the_restart_starts_each_tile_s_pv_and_lc_sums_from_zero(tree):
    """On the restart path every PV and lc wgmma of the tile loop writes sums allocated in that
    tile, at every k-step of the tile, from zero at the first (scale-D 0); off it the whole-D
    body's PV wgmmas accumulate into the totals in place. After the wait that retires them, the
    tile's sums join the totals in one multiply-add with the max correction, once per tile and once
    for every element of every total (every output half of every group on the grouped body), and
    the in-place rescale of the totals runs only off the restart path, so the correction is
    applied once either way. Each call and each join is checked by the complete chain of guards
    around it and by the ranges of the loops around it, at two tile sizes and at head dimensions
    256 and 512 for the grouped body, so that no extra guard or shortened loop can skip a step, an
    element or a group; each grouped PV wgmma writes the output half of its own loop step; each
    sum is allocated once per tile (per group on the grouped body), ahead of its k-step chain and
    outside it, and is not rebound before the chain ends; a commit and then a wait follow the last
    wgmma writing a join's sums, in the same iteration and under guards the join also has; off the
    restart path the whole-D PV sums are the totals themselves, bound on that body ahead of the
    PV chain; and nothing in the tile loop leaves it early or repeats a statement except the
    `for` loops the test reads (no other `while`, no function or comprehension, and every `with`
    a profiling range). Finally, on the restart path the statements of the tile loop that name a
    total, a tile's sum or the max correction are exactly the softmax that writes the correction,
    the sums' aliases and allocations, the wgmmas that write the sums and the joins, so nothing
    else rescales a total, alters a sum before its join or rewrites the correction."""
    consumer = _method(tree, "_consumer_wg")
    tile_loop = next(n for n in ast.walk(consumer) if isinstance(n, ast.While)
                     and ast.unparse(n.test) == "wg_tile_idx < wg_tile_count")
    in_loop = list(ast.walk(tile_loop))
    # no statement leaves the tile loop or an iteration of it early, so a guard is the only way
    # to skip a statement in it
    assert not [n for n in in_loop if isinstance(n, (ast.Break, ast.Continue, ast.Return, ast.Raise))]
    assert _enclosing_ifs(consumer, tile_loop) == []
    # and nothing repeats a statement except the `for` loops whose ranges are read below: the
    # only compound statements in the tile loop are `if`, `for` and the profiling ranges' `with`
    compound = [n for n in in_loop if n is not tile_loop and isinstance(n, ast.stmt) and hasattr(n, "body")]
    assert all(isinstance(n, (ast.If, ast.For, ast.With)) for n in compound), \
        sorted({type(n).__name__ for n in compound})
    assert all(len(n.items) == 1 and ast.unparse(n.items[0].context_expr).startswith("annotate.range(")
               for n in compound if isinstance(n, ast.With))
    assert not [n for n in in_loop if isinstance(n, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef,
                                                     ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))]
    total_of = {"rPvSumLo": "rOutAccumLo", "rPvSumHi": "rOutAccumHi", "rPvSum": "rOutAccum", "rLcTile": "rLcAccum"}
    base = lambda e: ast.unparse(e.value if isinstance(e, ast.Subscript) else e)
    calls = [c for c in _calls(consumer, "_wgmma_n8k16_bf16") if base(c.args[0]) in total_of]
    assert sorted(base(c.args[0]) for c in calls) == ["rLcTile", "rLcTile", "rPvSum", "rPvSumHi", "rPvSumLo"]
    # what each call writes: a whole sum tensor, or on the grouped body the output half `hh` of
    # its own loop step, so that every half a join reads was written
    assert sorted(ast.unparse(c.args[0]) for c in calls) == sorted([
        "rLcTile", "rLcTile", "rPvSum[None, hh]", "rPvSumHi", "rPvSumLo"])
    on, off = (types.SimpleNamespace(pv_tile_restart=r) for r in (True, False))
    whole, grouped = ("cutlass.const_expr(W is None)", "body"), ("cutlass.const_expr(W is None)", "orelse")
    restart = ("cutlass.const_expr(self.pv_tile_restart)", "body")
    assert [ast.unparse(b) for b in _binds_of(consumer, "v_groups")] == ["v_groups = D // W"]

    def loops(node, pv_k_steps, D, W):
        """(loop variable, the range it runs) of every loop between the tile loop and `node`."""
        sizes = dict(pv_k_steps=pv_k_steps, accum_fragment_size=4, D=D, W=W, v_groups=D // W)
        got = []
        for f in _enclosing_fors(tile_loop, node):
            assert isinstance(f.target, ast.Name) and isinstance(f.iter, ast.Call), ast.unparse(f.target)
            got.append((f.target.id, _range_of(f.iter, **sizes)))
        return got

    # the loops each restart call must sit in, by body: the k-step loop, and on the grouped body
    # the group loop and, for PV, the loop over the group's output halves
    expected_loops = {
        "rPvSumLo": lambda n, D, W: [("kt", range(n))],
        "rPvSumHi": lambda n, D, W: [("kt", range(n))],
        "rPvSum": lambda n, D, W: [("g", range(D // W)), ("kt", range(n)), ("hh", range(W // 64))],
        "rLcTile": lambda n, D, W: [("kt", range(n))],
    }
    # the complete chain of guards around each call: the whole-D PV pair on the whole-D body
    # alone (it also runs off the restart path), the whole-D lc call under the restart, and the
    # grouped PV halves and lc call on the grouped body
    grouped_restart = [grouped, ("cutlass.const_expr(self.lc_ones_quad)", "orelse"), restart]
    assert sorted((base(c.args[0]), _enclosing_ifs(consumer, c)) for c in calls) == sorted([
        ("rPvSumLo", [whole]), ("rPvSumHi", [whole]), ("rPvSum", [grouped]), ("rLcTile", [whole, restart]),
        ("rLcTile", grouped_restart)])
    for c in calls:
        name = base(c.args[0])
        assert c in in_loop
        for n, D in itertools.product((2, 4), (256, 512)):
            got = loops(c, n, D, 128)
            assert got == expected_loops[name](n, D, 128), (ast.unparse(c), got)
            (steps,) = [r for v, r in got if v == "kt"]
            # the k-steps the call actually runs: the first from zero, every later one accumulating
            assert [_eval(c.args[6], self=on, kt=kt) for kt in steps] == [0] + [1] * (n - 1), ast.unparse(c)
            if name in ("rPvSumLo", "rPvSumHi"):
                assert [_eval(c.args[6], self=off, kt=kt) for kt in steps] == [1] * n
    # each call's sum is allocated on the call's body ahead of the call, in the same iteration of
    # every loop around the call other than its k-step and half loops (so once per tile, and once
    # per group on the grouped body), under guards that the call has or the restart's, and nothing
    # rebinds it before the last call on that body that writes it
    for c in calls:
        name = base(c.args[0])
        guards_c = _enclosing_ifs(consumer, c)
        same_body = [d for d in calls if base(d.args[0]) == name and _enclosing_ifs(consumer, d)[0] == guards_c[0]]
        allocs = [b for b in _binds_of(consumer, name) if ast.unparse(b.value).startswith("cute.make_rmem_tensor(")
                  and _enclosing_ifs(consumer, b)[0] == guards_c[0]]
        assert len(allocs) == 1, ast.unparse(c)
        (alloc,) = allocs
        assert _position(alloc) < min(_position(d) for d in same_body)
        assert _enclosing_fors(tile_loop, alloc) == [
            f for f in _enclosing_fors(tile_loop, c) if f.target.id not in ("kt", "hh")], ast.unparse(c)
        guards_a = [x for x in _enclosing_ifs(consumer, alloc) if x != restart]
        assert guards_a == guards_c[:len(guards_a)], ast.unparse(c)
        last_write = max(_position(d) for d in same_body)
        assert not [b for b in _binds_of(consumer, name) if _position(alloc) < _position(b) <= last_write]
    for name, total in total_of.items():
        binds = [ast.unparse(b) for b in _binds_of(consumer, name)]
        allocs = [b for b in _binds_of(consumer, name) if ast.unparse(b.value).startswith("cute.make_rmem_tensor(")]
        assert allocs and all(b in in_loop for b in allocs)
        if name in ("rPvSumLo", "rPvSumHi"):
            # the totals themselves off the restart path, a tensor of the tile's own on it; the
            # alias is bound on the whole-D body under no other guard, in no loop of the tile,
            # ahead of the PV chain, so every whole-D build reaches the chain with it bound
            assert binds[0] == f"{name} = {total}" and len(allocs) == 1 == len(binds) - 1
            assert _enclosing_ifs(consumer, allocs[0])[-1] == restart
            alias = _binds_of(consumer, name)[0]
            assert _enclosing_ifs(consumer, alias) == [whole] and _enclosing_fors(tile_loop, alias) == []
            assert alias in in_loop
            assert _position(alias) < _position(allocs[0]) < min(
                _position(c) for c in calls if base(c.args[0]) == name)
        else:
            assert len(allocs) == len(binds)
    rescales = [n for n in in_loop if isinstance(n, ast.Assign)
                and "rMaxCorrection" in ast.unparse(n.value) and base(n.targets[0]) in total_of.values()]
    assert sorted(ast.unparse(n) for n in rescales) == sorted([
        "rLcAccum[j] = rLcAccum[j] * rMaxCorrection[j]",
        "rLcAccum[j] = rLcAccum[j] * rMaxCorrection[j] + rLcTile[j]",
        "rLcAccum[j] = rLcAccum[j] * rMaxCorrection[j] + rLcTile[j]",
        "rOutAccum[j, W // 64 * g + hh] = rOutAccum[j, W // 64 * g + hh] * rMaxCorrection[j] + rPvSum[j, hh]",
        "rOutAccumHi[j] = rOutAccumHi[j] * rMaxCorrection[j]",
        "rOutAccumHi[j] = rOutAccumHi[j] * rMaxCorrection[j] + rPvSumHi[j]",
        "rOutAccumLo[j] = rOutAccumLo[j] * rMaxCorrection[j]",
        "rOutAccumLo[j] = rOutAccumLo[j] * rMaxCorrection[j] + rPvSumLo[j]"])
    for n in rescales:
        if ast.unparse(n.value).endswith("* rMaxCorrection[j]"):
            assert ("cutlass.const_expr(not self.pv_tile_restart)", "body") in _enclosing_ifs(consumer, n)
    waits = _calls(consumer, "_wgmma_wait0")
    commits = _calls(consumer, "_wgmma_commit")
    for n in rescales:
        if ast.unparse(n.value).endswith("* rMaxCorrection[j]"):
            continue
        (summand,) = [s for s in total_of if f"+ {s}[" in ast.unparse(n.value)]
        on_grouped = grouped in _enclosing_ifs(consumer, n)
        # the complete chain of guards: the restart's joins on the whole-D body, every join on the
        # grouped body, and nothing more
        guards = _enclosing_ifs(consumer, n)
        expected_guards = ((grouped_restart if summand == "rLcTile" else [grouped])
                           if on_grouped else [whole, restart])
        assert guards == expected_guards, (ast.unparse(n), guards)
        # once per tile, over every element: the loops around the join are its element loops
        # (and on the grouped body the group loop), each over its whole range, and nothing else
        for D in (256, 512):
            got = loops(n, 4, D, 128)
            if on_grouped and summand == "rPvSum":
                assert got == [("g", range(D // 128)), ("j", range(4)), ("hh", range(2))], got
                # the output half each (g, hh) joins: every half of the total, each exactly once
                half = n.targets[0].slice.elts[1]
                halves = sorted(_eval(half, W=128, g=g, hh=hh) for g in range(D // 128) for hh in range(2))
                assert halves == list(range(D // 64)), halves
            else:
                assert got == [("j", range(4))], got
        # after a wait that retires the last wgmma writing its sums, on the same body: the wait
        # sits between that wgmma and the join, in the same iteration of every loop around the
        # join other than its element loops (the group loop on the grouped body), in no loop of
        # its own, and under guards that all also guard the join, so it runs whenever the join does
        writes = [c for c in calls if base(c.args[0]) == summand
                  and (grouped in _enclosing_ifs(consumer, c)) == on_grouped]
        last = max(writes, key=_position)
        outer = [f for f in _enclosing_fors(tile_loop, n) if f.target.id not in ("j", "hh")]
        assert _enclosing_fors(tile_loop, last)[:len(outer)] == outer, ast.unparse(n)
        def runs_with_join(x):
            return (_enclosing_fors(tile_loop, x) == outer
                    and _enclosing_ifs(consumer, x) == guards[:len(_enclosing_ifs(consumer, x))])
        # and a commit between that wgmma and the wait, under the same conditions, so that the
        # wait covers the group the wgmma belongs to
        retiring = [w for w in waits if _position(last) < _position(w) < _position(n) and runs_with_join(w)
                    and any(_position(last) < _position(k) < _position(w) and runs_with_join(k)
                            for k in commits)]
        assert retiring, ast.unparse(n)
    # every statement of the tile loop that can run on the restart path (some compile-time setting
    # of it satisfies the statement's guards) and names a total, a tile's sum or the correction
    variants = [dict(self=types.SimpleNamespace(pv_tile_restart=True, fold=True, v_axis="token",
                                                lc_ones_quad=False, k_axis=ka, bits=b),
                     V_CH=False, W=w, kt=kt, g=g, v_groups=2)
                for ka, b, w, kt, g in itertools.product(("token", "channel"), (2, 4), (None, 128), range(4), range(2))]
    named = set(total_of) | set(total_of.values()) | {"rMaxCorrection"}
    on_restart = [n for n in _statements_naming(tile_loop, named) if _may_run(_enclosing_ifs(consumer, n), variants)]
    softmax = "self._online_softmax(K_CH, K_TOK64, K_TOK_QTR, V_CH, consumer_wg_idx, lane, rKScale, rKScaleHi, rKZero, " \
        "rMaxCorrection, rNewMax, rOldMax, rScoreAccum, rScoreAccum2, rVScale, sP, sQRowSum, sRunningMax, " \
        "sWarpMax, tQkCoord, valid_token_count, warp_idx_in_wg, wg_thread_idx)"
    assert sorted(ast.unparse(n) for n in on_restart) == sorted([
        softmax,
        "rPvSumLo = rOutAccumLo", "rPvSumHi = rOutAccumHi",
        "rPvSumLo = cute.make_rmem_tensor(accum_fragment_size, Float32)",
        "rPvSumHi = cute.make_rmem_tensor(accum_fragment_size, Float32)",
        "rPvSum = cute.make_rmem_tensor((accum_fragment_size, W // 64), Float32)",
        "rLcTile = cute.make_rmem_tensor(accum_fragment_size, Float32)",
        "rLcTile = cute.make_rmem_tensor(accum_fragment_size, Float32)",
    ] + [ast.unparse(c) for c in calls] + [ast.unparse(n) for n in rescales
                                            if not ast.unparse(n.value).endswith("* rMaxCorrection[j]")])
    # the correction is written once per tile, by the softmax, before any join reads it
    (sm,) = [n for n in on_restart if ast.unparse(n) == softmax]
    assert all(_position(sm) < _position(n) for n in rescales)


def test_the_consumer_reads_the_ones_quad_under_its_gate(tree):
    """The consumer loads its row of sLcOnes once, with one 16-byte copy into four registers,
    before its tile loop and only under `lc_ones_quad`; the tile's lc wgmmas are exactly four
    calls, each under its complete chain of guards: on the whole-D body one reading the loaded
    quad, one the two-value form and one the per-tile operand of the restart (`pv_tile_restart`),
    and on the grouped body one reading the restart's per-tile operand; the PV fence of the
    quad's first k-step is the only other reader of the quad, passing the same four registers
    through itself; and the two-value form is bound only on the fold-off path."""
    consumer = _method(tree, "_consumer_wg")
    tile_loop = next(n for n in ast.walk(consumer) if isinstance(n, ast.While)
                     and ast.unparse(n.test) == "wg_tile_idx < wg_tile_count")
    # one read of sLcOnes, a whole row (which row each thread takes, the next test evaluates)
    loads = _subscripts(consumer, "sLcOnes")
    assert len(loads) == 1 and isinstance(loads[0].slice, ast.Tuple) and len(loads[0].slice.elts) == 2
    assert ast.unparse(loads[0].slice.elts[0]) == "None"
    (row_bind,) = _binds_of(consumer, "sLcOnesRow")
    assert isinstance(row_bind, ast.Assign) and row_bind.value is loads[0]
    assert [ast.unparse(b) for b in _binds_of(consumer, "rLcOnes")] == \
        ["rLcOnes = None", "rLcOnes = cute.make_rmem_tensor(4, Int32)"]
    copies = [c for c in _calls_attr(consumer, "cute.autovec_copy") if "sLcOnesRow" in ast.unparse(c)]
    assert [ast.unparse(c) for c in copies] == [
        "cute.autovec_copy(cute.make_tensor(sLcOnesRow.iterator.align(16), sLcOnesRow.layout), rLcOnes)"]
    for node in (loads[0], copies[0], _binds_of(consumer, "rLcOnes")[1]):
        # no runtime condition either: every thread of the warpgroup loads its row
        assert _enclosing_ifs(consumer, node) == [("cutlass.const_expr(self.lc_ones_quad)", "body")]
        assert node not in list(ast.walk(tile_loop)) and _position(node) < _position(tile_loop)
    # nothing but the copy and the PV fence writes the four registers, and nothing but the quad
    # wgmma and that fence reads them: every other use of the name (an element store, a method
    # call such as fill) is refused. The fence takes the quad from the operand list of the PV
    # chain's first k-step, `pv_operands`, which is its one other use; the fence passes each
    # register through itself as a tied operand and writes the same value back (`_wgmma_fence`)
    parents = {child: node for node in ast.walk(consumer) for child in ast.iter_child_nodes(node)}
    quad_calls = [c for c in _calls(consumer, "_wgmma_n8k16_bf16") if "rLcOnes" in ast.unparse(c)]
    assert len(quad_calls) == 2
    quad_call = quad_calls[0]
    uses = [n for n in ast.walk(consumer) if isinstance(n, ast.Name) and n.id == "rLcOnes" and isinstance(n.ctx, ast.Load)]
    assert len(uses) == 11
    (append,) = [c for c in _calls_attr(consumer, "pv_operands.append") if ast.unparse(c.args[0]) == "rLcOnes"]
    (grp_fence,) = [c for c in _calls(consumer, "_wgmma_fence") if "rLcOnes" in ast.unparse(c)]
    for n in uses:
        parent = parents[n]
        assert (parent is copies[0] and n is copies[0].args[-1]) or (parent is append and n is append.args[0]) or \
            (parent is grp_fence) or \
            (isinstance(parent, ast.Subscript) and isinstance(parent.ctx, ast.Load) and parents[parent] in quad_calls), ast.unparse(parent)
    assert _enclosing_ifs(consumer, append)[-1] == ("cutlass.const_expr(kt == 0 and self.lc_ones_quad)", "body")
    (pv_fence,) = [c for c in _calls(consumer, "_wgmma_fence")
                   if _position(append) < _position(c) < _position(quad_call)]
    assert ast.unparse(pv_fence) == "_wgmma_fence(*pv_operands)"
    # and the list the quad was appended to is the list the fence takes: no rebinding between
    assert [b for b in _binds_of(consumer, "pv_operands")
            if _position(append) < _position(b) < _position(pv_fence)] == []
    lc_calls = [c for c in _calls(consumer, "_wgmma_n8k16_bf16") if ast.unparse(c.args[0]) in ("rLcAccum", "rLcTile")]
    # the complete chain of guards, runtime ones included, inside the tile loop's PV block
    got = [(ast.unparse(c), [g for g in _enclosing_ifs(consumer, c) if g not in _enclosing_ifs(consumer, tile_loop)
                             and g[0] != "wg_tile_idx < wg_tile_count"]) for c in lc_calls]
    restart, quad = "cutlass.const_expr(self.pv_tile_restart)", "cutlass.const_expr(self.lc_ones_quad)"
    whole, grouped = ("cutlass.const_expr(W is None)", "body"), ("cutlass.const_expr(W is None)", "orelse")
    tile_call = ("_wgmma_n8k16_bf16(rLcTile, rLcATile[4 * kt], rLcATile[4 * kt + 1], rLcATile[4 * kt + 2], "
                 "rLcATile[4 * kt + 3], smem_b_desc, 0 if kt == 0 else 1, 0)")
    assert sorted(got) == sorted([
        ("_wgmma_n8k16_bf16(rLcAccum, rLcOnes[0], rLcOnes[1], rLcOnes[2], rLcOnes[3], smem_b_desc, 1, 0)",
         [whole, (quad, "body")]),
        ("_wgmma_n8k16_bf16(rLcAccum, rLcOnes[0], rLcOnes[1], rLcOnes[2], rLcOnes[3], smem_b_desc, 1, 0)",
         [grouped, (quad, "body")]),
        ("_wgmma_n8k16_bf16(rLcAccum, rLcOneZero[0], rLcOneZero[1], rLcOneZero[0], rLcOneZero[1], smem_b_desc, 1, 0)",
         [whole, (quad, "orelse"), ("cutlass.const_expr(not self.fold)", "body")]),
        (tile_call, [whole, (restart, "body")]),
        (tile_call, [grouped, (quad, "orelse"), (restart, "body")]),
    ])
    assert all(c in list(ast.walk(tile_loop)) for c in lc_calls)
    for name in ("rLcOneZero", "rLcOneZero[0]", "rLcOneZero[1]"):
        binds = _binds_of(consumer, name)
        assert binds
        for bind in binds:
            assert _enclosing_const_expr_tests(consumer, bind) == [("cutlass.const_expr(not self.fold)", "body")]


def test_each_consumer_thread_loads_its_own_row_of_the_ones_constant(tree):
    """Evaluating the kernel's role dispatch, the consumer's warpgroup-thread binding and its row
    selection for every thread of the CTA: consumer A is warps 0-3 and B warps 4-7, each thread's
    warpgroup index is its position in its own warpgroup, and exactly the first four threads of
    each warpgroup load row 0, the ones row, while every other consumer thread loads row 1."""
    kernel_fn = _method(tree, "kernel")
    consumer = _method(tree, "_consumer_wg")
    assert _only_binding(kernel_fn, "thread_idx") == "thread_idx, _, _ = cute.arch.thread_idx()"
    assert _only_binding(kernel_fn, "warp_idx") == "warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())"
    calls = [c for c in ast.walk(kernel_fn) if isinstance(c, ast.Call) and ast.unparse(c.func) == "self._consumer_wg"]
    chains = [(_eval(ast.parse(_args_by_param(tree, c, "_consumer_wg")["consumer_wg_idx"], mode="eval").body),
               [(ast.parse(test, mode="eval").body, branch) for test, branch in _enclosing_ifs(kernel_fn, c)])
              for c in calls]
    (bind,) = _binds_of(consumer, "wg_thread_idx")
    (row_sel,) = [s for s in _subscripts(consumer, "sLcOnes")]
    row_expr = row_sel.slice.elts[1]
    for t in range(384):
        warp = t // 32
        taken = [wg for wg, chain in chains
                 if all(bool(_eval(test, warp_idx=warp)) == (branch == "body") for test, branch in chain)]
        if warp >= 8:
            assert taken == [], t                              # the IO warpgroup runs no consumer
            continue
        (wg,) = taken
        assert wg == warp // 4, t
        wg_thread = _eval(bind.value, thread_idx=t, consumer_wg_idx=wg)
        assert wg_thread == t % 128, t
        assert _eval(row_expr, wg_thread_idx=wg_thread, Int32=int) == (0 if t % 128 < 4 else 1), t


def test_each_builder_warp_stores_only_its_own_plane(tree):
    """Evaluating the kernel's role dispatch, the builder's lane and plane bindings and every
    runtime guard of its two plane stores: the builder runs in warps 10 and 11 only, and warp 10
    stores only into plane 0 (`sQ`) and warp 11 only into plane 1 (`sQPlane1`), every lane of
    each."""
    builder = _method(tree, "_build_q_planes")
    kernel_fn = _method(tree, "kernel")
    # the builder runs in exactly warps 10 and 11, by the kernel's role dispatch
    (call,) = [c for c in ast.walk(kernel_fn) if isinstance(c, ast.Call) and ast.unparse(c.func) == "self._build_q_planes"]
    dispatch = [(ast.parse(test, mode="eval").body, branch) for test, branch in _enclosing_ifs(kernel_fn, call)]
    assert {w for w in range(12)
            if all(bool(_eval(test, warp_idx=w)) == (branch == "body") for test, branch in dispatch)} == {10, 11}
    (plane_bind,) = _binds_of(builder, "builder_plane")
    (lane_bind,) = _binds_of(builder, "builder_lane")
    k_guard = ("cutlass.const_expr(self.k_axis == 'channel')", "body")
    for name, plane in (("sQFlat", 0), ("sQPlane1Flat", 1)):
        (store,) = _subscripts(builder, name)
        guards = _enclosing_ifs(builder, store)
        assert guards[:2] == [k_guard, ("tile_count > 0", "body")], guards
        runtime = [(ast.parse(test, mode="eval").body, branch) for test, branch in guards[2:]]
        stores = set()
        for t in range(320, 384):
            env = {"builder_plane": _eval(plane_bind.value, warp_idx=t // 32),
                   "builder_lane": _eval(lane_bind.value, thread_idx=t)}
            if all(bool(_eval(test, **env)) == (branch == "body") for test, branch in runtime):
                stores.add((t // 32, env["builder_lane"]))
        assert stores == {(10 + plane, lane) for lane in range(32)}, (name, sorted(stores))


def test_the_consumer_calls_pass_the_ones_buffer(tree):
    kernel_fn = _method(tree, "kernel")
    calls = [c for c in ast.walk(kernel_fn) if isinstance(c, ast.Call)
             and ast.unparse(c.func) == "self._consumer_wg"]
    assert len(calls) == 2
    for call in calls:
        args = _args_by_param(tree, call, "_consumer_wg")
        assert "sLcOnes" in args, sorted(args)
        for param, arg in args.items():
            if param != "consumer_wg_idx":
                assert arg == param, (param, arg)


def test_the_kernel_writes_slcones_before_the_cta_barrier(tree):
    """Thread 0 writes every word of both rows before the prologue's CTA barrier, which
    publishes them, only under the gate that allocates the buffer; interpreting the writer
    gives row 0 = (ones, 0, ones, 0) and row 1 = zeros, the words the fold-off form gives
    threads 0-3 and every other thread."""
    kernel_fn = _method(tree, "kernel")
    for name in ("lc_ones_shape", "lc_ones_stride"):
        assert _only_binding(kernel_fn, name) == "lc_ones_shape, lc_ones_stride = _lc_ones_layout()"
    allocs = _binds_of(kernel_fn, "sLcOnes")
    assert [ast.unparse(a) for a in allocs] == [
        "sLcOnes = sQRowSum",
        "sLcOnes = smem.allocate_tensor(Int32, cute.make_layout(lc_ones_shape, stride=lc_ones_stride), byte_alignment=16)"]
    assert _enclosing_const_expr_tests(kernel_fn, allocs[1]) == [("cutlass.const_expr(self.lc_ones_quad)", "body")]
    stores = [n for n in ast.walk(kernel_fn) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Subscript) and ast.unparse(t.value) == "sLcOnes" for t in n.targets)]
    (loop,) = {n for n in ast.walk(kernel_fn) if isinstance(n, ast.For) and all(s in n.body for s in stores)}
    assert ast.unparse(loop.iter).startswith("cutlass.range_constexpr(") and loop.body == stores
    words = {}
    for e in _range_of(loop.iter):
        for st in stores:
            words[_eval(st.targets[0].slice, **{loop.target.id: e})] = _eval(st.value, Int32=int, **{loop.target.id: e})
    assert words == {(e, row): (ONE_PAIR if row == 0 and e % 2 == 0 else 0) for e in range(4) for row in range(2)}
    for st in stores:
        assert _enclosing_ifs(kernel_fn, st) == [("thread_idx == 0", "body"), ("cutlass.const_expr(self.lc_ones_quad)", "body")]
    thread0 = next(n for n in kernel_fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == "thread_idx == 0")
    assert all(s in list(ast.walk(thread0)) for s in stores)
    following = kernel_fn.body[kernel_fn.body.index(thread0) + 1]
    assert ast.unparse(following) == "cute.arch.barrier()"


def test_the_v_drain_reads_each_fragment_at_static_offsets(tree):
    """Under channel-wise V the drain prefetch locates the page's scale rows once by the widened
    page, moves the origin to each fragment's first scale, and reads s and z of both 64-channel
    halves at static offsets, for every fragment, and the drain takes each of the four from the
    register it was loaded into. Each z sits D past its s, which is checked at D = 128 and 256. The
    high half sits 64 channels on, the width of one PV mma's output and of the kernel's low output
    tile, which is half of D only at D = 128. The drain runs only under channel-wise V, which the
    constructor refuses at D = 256 (`quant_layout.sm90_wide_refusal`), so that offset is checked
    at D = 128 alone."""
    consumer = _method(tree, "_consumer_wg")
    # the guard is channel-wise V, whatever K is
    (v_ch,) = _binds_of(consumer, "V_CH")
    for k_axis, v_axis in itertools.product(("token", "channel"), repeat=2):
        self = types.SimpleNamespace(k_axis=k_axis, v_axis=v_axis)
        assert bool(_eval(v_ch.value, self=self)) == (v_axis == "channel")
    # the page is the block table's entry for this request and this tile, read once
    assert _only_binding(consumer, "v_page_pf") == "v_page_pf = mPT[Int64(batch_idx), global_tile_idx]"
    assert _only_binding(consumer, "gVDrainPage") == "gVDrainPage = mScaleRaw[Int64(v_page_pf), None, None]"
    assert _only_binding(consumer, "gVDrainFrag") == \
        "gVDrainFrag = cute.domain_offset((kv_head_idx, self.qoff['v_ch'] + dl_p), gVDrainPage)"
    assert not _subscripts(consumer, "mScaleRaw")[1:]
    # the fragment's origin is its own channel row: the first coordinate of its C fragment
    assert _only_binding(consumer, "dl_p") == "dl_p, _ = tPvCoord[j]"
    drain_loop = next(n for n in ast.walk(consumer) if isinstance(n, ast.For)
                      and _binds_of(consumer, "gVDrainFrag")[0] in n.body)
    assert _binds_of(consumer, "dl_p")[0] in drain_loop.body and ast.unparse(drain_loop.target) == "j"
    # every fragment of the C layout is prefetched and drained: both loops over j run PG // 2
    # fragments, and the register tensor holds four scales for each
    (frag_count,) = _binds_of(consumer, "accum_fragment_size")
    fragments = _eval(frag_count.value, PG=PG)
    use_loop = next(n for n in ast.walk(consumer) if isinstance(n, ast.For)
                    and any(isinstance(u, ast.Assign) and ast.unparse(u.targets[0]) == "s_lo"
                            and ast.unparse(u.value).startswith("rVDrain[") for u in n.body))
    for loop in (drain_loop, use_loop):
        assert list(_range_of(loop.iter, accum_fragment_size=fragments)) == list(range(PG // 2))
    (alloc,) = _binds_of(consumer, "rVDrain")
    assert _eval(alloc.value.args[0], accum_fragment_size=fragments) == 4 * (PG // 2)
    assert _position(_binds_of(consumer, "dl_p")[0]) < _position(_binds_of(consumer, "gVDrainFrag")[0])
    reads = sorted((n for n in ast.walk(consumer) if isinstance(n, ast.Assign)
                    and any(ast.unparse(t).startswith("rVDrain[") for t in n.targets)
                    and "gVDrainFrag" in ast.unparse(n.value)), key=_position)
    uses = sorted((n for n in ast.walk(consumer) if isinstance(n, ast.Assign)
                   and ast.unparse(n.value).startswith("rVDrain[")), key=_position)
    assert len(reads) == 4 and sorted(ast.unparse(u.targets[0]) for u in uses) == ["s_hi", "s_lo", "z_hi", "z_lo"]
    assert all(u in use_loop.body for u in uses)
    # the scale each name must hold: s of the low 64 channels at 0 and its z D further on, at both
    # head dimensions; the high half's pair 64 channels on, at D = 128 only
    want = {"s_lo": 0, "z_lo": "D", "s_hi": 64, "z_hi": "D + 64"}
    for D, j in itertools.product((128, 256), range(PG // 2)):
        held = {}
        for r in reads:
            (target,) = r.targets
            assert isinstance(target, ast.Subscript) and ast.unparse(target.value) == "rVDrain"
            conv = r.value      # gVDrainFrag[(row, offset)].to(Float32)
            assert conv.func.attr == "to" and [ast.unparse(a) for a in conv.args] == ["Float32"]
            assert ast.unparse(conv.func.value.value) == "gVDrainFrag"
            row, offset = conv.func.value.slice.elts
            assert _eval(row) == 0
            held[_eval(target.slice, **{drain_loop.target.id: j})] = _eval(offset, D=D)
        assert sorted(held) == list(range(4 * j, 4 * j + 4))
        got = {}
        for u in uses:
            assert ast.unparse(u.value.value) == "rVDrain"
            # evaluated at the use loop's own target, so a body indexing by any other name fails
            got[u.targets[0].id] = held[_eval(u.value.slice, **{use_loop.target.id: j})]
        assert got["s_lo"] == 0, (D, j, got)
        assert got["z_lo"] - got["s_lo"] == D and got["z_hi"] - got["s_hi"] == D, (D, j, got)
        if D == 128:
            assert got == {name: _eval(ast.parse(str(w), mode="eval").body, D=D) for name, w in want.items()}, (j, got)
    # the loads, the page view and the drain's reads are compiled exactly under channel-wise V,
    # their compile-time guards evaluated for every K and V axis
    cutlass_ns = types.SimpleNamespace(const_expr=lambda x: x)
    for k_axis, v_axis in itertools.product(("token", "channel"), repeat=2):
        self = types.SimpleNamespace(k_axis=k_axis, v_axis=v_axis, head_dim=128)
        env = {"self": self, "cutlass": cutlass_ns, "V_CH": _eval(v_ch.value, self=self)}
        for node in reads + uses + [_binds_of(consumer, "gVDrainPage")[0]]:
            guards = _enclosing_const_expr_tests(consumer, node)
            assert guards
            compiled = all(bool(_eval(ast.parse(test, mode="eval").body, **env)) == (branch == "body")
                           for test, branch in guards)
            assert compiled == (v_axis == "channel"), (ast.unparse(node), k_axis, v_axis)
    # loaded and drained in the same tile, both when that tile ends its V group: the one runtime
    # condition of each is the tile's own `v_group_ends`, and the loads come first
    tile_loop = next(n for n in ast.walk(consumer) if isinstance(n, ast.While)
                     and ast.unparse(n.test) == "wg_tile_idx < wg_tile_count")
    assert len(_binds_of(consumer, "v_group_ends")) == 1
    assert _binds_of(consumer, "v_group_ends")[0] in list(ast.walk(tile_loop))
    for node in reads + uses:
        assert node in list(ast.walk(tile_loop))
        inner = [g for g in _enclosing_ifs(tile_loop, node) if not g[0].startswith("cutlass.const_expr(")]
        assert inner == [("v_group_ends", "body")], (ast.unparse(node), inner)
    assert max(map(_position, reads)) < min(map(_position, uses))


def test_the_builder_stores_through_the_split_and_reads_the_staged_query(tree):
    """The builder's only plane stores go through the flat views at
    `plane_elem + (builder_lane ^ lane_xor)`, inside the group loop, with the split taken once
    from the plane's own layout and the allocated alignment. The query is read from global
    memory only by the staging, which runs for every quarter before the group loop, and each
    build loads a quarter's words before the head loop that uses them. The scale prefetch
    stays in registers, sized and looped by D // 32."""
    builder = _method(tree, "_build_q_planes")
    for D in HEAD_DIMS:
        assert _eval(_binds_of(builder, "NQ")[0].value, D=D) == D // 32
        assert _eval(_binds_of(builder, "rPfScale")[0].value.args[0], NQ=D // 32) == 2 * (D // 32)
    assert len(_binds_of(builder, "NQ")) == len(_binds_of(builder, "rPfScale")) == 1
    assert ast.unparse(_binds_of(builder, "rPfScale")[0].value.func) == "cute.make_rmem_tensor"
    assert _only_binding(builder, "QW") == "QW = _build_q_layout(HPK, D)[0][0]"
    assert _only_binding(builder, "plane_base_alignment") == \
        "plane_base_alignment = min(sQ.iterator.alignment, sQPlane1.iterator.alignment)"
    assert _only_binding(builder, "plane_stores") == \
        "plane_stores = self._plane_store_offsets(sQ_layout, plane_base_alignment)"
    for name, plane in (("sQFlat", "sQ"), ("sQPlane1Flat", "sQPlane1")):
        assert _only_binding(builder, name) == (
            f"{name} = cute.make_tensor(cute.recast_ptr({plane}.iterator, dtype=BF16), "
            f"cute.make_layout(cute.cosize(sQ_layout.outer)))")
    for name in ("sQ", "sQPlane1"):
        assert not _subscripts(builder, name), name
    flat = _subscripts(builder, "sQFlat") + _subscripts(builder, "sQPlane1Flat")
    assert sorted(ast.unparse(s.slice) for s in flat) == ["plane_elem + (builder_lane ^ lane_xor)"] * 2
    assert all(isinstance(s.ctx, ast.Store) for s in flat)
    for name in ("plane_elem", "lane_xor"):
        assert _only_binding(builder, name) == "plane_elem, lane_xor = plane_stores[h, qtr]"
    group_loop = next(n for n in ast.walk(builder) if isinstance(n, ast.While) and ast.unparse(n.test) == "g <= g1")
    assert all(s in list(ast.walk(group_loop)) for s in flat)
    # the staging reads channel lane + 32 qtr (which heads each word packs, the staged-words test
    # interprets); the global query is read nowhere else
    (qd,) = _binds_of(builder, "qd")
    for D in HEAD_DIMS:
        for lane, qtr in itertools.product(range(32), range(D // 32)):
            assert _eval(qd.value, builder_lane=lane, qtr=qtr) == lane + 32 * qtr
    assert len(_binds_of(builder, "q_lo")) == len(_binds_of(builder, "q_hi")) == 1
    q_reads = _subscripts(builder, "gQPlanes")
    assert len(q_reads) == 2 and not any(r in list(ast.walk(group_loop)) for r in q_reads)
    staged = sorted(_subscripts(builder, "sBuildQ"), key=_position)
    assert [ast.unparse(s) for s in staged] == ["sBuildQ[None, builder_lane, qtr, builder_plane]"] * 2
    for name in ("sBuildQStore", "sBuildQLoad"):
        assert _only_binding(builder, name) == f"{name} = sBuildQ[None, builder_lane, qtr, builder_plane]"
    store_copy, load_copy = [c for c in _calls_attr(builder, "cute.autovec_copy")]
    assert ast.unparse(store_copy) == \
        "cute.autovec_copy(rQStage, cute.make_tensor(sBuildQStore.iterator.align(4 * QW), sBuildQStore.layout))"
    assert ast.unparse(load_copy) == \
        "cute.autovec_copy(cute.make_tensor(sBuildQLoad.iterator.align(4 * QW), sBuildQLoad.layout), rQWords)"
    staging_loop = next(n for n in builder.body[-1].body if isinstance(n, ast.For)
                        and store_copy in list(ast.walk(n)))
    assert ast.unparse(staging_loop.target) == "qtr"
    assert _position(staging_loop) < _position(group_loop) and store_copy not in list(ast.walk(group_loop))
    build_qtr = next(n for n in ast.walk(group_loop) if isinstance(n, ast.For) and ast.unparse(n.target) == "qtr"
                     and load_copy in list(ast.walk(n)))
    head_loop = next(n for n in build_qtr.body if isinstance(n, ast.For) and ast.unparse(n.target) == "h")
    assert _position(load_copy) < _position(head_loop)
    # which word and half each head reads, the staged-words test interprets; here, that both are
    # bound once, inside the head loop
    for name in ("w_q", "qv"):
        binds = _binds_of(builder, name)
        assert len(binds) == 1 and binds[0] in list(ast.walk(head_loop)), name
    assert "Q_HOIST" not in {n.id for n in ast.walk(builder) if isinstance(n, ast.Name)}
    # every quarter loop (staging, the two prefetches, the build) runs the D // 32 quarters
    loops = [n for n in ast.walk(builder) if isinstance(n, ast.For) and ast.unparse(n.target) == "qtr"]
    assert len(loops) == 4 and staging_loop in loops
    for D in HEAD_DIMS:
        for loop in loops:
            assert list(_range_of(loop.iter, NQ=D // 32)) == list(range(D // 32))


#: The three packed-bf16 helpers the builder's staging and read-back use, pinned whole: the model
#: below gives them these semantics, so an edit to any of them must be matched by an edit to it.
_HELPER_ASM = {
    "_cvt2_bf16x2": ("llvm.inline_asm(T.i32(), [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)], "
                     "'cvt.rn.bf16x2.f32 $0, $1, $2;', '=r,f,f', has_side_effects=False, is_align_stack=False, "
                     "asm_dialect=llvm.AsmDialect.AD_ATT)", "return Int32(rst)"),
    "_bf16x2_lo_f32": ("llvm.inline_asm(T.f32(), [Int32(w).ir_value(loc=loc, ip=ip)], "
                       "'{\\n.reg .b32 t;\\nshl.b32 t, $1, 16;\\nmov.b32 $0, t;\\n}\\n', '=f,r', has_side_effects=False, "
                       "is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)", "return Float32(rst)"),
    "_bf16x2_hi_f32": ("llvm.inline_asm(T.f32(), [Int32(w).ir_value(loc=loc, ip=ip)], "
                       "'{\\n.reg .b32 t;\\nand.b32 t, $1, 0xFFFF0000;\\nmov.b32 $0, t;\\n}\\n', '=f,r', has_side_effects=False, "
                       "is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)", "return Float32(rst)"),
}


def _helper_semantics(tree):
    """What the three packed-bf16 helpers do, once their inline PTX is the pinned text:
    `cvt.rn.bf16x2.f32 d, a, b` puts a (the `hi` parameter) in the upper half of d and b (`lo`) in
    the lower; the low extract shifts the word left by 16 and the high one masks its upper half,
    each into an f32 whose bits are the result."""
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name, (asm, ret) in _HELPER_ASM.items():
        body = [st for st in fns[name].body if not isinstance(st, ast.Expr)]
        assert [ast.unparse(st) for st in body] == [f"rst = {asm}", ret], name
    assert [a.arg for a in fns["_cvt2_bf16x2"].args.args[:2]] == ["hi", "lo"]

    def pack(hi, lo):
        return (hi << 16) | lo

    return ["hi", "lo"], pack, {"_bf16x2_lo_f32": lambda word: (word << 16) & 0xFFFFFFFF,
                                "_bf16x2_hi_f32": lambda word: word & 0xFFFF0000}


def test_the_staged_words_give_every_head_its_own_query(tree, kernel):
    """Interpreting the builder's own staging loop and its per-build read-back, with the packing
    helpers' semantics read from their inline PTX and operand wiring, at D = 128 and 256 and every
    ratio: the staging reads each real head of its kv head's head-tile exactly once and no other
    query row, an odd ratio's last word carries a zero high half, every build's head loop visits
    exactly the real heads, and head h reads back exactly its own bf16 query bits, whatever
    padding the word count adds."""
    params, pack, extract = _helper_semantics(tree)
    builder = _method(tree, "_build_q_planes")
    group_loop = next(n for n in ast.walk(builder) if isinstance(n, ast.While) and ast.unparse(n.test) == "g <= g1")
    (word_loop,) = [n for n in ast.walk(builder) if isinstance(n, ast.For) and ast.unparse(n.target) == "w"]
    assert word_loop not in list(ast.walk(group_loop))
    # the packed words reach shared memory only after the loop that packs them, from the tensor
    # it packs into, and both register tensors hold QW words
    staging_qtr = next(n for n in builder.body[-1].body if isinstance(n, ast.For) and word_loop in n.body)
    (store_copy,) = [c for c in _calls_attr(staging_qtr, "cute.autovec_copy")]
    assert ast.unparse(store_copy.args[0]) == "rQStage" and _position(word_loop) < _position(store_copy)
    assert _binds_of(staging_qtr, "sBuildQStore") and _position(word_loop) < _position(_binds_of(staging_qtr, "sBuildQStore")[0])
    for name in ("rQStage", "rQWords"):
        (alloc,) = [b for b in _binds_of(builder, name)]
        assert ast.unparse(alloc.value.func) == "cute.make_rmem_tensor" and ast.unparse(alloc.value.args[1]) == "Int32"
    (guard,) = [n for n in word_loop.body if isinstance(n, ast.If)]
    assert ast.unparse(guard.test).startswith("cutlass.const_expr(")
    q_lo, q_hi, packed = guard.body
    assert [ast.unparse(t) for t in (q_lo.targets[0], q_hi.targets[0], packed.targets[0])] == ["q_lo", "q_hi", "rQStage[w]"]
    (pad,) = guard.orelse
    assert ast.unparse(pad) == "rQStage[w] = Int32(0)"
    call = packed.value
    assert ast.unparse(call.func) == "_cvt2_bf16x2"
    # the packing's operands by parameter, passed positionally or by keyword
    operands = _bind_helper(tree, call)
    assert sorted(operands) == sorted(params)
    head_loop = next(n for n in ast.walk(group_loop) if isinstance(n, ast.For) and ast.unparse(n.target) == "h"
                     and _binds_of(n, "w_q"))
    (w_q,) = _binds_of(head_loop, "w_q")
    (qv,) = _binds_of(head_loop, "qv")

    def row_of(subscript, **names):   # gQPlanes[(<head>, <channel>)].to(Float32): the element it reads
        # the bf16 query value widened to f32, which the packing's cvt.rn.bf16x2 rounds back exactly
        assert subscript.func.attr == "to" and [ast.unparse(a) for a in subscript.args] == ["Float32"]
        assert ast.unparse(subscript.func.value.value) == "gQPlanes"
        head, channel = subscript.func.value.slice.elts
        return _eval(head, **names), _eval(channel, **names)

    # the staged and the read-back register tensors are touched only by the staging loop, the
    # two copies and the head loop's word read
    parents = {child: node for node in ast.walk(builder) for child in ast.iter_child_nodes(node)}
    (store_copy,) = [c for c in _calls_attr(builder, "cute.autovec_copy") if ast.unparse(c.args[0]) == "rQStage"]
    (load_copy,) = [c for c in _calls_attr(builder, "cute.autovec_copy") if ast.unparse(c.args[-1]) == "rQWords"]
    # every lane stages its words and every build loads them: no runtime condition but the
    # builder's own (the tile count, around the group loop)
    k_guard = ("cutlass.const_expr(self.k_axis == 'channel')", "body")
    assert _enclosing_ifs(builder, store_copy) == [k_guard]
    assert _enclosing_ifs(builder, load_copy) == [k_guard, ("tile_count > 0", "body")]
    for name, allowed in (("rQStage", {id(store_copy)}), ("rQWords", {id(load_copy)})):
        for n in (n for n in ast.walk(builder) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)):
            parent = parents[n]
            ok = id(parent) in allowed or (
                isinstance(parent, ast.Subscript) and (
                    (name == "rQStage" and isinstance(parent.ctx, ast.Store) and parents[parent] in (packed, pad)) or
                    (name == "rQWords" and isinstance(parent.ctx, ast.Load) and parents[parent] is w_q)))
            assert ok, (name, ast.unparse(parent))

    (hpk,) = _binds_of(builder, "HPK")
    # the query rows are this CTA's request's: the builder's view of them is the request's slice,
    # by the request coordinate the kernel takes from its block index and passes to the builder
    # (the eager builds' block index, or the graph split build's slot item over the KV heads)
    assert sorted(ast.unparse(b) for b in _binds_of(_method(tree, "kernel"), "batch_idx")) == [
        "batch_idx = item // kv_head_count",
        "split_idx, q_group_idx, batch_idx = cute.arch.block_idx()"]
    (gq,) = _binds_of(builder, "gQPlanes")
    assert ast.unparse(gq.value.value) == "mQ" and len(gq.value.slice.elts) == 3
    assert [ast.unparse(e) for e in gq.value.slice.elts[1:]] == ["None", "None"]
    # the builder's lanes and planes, from its own bindings: IO warps 10 and 11, each lane its own
    (lane_bind,) = _binds_of(builder, "builder_lane")
    (plane_bind,) = _binds_of(builder, "builder_plane")
    threads = range(320, 384)
    assert {(_eval(plane_bind.value, warp_idx=t // 32), _eval(lane_bind.value, thread_idx=t)) for t in threads} == \
        {(p, l) for p in range(2) for l in range(32)}
    for head_dim in HEAD_DIMS:
        quarters = head_dim // 32
        # distinct bf16 bit patterns per (request, query row, channel)
        q = {(b, r, c): 0x3F80 + (7 * r + 131 * c + 977 * b) % 0x3F00
             for b in range(6) for r in range(8 * 4) for c in range(head_dim)}
        for heads, kv_head, batch in itertools.product(RATIOS, (0, 1, 3), (0, 5)):
            # the builder's own head count, from the constructor's per-tile head count
            assert _eval(hpk.value, self=types.SimpleNamespace(heads_per_tile=heads)) == heads
            request = _eval(gq.value.slice.elts[0], batch_idx=batch, Int64=int)
            real_rows = [kv_head * heads + h for h in range(heads)]   # this kv head's query rows
            qw = kernel["_build_q_layout"](heads, head_dim)[0][0]
            for name in ("rQStage", "rQWords"):
                assert _eval(_binds_of(builder, name)[0].value.args[0], QW=qw) == qw
            # the first two quarters, a middle one and the last, so that a quarter taken modulo
            # D = 128's four fails at D = 256
            for lane, qtr in ((0, 0), (5, 1), (17, quarters // 2), (31, quarters - 1)):
                qd_value = _eval(_binds_of(builder, "qd")[0].value, builder_lane=lane, **{staging_qtr.target.id: qtr})
                words, rows_read = {}, []
                for w in _range_of(word_loop.iter, QW=qw):
                    env = {"w": w, "HPK": heads, "kv_head_idx": kv_head, "q_group_idx": kv_head, "heads_per_kv": heads, "qd": qd_value}
                    if _eval(guard.test.args[0], **env):
                        hi_expr = q_hi.value
                        assert ast.unparse(hi_expr.orelse) == "Float32(0.0)"
                        read = [row_of(q_lo.value, **env)]
                        if _eval(hi_expr.test, **env):
                            read.append(row_of(hi_expr.body, **env))
                        # every row the staging reads is one of this kv head's real heads, so no
                        # read reaches the next kv head's rows or past the request's HQ rows
                        for row, _ in read:
                            assert row in real_rows, (head_dim, heads, kv_head, w, row)
                        rows_read += [row for row, _ in read]
                        lo = q[(request,) + read[0]]
                        hi = q[(request,) + read[1]] if len(read) == 2 else 0
                        named = {"q_lo": lo, "q_hi": hi}
                        words[w] = pack(**{p: named[src] for p, src in operands.items()})
                    else:
                        words[w] = 0
                assert sorted(words) == list(range(qw))
                # each real head is staged exactly once, and nothing else is read
                assert sorted(rows_read) == real_rows, (head_dim, heads, kv_head)
                if heads % 2:
                    # an odd ratio's last word pads its high half with zero, the Float32(0.0) branch
                    assert words[heads // 2] >> 16 == 0, (head_dim, heads)
                visited = list(_range_of(head_loop.iter, HPK=heads))
                assert visited == list(range(heads))
                for h in visited:
                    assert ast.unparse(w_q.value.value) == "rQWords"   # the words this build loaded
                    at = {head_loop.target.id: h}                   # the head loop's own target
                    word = words[_eval(w_q.value.slice, **at)]
                    chosen = qv.value.body if _eval(qv.value.test, **at) else qv.value.orelse
                    (arg,) = _bind_helper(tree, chosen).values()   # the extract's operand, evaluated
                    got = extract[ast.unparse(chosen.func)](_eval(ast.parse(arg, mode="eval").body, w_q=word))
                    assert got == q[(batch, kv_head * heads + h, lane + 32 * qtr)] << 16, \
                        (head_dim, heads, kv_head, batch, h)


def test_the_builder_call_passes_the_staging_buffer_and_the_plane_layout(tree):
    kernel_fn = _method(tree, "kernel")
    (call,) = [c for c in ast.walk(kernel_fn) if isinstance(c, ast.Call) and ast.unparse(c.func) == "self._build_q_planes"]
    args = _args_by_param(tree, call, "_build_q_planes")
    assert all(arg == param for param, arg in args.items()), args
    assert {"sBuildQ", "sQ_layout", "sQ", "sQPlane1"} <= set(args)


def test_swizzle_of_reads_the_layouts_swizzle(kernel):
    swizzle_of = kernel["_swizzle_of"]
    for bits, base, shift in ((3, 4, 3), (2, 4, 3), (1, 1, 2)):
        layout = types.SimpleNamespace(inner=types.SimpleNamespace(num_bits=bits, num_base=base, num_shift=shift))
        assert swizzle_of(layout) == (bits, base, shift)


def test_the_kernel_builds_the_modelled_plane_layout(tree):
    """`_plane_layout` models `make_smem_layout(BF16, ROW_MAJOR, (PG, D))`; the kernel builds
    exactly that layout for the planes, so a different major mode or tile fails here."""
    call = _method(tree, "_launch")      # the launch `__call__` and `call_with_lse` share
    assert _only_binding(call, "sQ_layout") == \
        "sQ_layout = sm90_utils.make_smem_layout(BF16, LayoutEnum.ROW_MAJOR, (PG, D))"


def test_the_plane_split_reads_the_plane_layout_and_its_swizzle(tree):
    method = _method(tree, "_plane_store_offsets")
    (ret,) = [n for n in ast.walk(method) if isinstance(n, ast.Return)]
    assert ast.unparse(ret.value) == (
        "_plane_store_split(lambda h, d: cute.crd2idx((h, d), sQ_layout.outer), "
        "self.heads_per_tile, self.head_dim // 32, _swizzle_of(sQ_layout), BF16.width // 8, alignment)")


def test_the_kernel_allocates_the_builder_buffers_under_channel_wise_k_only(tree, kernel):
    """The two planes are aligned to the swizzle's period under channel-wise K and at 128
    bytes otherwise, so every token-wise-K configuration keeps its shared-memory map; sBuildQ
    is allocated from its helper under channel-wise K only; and both new buffers are
    allocated after every buffer that existed before them."""
    kernel_fn = _method(tree, "kernel")
    (bind,) = _binds_of(kernel_fn, "plane_alignment")
    sw = types.SimpleNamespace(inner=types.SimpleNamespace(num_bits=3, num_base=4, num_shift=3))
    for k_axis, expected in (("channel", 1024), ("token", 128)):
        self = type("Kernel", (), dict(k_axis=k_axis))()
        assert _eval(bind.value, self=self, sQ_layout=sw, _swizzle_period=kernel["_swizzle_period"],
                     _swizzle_of=kernel["_swizzle_of"]) == expected
    planes = [a for name in ("sQ", "sQPlane1") for a in _binds_of(kernel_fn, name)
              if "allocate_tensor" in ast.unparse(a)]
    assert [ast.unparse(a) for a in planes] == [
        "sQ = smem.allocate_tensor(BF16, sQ_layout.outer, byte_alignment=plane_alignment, swizzle=sQ_layout.inner)",
        "sQPlane1 = smem.allocate_tensor(BF16, sQ_layout.outer, byte_alignment=plane_alignment, swizzle=sQ_layout.inner)"]
    # the second plane exists under channel-wise K only; otherwise it is sQ itself (its guards
    # evaluated for every K and V axis)
    (alias,) = [a for a in _binds_of(kernel_fn, "sQPlane1") if a is not planes[1]]
    assert ast.unparse(alias) == "sQPlane1 = sQ"
    cutlass_ns = types.SimpleNamespace(const_expr=lambda x: x)
    for k_axis, v_axis in itertools.product(("token", "channel"), repeat=2):
        self = types.SimpleNamespace(k_axis=k_axis, v_axis=v_axis)
        reached = {id(node): all(bool(_eval(ast.parse(test, mode="eval").body, self=self, cutlass=cutlass_ns)) == (branch == "body")
                                 for test, branch in _enclosing_ifs(kernel_fn, node))
                   for node in (planes[1], alias)}
        assert reached == {id(planes[1]): k_axis == "channel", id(alias): k_axis != "channel"}, (k_axis, v_axis)
    for name in ("build_q_shape", "build_q_stride"):
        assert _only_binding(kernel_fn, name) == \
            "build_q_shape, build_q_stride = _build_q_layout(self.heads_per_tile, D)"
    allocs = _binds_of(kernel_fn, "sBuildQ")
    assert [ast.unparse(a) for a in allocs] == [
        "sBuildQ = sQRowSum",
        "sBuildQ = smem.allocate_tensor(Int32, cute.make_layout(build_q_shape, stride=build_q_stride), byte_alignment=16)"]
    assert _enclosing_const_expr_tests(kernel_fn, allocs[1]) == [("cutlass.const_expr(self.k_axis == 'channel')", "body")]
    order = [n for n in kernel_fn.body if "smem.allocate" in ast.unparse(n)]
    last_two = [ast.unparse(n.test) for n in order[-2:]]
    assert last_two == ["cutlass.const_expr(self.k_axis == 'channel')", "cutlass.const_expr(self.lc_ones_quad)"]


def test_the_builder_is_compiled_only_under_channel_wise_k(tree):
    """Everything the builder change touches sits under the builder's own channel-K guard, so
    a token-wise-K configuration compiles none of it."""
    builder = _method(tree, "_build_q_planes")
    guard = [n for n in builder.body if isinstance(n, ast.If)]
    assert len(guard) == 1 and ast.unparse(guard[0].test) == "cutlass.const_expr(self.k_axis == 'channel')"
    assert not guard[0].orelse
    for name in ("plane_stores", "sQFlat", "sQPlane1Flat", "rQStage", "rQWords"):
        binds = _binds_of(builder, name)
        assert binds and all(b in list(ast.walk(guard[0])) for b in binds), name
