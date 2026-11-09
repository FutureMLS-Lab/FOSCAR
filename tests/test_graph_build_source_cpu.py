#!/usr/bin/env python3
# Copyright (c) 2026 Donglin Zhuang
# SPDX-License-Identifier: MIT
"""Source rules of the graph builds (CUDA-graph mode) that no numeric gate can see.

1. The Blackwell split reduction reads no partial slot at or past the request's split count
   (THE RULE, below).
2. Its dependent-launch order (THE ORDER): the request's split count is read before the
   reduction's `griddepcontrol.wait`; in the one-piece path and in the merge the wait is a
   statement of the path's own body, at its top level (under no `if`, loop or `with`, so that
   every thread runs it); and in `reduction_kernel` and `reduction_merge` no partial is read
   anywhere but after one of those two waits, in the body it heads: a `.load()`, an element taken
   as a value, a `cute.copy` or `cute.autovec_copy` of one, and in the merge every such copy (its
   cp.async). A partial is a name of the reduction's partial tensors or views (`READ_PREFIXES`),
   or a name bound by assignment to an expression that mentions one anywhere (one level of
   binding, no dataflow beyond it), except that a name bound to a pure shape or stride
   expression is metadata; building a view reads nothing, and an element of a shape or stride,
   at any depth and through a metadata name, is an integer, not a read. A wait factored into a
   helper is refused, with a message that says so: the rule looks for the wait in the body. The
   decode triggers the dependent launch once its maxima and sums are stored, and stores its
   outputs on their own path, so a read placed before the wait, or reached by a thread that
   skipped a wait nested under a condition, races the decode.
3. Each graph decode build exits an idle CTA first (THE EXIT): on Hopper
   (`IntPingPongDecodeSm90.kernel`) and on Blackwell (`IntDecodeSm100._graph_work_entry`), the
   statement after the 16-byte entry load reads the item and the next one exits on the idle
   sentinel, and no barrier, shared-memory or tensor-memory allocation, register reallocation,
   descriptor prefetch or dependent-launch wait comes before it, in the kernel or, on
   Blackwell, in the entry helper's own statements before its exit (the helper itself; its
   callees are not searched). An idle CTA that exits after `alloc_tmem` holds tensor memory; one
   that waits first costs the whole grid a wait.

SCOPE. These rules are regression guards against plausible
edits to this source by a cooperating developer, not a proof: an AST rule cannot be complete, so
each closes the escapes reviews have found with the simple scope rule its text states.

WHY RULE 1 IS A SOURCE CHECK. The graph split build (CUDA-graph mode) allocates its partials for
the most pieces one request can get, S, and each request merges only its own k_b slots
(`GroupedQueryAttentionDecode.reduction_merge`, handed `nsplit = k_b`); slots k_b..S-1 hold
whatever an earlier step left there. A bound that reads S instead of k_b can load those slots
without changing any output: the register prefetch of the first 16 partial outputs, for one,
keeps its accumulation behind `j < nsplit`, so numeric checks on outputs and LSEs pass even
with the prefetch clamp at `splits - 1`. Such a load is out of bounds of the request's data, and a load
leaves nothing a numeric check sees. So the bounds are pinned where the kernel is traced from,
the source of `reduction_merge` and `reduction_kernel`, which the DSL traces into the kernel
statement for statement; this reads the source, not PTX, in which k_b and S are both run-time
registers that no text check can tell apart.

THE RULE.
- In `reduction_merge`, `splits` (the partials' first extent) is read exactly at the two
  places that size and place shared memory (`SPLITS_READS`), and `nsplit` (the count merged)
  exactly at `BOUNDS`, every loop bound, clamp and guard over the split axis, each once.
- `reduction_kernel` calls `reduction_merge` twice: with `splits` in the eager build
  (`split_counts is None`, where the two are equal) and with `k_b = split_counts[coord_b]` in
  the graph build, after its `k_b == 1` path, which reads slot 0 alone (`FAST_PATH_READS`).

The mutation tests apply each mutation to the source in memory and require its rule to reject
it, so no rule passes vacuously.

Run: pytest tests/test_graph_build_source_cpu.py
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src/foscar"
SOURCE = _SRC / "_sm100/gqa_decode.py"
HOPPER = _SRC / "_sm90/int_decode_pingpong.py"
BLACKWELL = _SRC / "_sm100/int_fold_decode.py"

#: Every split-axis bound of the merge, as `ast.unparse` writes it, each read once: the
#: cp.async guard and strided loop of the maxima and sums, the prefetch clamp, the tail staging,
#: the four-chain maximum and its remainder, the one-chain maximum, and both accumulation loops.
BOUNDS = (
    "tidx < nsplit",
    "cutlass.range(thr_per_blk + tidx, nsplit, thr_per_blk)",
    "cutlass.min(Int32(j), nsplit - 1)",
    "cutlass.range(o_pf, nsplit)",
    "cutlass.range(nsplit // 4, unroll=4)",
    "nsplit // 4 * 4",
    "r_c < nsplit",
    "r_c + 1 < nsplit",
    "r_c + 2 < nsplit",
    "cutlass.range(nsplit, unroll=8)",
    "j < nsplit",
    "cutlass.range(o_pf, nsplit, unroll=8)",
)
#: The two reads of the physical extent in the merge: the staged tail's place after the maxima
#: and sums, and its shape.
SPLITS_READS = (
    "smem_ptr + 2 * splits",
    "cute.make_layout((d_per_blk, splits), stride=(1, d_per_blk))",
)
#: What the graph build's one-piece path reads of the partials: slot 0, nothing else.
FAST_PATH_READS = ("gM_partial[0, 0]", "gL_partial[0, 0]", "tCgO_partial[None, 0]")
PARTIALS = ("gM_partial", "gL_partial", "tCgO_partial")
#: THE ORDER's partials: every name starting with one of these is the reduction's partial
#: maxima, sums or outputs, as its parameters, its per-CTA views and the merge's arguments.
READ_PREFIXES = PARTIALS + ("gO_partial", "o_partial_dhsb", "l_partial_hsb", "m_partial_hsb")
#: The calls that copy memory (THE ORDER): any other call whose name ends in `copy` builds an
#: atom or a tiled copy and reads nothing.
COPIES = ("cute.copy", "cute.autovec_copy")


def _method(tree, name, cls_name="GroupedQueryAttentionDecode"):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    return next(m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == name)


def _innermost(fn, name):
    """For each load of `name` in `fn`, the expression it bounds, unparsed: the nearest
    enclosing call or comparison, reached through arithmetic, tuples and keywords, or else the
    outermost arithmetic around it (an assignment's value)."""
    parents = {c: p for p in ast.walk(fn) for c in ast.iter_child_nodes(p)}
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
            cur = node
            while isinstance(parents.get(cur), (ast.BinOp, ast.UnaryOp, ast.Tuple, ast.keyword)):
                cur = parents[cur]
            if isinstance(parents.get(cur), (ast.Call, ast.Compare)):
                cur = parents[cur]
            out.append(ast.unparse(cur))
    return out


def check_rule(src):
    tree = ast.parse(src)
    merge = _method(tree, "reduction_merge")
    assert merge.args.args[-1].arg == "nsplit", "reduction_merge's last argument is nsplit"
    sites = _innermost(merge, "nsplit")
    for bound in BOUNDS:
        assert sites.count(bound) == bound.count("nsplit") and bound in sites, (
            f"reduction_merge must bound by `{bound}`; its nsplit reads are {sites}")
    assert len(sites) == sum(b.count("nsplit") for b in BOUNDS), (
        f"nsplit is read outside BOUNDS: {sites}")
    reads = _innermost(merge, "splits")
    assert sorted(reads) == sorted(SPLITS_READS), (
        f"reduction_merge may read splits only to size and place shared memory "
        f"({SPLITS_READS}); it reads it at {reads}")

    kern = _method(tree, "reduction_kernel")
    calls = [n for n in ast.walk(kern) if isinstance(n, ast.Call)
             and ast.unparse(n.func).endswith("reduction_merge")]
    assert [ast.unparse(c.args[-1]) for c in calls] == ["splits", "k_b"], (
        f"reduction_kernel merges `splits` slots (eager) and `k_b` (graph); found "
        f"{[ast.unparse(c) for c in calls]}")
    k_b = [n for n in ast.walk(kern) if isinstance(n, ast.Assign)
           and [ast.unparse(t) for t in n.targets] == ["k_b"]]
    assert [ast.unparse(a.value) for a in k_b] == ["split_counts[coord_b]"], (
        "k_b is the request's own split count, split_counts[coord_b]")
    fast = next(n for n in ast.walk(kern) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "k_b == 1")
    fast_reads = sorted({ast.unparse(n) for s in fast.body for n in ast.walk(s)
                         if isinstance(n, ast.Subscript) and ast.unparse(n.value) in PARTIALS})
    assert fast_reads == sorted(FAST_PATH_READS), (
        f"the one-piece path reads slot 0 only; it reads {fast_reads}")


def test_the_reduction_reads_only_its_own_slots():
    check_rule(SOURCE.read_text())


MUTATIONS = {
    "the prefetch clamp at the capacity": (
        "cutlass.min(Int32(j), nsplit - 1)", "cutlass.min(Int32(j), splits - 1)"),
    "the prefetch unclamped": ("cutlass.min(Int32(j), nsplit - 1)", "Int32(j)"),
    "the tail staging to the capacity": (
        "cutlass.range(o_pf, nsplit):", "cutlass.range(o_pf, splits):"),
    "the maxima and sums copied for every slot": ("if tidx < nsplit:", "if tidx < splits:"),
    "the strided copy of the maxima and sums to the capacity": (
        "cutlass.range(thr_per_blk + tidx, nsplit, thr_per_blk)",
        "cutlass.range(thr_per_blk + tidx, splits, thr_per_blk)"),
    "the four-chain maximum over the capacity": (
        "cutlass.range(nsplit // 4, unroll=4)", "cutlass.range(splits // 4, unroll=4)"),
    "the one-chain maximum over the capacity": (
        "cutlass.range(nsplit, unroll=8)", "cutlass.range(splits, unroll=8)"),
    "the accumulation over the capacity": (
        "cutlass.range(o_pf, nsplit, unroll=8)", "cutlass.range(o_pf, splits, unroll=8)"),
    "the graph merge handed the capacity": (
        "reduction_merge(*merge_args, k_b)", "reduction_merge(*merge_args, splits)"),
    "the one-piece path reads slot 1": ("max_final = gM_partial[0, 0]",
                                       "max_final = gM_partial[0, 1]"),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_the_rule_rejects_a_mutation(name):
    old, new = MUTATIONS[name]
    src = SOURCE.read_text()
    assert src.count(old) == 1, f"mutation {name!r}: `{old}` is not unique in the source"
    with pytest.raises(AssertionError):
        check_rule(src.replace(old, new))


# ---------------------------------------------------------------------------
# THE ORDER: the reduction's dependent-launch wait
# ---------------------------------------------------------------------------


def _calls(node, suffix):
    """The calls under `node` whose function, unparsed, ends with `suffix`, by line."""
    return sorted((n.lineno, n) for n in ast.walk(node)
                  if isinstance(n, ast.Call) and ast.unparse(n.func).endswith(suffix))


def _names_partial(node, aliases):
    """Whether `node` mentions a partial: a name starting with READ_PREFIXES, or an alias."""
    return any(isinstance(n, ast.Name) and (n.id.startswith(READ_PREFIXES) or n.id in aliases)
               for n in ast.walk(node))


def _aliases(fn):
    """`(partials, metadata)`: the names `fn` binds by plain assignment (a name or a tuple of
    names) to an expression that mentions a partial name anywhere. A name bound to a pure shape
    or stride expression (`_is_metadata`: `tCgO_partial.shape`, `tCgO_partial.stride[1]`,
    `tCgO_partial.shape[0][1]`) is METADATA, whose elements at any depth are integers; every
    other one (the partial itself, a subscript or view of it, a tensor rebuilt from its iterator
    and layout, anything else built from it) is a PARTIAL. One level: a name bound from an alias
    is not followed."""
    partials, metadata = set(), set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and _names_partial(n.value, frozenset()):
            names = {t.id for target in n.targets for t in ast.walk(target)
                     if isinstance(t, ast.Name)}
            (metadata if _is_metadata(n.value, frozenset()) else partials).update(names)
    return partials, metadata


def _is_metadata(node, metadata):
    """Whether `node`, below any number of subscripts, is a tensor's `.shape` or `.stride`, or a
    METADATA alias (`_aliases`): integers, not memory. Only the subscripts themselves are
    exempt; `_reads` still walks into them, so a read inside (`X[None, 0].load().shape[0]`) is
    counted."""
    while isinstance(node, ast.Subscript):
        node = node.value
    return ((isinstance(node, ast.Attribute) and node.attr in ("shape", "stride"))
            or (isinstance(node, ast.Name) and node.id in metadata))


def _is_view_index(index):
    """Whether a subscript's index makes a view (it holds a `None` mode or a slice) rather than
    naming one element."""
    parts = index.elts if isinstance(index, ast.Tuple) else [index]
    return any(isinstance(p, ast.Slice) or (isinstance(p, ast.Constant) and p.value is None)
               for p in parts)


def _reads(nodes, copies, aliases=frozenset(), metadata=frozenset()):
    """`(line, node)` of every READ of a partial under `nodes`: a `.load()` of one, an element of
    one taken as a value (a subscript with no `None` mode or slice), a `COPIES` call that names
    one, and, with `copies`, every `COPIES` call (the merge's cp.async, whatever it copies).
    Building a view of a partial reads no memory, so a view made before the wait is not a read
    (kernel review KB2), and neither is building a copy atom or a tiled copy (KB round 4, P4)."""
    out = []
    for root in nodes:
        for n in ast.walk(root):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "load" and _names_partial(n.func.value, aliases)):
                out.append((n.lineno, n))
            elif (isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Load)
                  and _names_partial(n.value, aliases | metadata)
                  and not _is_view_index(n.slice) and not _is_metadata(n.value, metadata)):
                out.append((n.lineno, n))
            elif (isinstance(n, ast.Call) and ast.unparse(n.func) in COPIES
                  and (copies or any(_names_partial(a, aliases) for a in n.args))):
                out.append((n.lineno, n))
    return sorted(out, key=lambda r: r[0])


def _reads_outside(fn, waits, copies, where):
    """The lines of `fn`'s partial reads (with its one-level aliases) that are not in the
    statements after one of `waits`, each `(body, index of its top-level wait)`."""
    allowed = {id(n) for body, at in waits for st in body[at + 1:] for n in ast.walk(st)}
    partials, metadata = _aliases(fn)
    bad = [line for line, node in _reads([fn], copies, partials, metadata)
           if id(node) not in allowed]
    assert not bad, (
        f"{where} reads a partial, or copies, outside the statements after its top-level "
        f"dependent-launch wait, at lines {bad}")


def _top_level_wait(body, where):
    """The index in `body` of its one top-level `griddepcontrol_wait()` statement; a wait nested
    under an `if`, a loop or a `with` of `body` is refused, because a thread that skips it reads
    the partials without waiting."""
    top = [i for i, st in enumerate(body) if isinstance(st, ast.Expr)
           and isinstance(st.value, ast.Call)
           and ast.unparse(st.value.func).endswith("griddepcontrol_wait")]
    nested = [line for st in body
              if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call))
              for line, _ in _calls(st, "griddepcontrol_wait")]
    assert len(top) == 1 and not nested, (
        f"{where} waits for the decode once, at the top level of its body, where every thread "
        f"runs it (a wait in a helper is not seen here); top-level waits at statements {top}, "
        f"nested waits at lines {nested}")
    return top[0]


def check_order(src):
    tree = ast.parse(src)
    kern = _method(tree, "reduction_kernel")
    k_b = next(n for n in ast.walk(kern) if isinstance(n, ast.Assign)
               and [ast.unparse(t) for t in n.targets] == ["k_b"])
    waits = _calls(kern, "griddepcontrol_wait")
    assert all(line > k_b.lineno for line, _ in waits), (
        "the graph reduction reads its split count k_b before any griddepcontrol.wait: the host "
        "wrote it before the replay, not the decode it depends on")
    fast = next(n for n in ast.walk(kern) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "k_b == 1")
    at = _top_level_wait(fast.body, "the one-piece path")
    assert _reads(fast.body[at:], copies=False), "the one-piece path reads its partials"
    _reads_outside(kern, [(fast.body, at)], False, "reduction_kernel")
    merge = _method(tree, "reduction_merge")
    at = _top_level_wait(merge.body, "reduction_merge")
    _reads_outside(merge, [(merge.body, at)], True, "reduction_merge")


def test_the_reduction_waits_for_the_decode_before_it_reads_a_partial():
    check_order(SOURCE.read_text())


# ---------------------------------------------------------------------------
# THE EXIT: each graph decode build exits an idle CTA first
# ---------------------------------------------------------------------------

#: The calls an idle CTA must not reach: barriers and their set-up, shared- and tensor-memory
#: allocation, register reallocation, descriptor prefetch and the dependent-launch wait. (The
#: shared-memory allocator's constructor, `SmemAllocator()`, is a trace-time object that emits
#: nothing, and Blackwell builds it before the entry.)
BEFORE_EXIT = ("barrier", "sync_threads", "arrive_and_wait", "mbarrier_init", "allocate_array",
               "allocate_tensor", "alloc_tmem", "setmaxregister_increase",
               "setmaxregister_decrease", "prefetch_descriptor", "griddepcontrol_wait")


def _sequence(body, load, item, exit_call):
    """The entry load (a statement starting with `load`, or any of a tuple of spellings), the
    item read and the exit, as three consecutive statements of `body`; their statements, or an
    AssertionError naming what is out of place."""
    text = [ast.unparse(st) for st in body]
    at = next((i for i, t in enumerate(text) if t.startswith(load)), None)
    assert at is not None, f"no entry load `{load}...`"
    assert text[at + 1:at + 3] == [item, exit_call], (
        f"the entry load must be followed by `{item}` and then `{exit_call}`; it is followed "
        f"by {text[at + 1:at + 3]}")
    return body[at + 2]


def check_exit(hopper_src, blackwell_src):
    # Hopper: in the kernel's graph branch, load, item, exit; and nothing of BEFORE_EXIT above.
    tree = ast.parse(hopper_src)
    kern = _method(tree, "kernel", "IntPingPongDecodeSm90")
    exits = _calls(tree, "_exit_if_idle")
    assert len(exits) == 1, "Hopper's graph build exits an idle CTA at exactly one place"
    branch = next(n for n in ast.walk(kern) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "cutlass.const_expr(mWork is None)")
    exit_st = _sequence(branch.orelse, "cute.autovec_copy(", "item = "
                        "cute.arch.make_warp_uniform(rWork[0])", "_exit_if_idle(item)")
    for name in BEFORE_EXIT:
        early = [line for line, _ in _calls(kern, name) if line < exit_st.lineno]
        assert not early, f"Hopper's {name} at line {early} comes before the idle CTA's exit"
    # Blackwell: the entry helper loads, reads the item and exits; the kernel calls it before
    # anything of BEFORE_EXIT.
    tree = ast.parse(blackwell_src)
    entry = _method(tree, "_graph_work_entry", "IntDecodeSm100")
    exits = _calls(tree, "_exit_if_negative")
    assert len(exits) == 1, "Blackwell's graph build exits an idle CTA at exactly one place"
    exit_st = _sequence(entry.body, ("cute.copy(", "cute.autovec_copy("),
                        "item = cute.arch.make_warp_uniform(rWork[0])", "_exit_if_negative(item)")
    prefix = entry.body[:entry.body.index(exit_st)]
    for name in BEFORE_EXIT:
        early = [line for st in prefix for line, _ in _calls(st, name)]
        assert not early, (f"Blackwell's {name} at line {early} comes before the idle CTA's exit "
                           f"in the entry helper")
    kern = _method(tree, "kernel", "IntDecodeSm100")
    sites = _calls(kern, "_graph_work_entry")
    assert len(sites) == 1, "the kernel reads its work entry at exactly one place"
    for name in BEFORE_EXIT:
        early = [line for line, _ in _calls(kern, name) if line < sites[0][0]]
        assert not early, f"Blackwell's {name} at line {early} comes before the work entry"


def test_each_graph_decode_build_exits_an_idle_cta_first():
    check_exit(HOPPER.read_text(), BLACKWELL.read_text())


#: The merge's dependent-launch wait and its register prefetch of the first partial outputs, as
#: the source spells them (anchors of the mutants and controls below).
MERGE_WAIT = ("        cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after PDL wait\n"
              "        cute.arch.griddepcontrol_wait()\n\n        if tidx < nsplit:\n")
#: A view of the partial outputs rebuilt from their iterator and layout.
REBUILT_VIEW = ("        outputs = cute.make_tensor(tCgO_partial.iterator.align(16), "
                "tCgO_partial.layout)\n")
PREFETCH = ("        o_pf = reduction_prefetch_splits if cutlass.const_expr(d % d_per_blk == 0) "
            "else 0\n        rO_pf = [\n            tCgO_partial[None, cutlass.min(Int32(j), "
            "nsplit - 1)].load()\n            for j in range(o_pf)\n        ]\n")
#: Ordering mutants, each a list of (file, old, new) edits that its rule must reject.
ORDER_MUTANTS = {
    "R2: the merge reads the partials with no dependent-launch wait": [(SOURCE,
        "        cute.arch.griddepcontrol_wait()\n\n        if tidx < nsplit:",
        "\n        if tidx < nsplit:")],
    "R3: the one-piece path loads slot 0 before the wait": [(SOURCE,
        "                cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after PDL "
        "wait\n                cute.arch.griddepcontrol_wait()\n                max_final = "
        "gM_partial[0, 0]\n                sum_0 = gL_partial[0, 0]\n",
        "                max_final = gM_partial[0, 0]\n                sum_0 = gL_partial[0, 0]\n"
        "                cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after PDL "
        "wait\n                cute.arch.griddepcontrol_wait()\n")],
    "R4: k_b read after the wait": [(SOURCE, "            k_b = split_counts[coord_b]\n",
        "            cute.arch.griddepcontrol_wait()\n            k_b = split_counts[coord_b]\n")],
    "R9: the merge's wait only under its first copy's condition": [(SOURCE,
        "        cute.arch.griddepcontrol_wait()\n\n        if tidx < nsplit:\n",
        "\n        if tidx < nsplit:\n            cute.arch.griddepcontrol_wait()\n")],
    "R10: the one-piece path waits on thread 0 only": [(SOURCE,
        "                cute.arch.griddepcontrol_wait()\n                max_final = "
        "gM_partial[0, 0]",
        "                if tidx == 0:\n                    cute.arch.griddepcontrol_wait()\n"
        "                max_final = gM_partial[0, 0]")],
    "R11: the merge's wait under a never-true guard": [(SOURCE,
        "        cute.arch.griddepcontrol_wait()\n\n        if tidx < nsplit:\n",
        "        if tidx < 0:\n            cute.arch.griddepcontrol_wait()\n\n"
        "        if tidx < nsplit:\n")],
    "R12: slot 0's maximum read in reduction_kernel before the one-piece path": [
        (SOURCE, "            k_b = split_counts[coord_b]\n            if k_b == 1:\n",
         "            k_b = split_counts[coord_b]\n            max_pre = gM_partial[0, 0]\n"
         "            if k_b == 1:\n"),
        (SOURCE, "                max_final = gM_partial[0, 0]\n",
         "                max_final = max_pre\n")],
    "R12b: slot 0's output loaded in reduction_kernel before the one-piece path": [
        (SOURCE, "            k_b = split_counts[coord_b]\n            if k_b == 1:\n",
         "            k_b = split_counts[coord_b]\n            rO_pre = tCgO_partial[None, 0]"
         ".load()\n            if k_b == 1:\n"),
        (SOURCE, "                    rO_0 = tCgO_partial[None, 0].load()\n",
         "                    rO_0 = rO_pre\n")],
    "R13: the one-piece path loads slot 0 through an alias before its wait": [(SOURCE,
        "                cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after PDL "
        "wait\n                cute.arch.griddepcontrol_wait()\n                max_final = "
        "gM_partial[0, 0]\n",
        "                slot0 = tCgO_partial[None, 0]\n                rO_early = slot0.load()\n"
        "                cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after PDL "
        "wait\n                cute.arch.griddepcontrol_wait()\n                max_final = "
        "gM_partial[0, 0]\n")],
    "the merge's prefetch through an alias, moved before the wait": [
        (SOURCE, PREFETCH, ""),
        (SOURCE, MERGE_WAIT, "        outputs = tCgO_partial\n"
         + PREFETCH.replace("tCgO_partial", "outputs") + MERGE_WAIT)],
    "the merge's prefetch through a rebuilt view, moved before the wait": [
        (SOURCE, PREFETCH, ""),
        (SOURCE, MERGE_WAIT, REBUILT_VIEW + PREFETCH.replace("tCgO_partial", "outputs")
         + MERGE_WAIT)],
    "a load inside a shape subscript, before the merge's wait": [(SOURCE, MERGE_WAIT,
        "        n_early = tCgO_partial[None, 0].load().shape[0]\n" + MERGE_WAIT)],
    "an element of a partial's iterator, before the merge's wait": [(SOURCE, MERGE_WAIT,
        "        first_early = tCgO_partial.iterator[0]\n" + MERGE_WAIT)],
    "the one-piece path's wait only where the d block is whole": [(SOURCE,
        "                cute.arch.griddepcontrol_wait()\n                max_final = "
        "gM_partial[0, 0]\n",
        "                if not_oob_d:\n                    cute.arch.griddepcontrol_wait()\n"
        "                max_final = gM_partial[0, 0]\n")],
}
#: Correct code the order rule must accept: a view of a partial built before
#: the merge's wait reads no memory.
ORDER_ACCEPTS = {
    "the merge's prefetch through a rebuilt view, after the wait": [
        (SOURCE, PREFETCH, PREFETCH.replace("tCgO_partial", "outputs")),
        (SOURCE, MERGE_WAIT, REBUILT_VIEW + MERGE_WAIT)],
    "an extent of a partial's shape taken before the merge's wait": [(SOURCE, MERGE_WAIT,
        "        n_slots = tCgO_partial.shape[1]\n" + MERGE_WAIT)],
    "an extent through a name bound to the shape, before the wait": [
        (SOURCE, MERGE_WAIT, "        partial_shape = tCgO_partial.shape\n"
         "        n_slots = partial_shape[1]\n" + MERGE_WAIT)],
    "a stride through a name bound to the stride, before the wait": [
        (SOURCE, MERGE_WAIT, "        partial_stride = tCgO_partial.stride\n"
         "        slot_step = partial_stride[1]\n" + MERGE_WAIT)],
    "a nested CuTe extent of a partial's shape, before the wait": [(SOURCE, MERGE_WAIT,
        "        n_slots = tCgO_partial.shape[0][1]\n" + MERGE_WAIT)],
    "the merge's prefetch through an alias, after the wait": [
        (SOURCE, PREFETCH, PREFETCH.replace("tCgO_partial", "outputs")),
        (SOURCE, MERGE_WAIT, "        outputs = tCgO_partial\n" + MERGE_WAIT)],
    "P4: a tiled copy built in the merge before the wait": [(SOURCE, MERGE_WAIT,
        "        tiled_probe = cute.make_tiled_copy(cute.make_copy_atom("
        "cute.nvgpu.CopyUniversalOp(), Float32), cute.make_ordered_layout((thr_per_blk, "
        "d_per_thr), order=(1, 0)), (d_per_blk,))\n" + MERGE_WAIT)],
    "O1: a view of a partial built before the merge's wait": [(SOURCE,
        "        cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after PDL wait\n"
        "        cute.arch.griddepcontrol_wait()\n\n        if tidx < nsplit:\n",
        "        gL_partial_first = gL_partial[None, 0]\n"
        "        cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after PDL wait\n"
        "        cute.arch.griddepcontrol_wait()\n\n        if tidx < nsplit:\n")],
}
EXIT_MUTANTS = {
    "H1b: Hopper's exit after the set-up barrier": [
        (HOPPER, "            _exit_if_idle(item)\n", ""),
        (HOPPER, "        cute.arch.barrier()\n\n        # per-request tile count from seq_lens",
         "        cute.arch.barrier()\n        if cutlass.const_expr(mWork is not None):\n"
         "            _exit_if_idle(item)\n\n        # per-request tile count from seq_lens")],
    "D1b: Blackwell's exit after alloc_tmem": [
        (BLACKWELL, "        _exit_if_negative(item)\n", ""),
        (BLACKWELL, "            coord_hb, kv_split_idx, kv_splits, graph_seqlen = "
         "self._graph_work_entry(\n                mWork, bidz)\n",
         "            coord_hb, kv_split_idx, kv_splits, graph_seqlen = "
         "self._graph_work_entry(\n                mWork, bidz)\n            graph_item = "
         "coord_hb\n"),
        (BLACKWELL, "            cute.arch.alloc_tmem(tmem_alloc_cols, tmem_ptr_smem_ptr)\n"
         "        init_warp += 1\n",
         "            cute.arch.alloc_tmem(tmem_alloc_cols, tmem_ptr_smem_ptr)\n"
         "        init_warp += 1\n        if cutlass.const_expr(graph_split):\n"
         "            _exit_if_negative(graph_item)\n")],
    "D2: Blackwell's entry loaded after the wait": [
        (BLACKWELL, "            coord_hb, kv_split_idx, kv_splits, graph_seqlen = "
         "self._graph_work_entry(\n",
         "            cute.arch.griddepcontrol_wait()\n            coord_hb, kv_split_idx, "
         "kv_splits, graph_seqlen = self._graph_work_entry(\n")],
    "Hopper's idle CTA does not exit": [(HOPPER, "            _exit_if_idle(item)\n", "")],
    "D5: a CTA-wide sync before Blackwell's work entry": [
        (BLACKWELL, "            coord_hb, kv_split_idx, kv_splits, graph_seqlen = "
         "self._graph_work_entry(\n",
         "            cute.arch.sync_threads()\n            coord_hb, kv_split_idx, kv_splits, "
         "graph_seqlen = self._graph_work_entry(\n")],
    "C2: Blackwell's entry helper waits before its load": [
        (BLACKWELL, "        rWork = cute.make_rmem_tensor(4, Int32)\n        cute.copy(\n",
         "        cute.arch.griddepcontrol_wait()\n        rWork = cute.make_rmem_tensor(4, "
         "Int32)\n        cute.copy(\n")],
}


def _mutated(edits):
    """The three sources with `edits` applied. An anchor that does not occur exactly once FAILS
    the test that asked for it (`pytest.fail`, outside any `pytest.raises`), so a mutant whose
    anchor went stale can never read as rejected by the rule (kernel review KB1)."""
    texts = {path: path.read_text() for path in (SOURCE, HOPPER, BLACKWELL)}
    for path, old, new in edits:
        if texts[path].count(old) != 1:
            pytest.fail(f"the anchor `{old[:60]}...` occurs {texts[path].count(old)} times in "
                        f"{path.name}; the mutant must be re-anchored")
        texts[path] = texts[path].replace(old, new)
    return texts


@pytest.mark.parametrize("name", sorted(ORDER_MUTANTS))
def test_the_order_rule_rejects_a_mutant(name):
    source = _mutated(ORDER_MUTANTS[name])[SOURCE]
    with pytest.raises(AssertionError):
        check_order(source)


@pytest.mark.parametrize("name", sorted(ORDER_ACCEPTS))
def test_the_order_rule_accepts_correct_code(name):
    source = _mutated(ORDER_ACCEPTS[name])[SOURCE]
    check_order(source)
    check_rule(source)


#: Correct code the exit rule must accept: Blackwell's entry load spelled as Hopper's is (P6).
EXIT_ACCEPTS = {
    "P6: Blackwell's entry load as cute.autovec_copy": [(BLACKWELL,
        "        cute.copy(\n            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Int32, "
        "num_bits_per_copy=128),\n            cute.make_tensor(mWork[slot, None].iterator.align("
        "16), cute.make_layout(4)),\n            rWork)\n",
        "        cute.autovec_copy(\n            cute.make_tensor(mWork[slot, None].iterator.align("
        "16), cute.make_layout(4)), rWork)\n")],
}


@pytest.mark.parametrize("name", sorted(EXIT_ACCEPTS))
def test_the_exit_rule_accepts_correct_code(name):
    texts = _mutated(EXIT_ACCEPTS[name])
    check_exit(texts[HOPPER], texts[BLACKWELL])


@pytest.mark.parametrize("name", sorted(EXIT_MUTANTS))
def test_the_exit_rule_rejects_a_mutant(name):
    texts = _mutated(EXIT_MUTANTS[name])
    with pytest.raises(AssertionError):
        check_exit(texts[HOPPER], texts[BLACKWELL])

