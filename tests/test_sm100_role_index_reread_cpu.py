# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""The Blackwell kernel's persistent item loop chooses every warp role from one fresh read.

WHY. In a persistent build each CTA loops over work items, and at the top of each item every
warp picks its role by comparing its warp index with the role's warp number. Two rules hold
in a persistent build of either storage width:

- One fresh read of the thread index per item. When the warp index comes from the kernel-entry
  read and stays live through the loop, ptxas can keep it in a per-thread register, compile the
  role choice as a divergent branch, and make the MMA and TMA warps rebuild their uniform
  operands on every instruction they issue. So the kernel reads %tid.x once at the top of every
  item (`_tid_x_fresh`), masks it with 1023 (the largest thread index a CTA can have) so that
  the compiler knows it is not negative and the divisions are shifts, and derives the item's
  warp index and warpgroup index from that one read.
- One opaque copy per warp check of the role dispatch. Compared on one per-item value, the four
  warp checks of the dispatch lose their source order (the PTX tests the MMA-KQ and TMA-VO warp
  numbers first), and ptxas can place those two bodies after the dequant loops, which moves
  every hot loop and makes INT2 1 to 2% slower. So each of those four checks compares its own
  copy of the warp index, made by `_opaque_i32`, a side-effecting `mov` the compiler can
  neither merge with another copy nor see through.

THE RULE, read from the source (the modules import the DSL, so they are parsed, not imported):

- The role choices checked are the ones every item makes. The item loop is exactly
  `for _item in cutlass.range(items_this_cta, unroll=1)`, a persistent build gives it
  `cute.ceil_div(num_items - bidz, grid_z_rt)` items, and no `return`, `raise`, `continue` or
  `break` in the loop (outside a nested function) leaves an item early.
- `kernel` binds `persistent` once, as `persistent = self.persistent`, and the constructor sets
  `self.persistent` once, from a closed list of inputs that does not name the storage width
  (`PERSISTENCE_INPUTS`: the requested flag, the reduction mode and the debug tap, never rebound),
  so the guard the rule keys on is the build's own flag, with no condition on the width.
- Neither role index from the kernel-entry read is read in a persistent build's view of the
  item loop. The entry role indices are the names `kernel` assigns before the loop from a
  `make_warp_uniform` call. The persistent view takes the body of
  `if cutlass.const_expr(persistent)`, the else arm of `if cutlass.const_expr(not persistent)`,
  and both arms of every other `if`. A condition that narrows the persistent arm, a width for
  example, therefore exposes its other arm.
- The item's role indices and the copy function are assigned under exactly
  `if cutlass.const_expr(persistent)`; the other arm assigns the indices from the entry values
  and the copy function as the identity, so a non-persistent build compiles unchanged. In a
  persistent build the loop calls `_tid_x_fresh` exactly once, and each name in `PER_ITEM` is
  bound exactly once in the loop, by exactly its expression there. So a second fresh read, a
  later overwrite, an unmasked read and an expression that mixes in an entry value all fail.
- Every role choice in the loop reads a per-item index, at every site, in the form
  `ROLE_CHOICES` gives, under the tests it gives: the four warp checks of the dispatch as
  `_role_w(warp_idx_l)`, the others as the index itself. A role choice is the test of an `if`
  or `elif` that compares a warp position with a role's warp or warpgroup number (or, for the
  varlen sequence-length load, with the warp number 1). The loop must hold exactly that list: a
  choice whose index is replaced by any other expression (an entry value, a shared copy, the
  bare index where a copy belongs), a choice that is removed, replaced by another test or put
  under another test, and a new choice not in the table all fail. `_role_w` is called nowhere
  else in the loop, so every copy belongs to one check. Any other comparison with a role
  number in the loop fails too, except one, matched exactly: the trace-time warp-budget
  assertion `init_warp_l <= self.threads_per_cta // warp_threads`.
- The two helpers in `int_fold_asm.py` are exactly the single side-effecting `mov` each needs
  to be, returned unchanged: `_tid_x_fresh` reads %tid.x and `_opaque_i32` copies its operand.
  Each definition is the only binding of its name in that module, and the kernel module binds
  each name only by importing it from there, so the functions the kernel calls are these.

WHAT IT DOES NOT COVER, because the kernel does not do it. Two warp positions still come from
the kernel-entry read in a persistent build: the warp's place in its warpgroup
(`warpgroup_widx`), which the softmax and correction roles receive, and the warp index the
persistent epilogue compares after the loop. Re-reading those too makes some persistent INT4
builds slower, so the kernel keeps the kernel-entry reads there. Thread positions (`tidx`,
`lane_idx`, `warpgroup_tidx`) are per-thread values and keep their kernel-entry reads.

The last test applies mutations to the sources in memory and requires the rule to reject each
one, so the rule cannot pass vacuously.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNEL = ROOT / "src/fosca/_sm100/int_fold_decode.py"
ASM = ROOT / "src/fosca/_sm100/int_fold_asm.py"


def _expr(text):
    return ast.dump(ast.parse(text, mode="eval").body)


PERSISTENT = _expr("cutlass.const_expr(persistent)")
NOT_PERSISTENT = _expr("cutlass.const_expr(not persistent)")
#: The warp checks of the role dispatch compare their own copy of the warp index.
COPY = "_role_w(warp_idx_l)"
#: The four warp checks of the dispatch, and the rest of the `elif` chain they belong to.
_EXIT = "exit_early"
_DISPATCH = [
    (COPY, "==", "tma_qk_warp_id"),
    (COPY, "==", "tma_vo_warp_id"),
    (COPY, "==", "mma_kq_warp_id"),
    (COPY, "==", "mma_vp_warp_id"),
    ("warpgroup_idx_l", "in", "softmax_warpgroup_ids"),
    ("warpgroup_idx_l", "==", "correction_warpgroup_id"),
    ("warpgroup_idx_l", "in", "dequant_warpgroup_ids"),
]


def _elif_chain(head, entries):
    """The `elif` arms after `if head:`, each under the negation of every test before it."""
    out, under = [], (f"not {head}",)
    for index, op, number in entries:
        out.append((index, op, number, under))
        under += (f"not {index} {op} {number}",)
    return out


#: Every role choice in the persistent item loop, in source order: (the per-item index it reads,
#: in exactly this form, the comparison, the role number, and the tests it sits under in the
#: loop, outermost first). A test is written as `ast.unparse` gives it where the choice is in the
#: test's body, and as `not <test>` where it is in the `else` or a later `elif`; the guard
#: `cutlass.const_expr(persistent)`, true in a persistent build, is left out.
ROLE_CHOICES = [
    ("warp_idx_l", "==", "1", ("cutlass.const_expr(is_varlen)",)),  # varlen: this item's length
    ("warp_idx_l", "==", "init_warp_l",
     ("cutlass.const_expr(do_atomic_red)",)),                    # cluster reduction mbarriers
    ("warp_idx_l", "==", "init_warp_l", ()),                     # Q load mbarrier
    ("warp_idx_l", "==", "init_warp_l", ()),                     # sM (column max)
    ("warp_idx_l", "==", "init_warp_l", ()),                     # sL (column sum)
    ("warp_idx_l", "==", "init_warp_l", ("cutlass.const_expr(fold_p)",)),  # sC (folded correction)
    ("warp_idx_l", "==", "mma_vp_warp_id", (_EXIT,)),            # early exit: tmem
    ("warpgroup_idx_l", "==", "correction_warpgroup_id",
     (_EXIT, "not warp_idx_l == mma_vp_warp_id")),               # early exit: final handshake
    *_elif_chain(_EXIT, _DISPATCH),                              # the role dispatch
    ("warp_idx_l", "==", "reduction_warp_id", ()),
]
ROLE_NUMBERS = {number for _, _, number, _ in ROLE_CHOICES} - {"1"}
#: The one comparison with a role number in the loop that is not a role choice: the trace-time
#: check that the init warps fit in the CTA. Matched exactly, inside an `assert`.
BUDGET_ASSERTION = _expr("init_warp_l <= self.threads_per_cta // warp_threads")
#: What a persistent build binds in the item loop, each name once, by exactly this expression:
#: the one fresh read, the two role indices derived from it, and the copy function.
PER_ITEM = {
    "tid_fresh": "_tid_x_fresh() & 1023",
    "warp_idx_l": "cute.arch.make_warp_uniform(tid_fresh // warp_threads)",
    "warpgroup_idx_l": "cute.arch.make_warp_uniform(tid_fresh // warpgroup_threads)",
    "_role_w": "_opaque_i32",
}
#: The item loop, and the item count a persistent build gives it: the last binding of
#: `items_this_cta` before the loop, in a persistent build's view of the kernel.
ITEM_LOOP = "cutlass.range(items_this_cta, unroll=1)"
ITEM_COUNT = "cute.ceil_div(num_items - bidz, grid_z_rt)"
#: The one binding of `persistent` in `kernel`: the constructor's flag.
PERSISTENT_FLAG = "self.persistent"
#: What the constructor may compute `self.persistent` from: these parameters of `__init__`, which
#: `__init__` never rebinds, and these builtins. It may also read a `self` attribute that the class
#: sets exactly once, in `__init__`, from the same inputs. The list is closed on purpose: none of
#: these names the storage width, and a change that computes persistence from anything else has to
#: extend it here, in review, rather than pass through a trace of what it calls.
PERSISTENCE_INPUTS = {"persistent", "reduction_mode", "debug_s"}
PERSISTENCE_BUILTINS = {"bool", "int"}
#: Harmless edits the rule must accept: a rewrite of the constructor's persistence over those
#: inputs, and a module-level helper or comprehension that uses the name `bool` locally.
ACCEPTED_REWRITES = {
    "a module-level comprehension uses bool as its own variable": [(
        "from fosca._sm100.int_fold_asm import (",
        "_PROBE = [bool for bool in (False, True)]\n\n"
        "from fosca._sm100.int_fold_asm import (")],
    "a module-level helper under a dead branch has a parameter named bool": [(
        "from fosca._sm100.int_fold_asm import (",
        "if False:\n    def _helper(bool):\n        return bool\n\n\n"
        "from fosca._sm100.int_fold_asm import (")],
    "persistence spelled through do_none_red": [(
        "        self.persistent = bool(int(persistent)) and (\n"
        "            (reduction_mode == \"none\" or reduction_mode is None)\n"
        "            and not debug_s\n"
        "        )\n",
        "        self.persistent = bool(int(persistent)) and self.do_none_red and not debug_s\n")],
}
#: What the non-persistent arm binds instead: the entry values, and the identity for the copy.
NON_PERSISTENT = {"warp_idx_l": "warp_idx", "warpgroup_idx_l": "warpgroup_idx",
                  "_role_w": "(lambda w: w)"}
#: The two inline-assembly helpers, each exactly this signature and, after its docstring, this one
#: statement: a single side-effecting `mov`, so it is neither hoisted nor merged, whose value is
#: returned unchanged.
HELPERS = {
    "_tid_x_fresh": ("*, loc=None, ip=None", 'return Int32(llvm.inline_asm(T.i32(), [], '
                     '"mov.u32 $0, %tid.x;", "=r", has_side_effects=True, is_align_stack=False))'),
    "_opaque_i32": ("x, *, loc=None, ip=None", 'return Int32(llvm.inline_asm(T.i32(), '
                    '[Int32(x).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=r,r", '
                    'has_side_effects=True, is_align_stack=False))'),
}
assert {index for index, _, _, _ in ROLE_CHOICES} == {"warp_idx_l", "warpgroup_idx_l", COPY}
_OPS = {ast.Eq: "==", ast.In: "in"}


def _class(src):
    return next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.ClassDef) and n.name == "IntDecodeSm100")


def _kernel(src):
    return next(m for m in _class(src).body
                if isinstance(m, ast.FunctionDef) and m.name == "kernel")


def _item_loop(fn):
    idx = [i for i, s in enumerate(fn.body) if isinstance(s, ast.For)
           and isinstance(s.target, ast.Name) and s.target.id == "_item"]
    assert len(idx) == 1, f"expected one `for _item` loop in kernel, found {len(idx)}"
    return idx[0]


def _loads(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _calls(node, name):
    return any(isinstance(c, ast.Call) and ast.unparse(c.func).split(".")[-1] == name
               for c in ast.walk(node))


def _pairs(assign):
    """(target name, value) for each name an assignment binds, unpacking a tuple of values."""
    out = []
    for tgt in assign.targets:
        if isinstance(tgt, ast.Name):
            out.append((tgt.id, assign.value))
        elif isinstance(tgt, ast.Tuple) and isinstance(assign.value, ast.Tuple) \
                and len(tgt.elts) == len(assign.value.elts):
            out += [(t.id, v) for t, v in zip(tgt.elts, assign.value.elts)
                    if isinstance(t, ast.Name)]
        elif isinstance(tgt, ast.Tuple):
            out += [(t.id, assign.value) for t in tgt.elts if isinstance(t, ast.Name)]
    return out


def _header(node):
    """How a compound statement other than `if` appears among the tests a node sits under."""
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return f"for {ast.unparse(node.target)} in {ast.unparse(node.iter)}"
    if isinstance(node, ast.While):
        return f"while {ast.unparse(node.test)}"
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return "with " + ", ".join(ast.unparse(item) for item in node.items)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"def {node.name}"
    return type(node).__name__.lower()


_COMPOUND = (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try,
             ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.Match)


def _persistent_view(stmts):
    """Every node a persistent build traces among `stmts` (see the module docstring), in source
    order, each mapped to the tests it sits under (written as `ROLE_CHOICES` says)."""
    out = {}

    def visit(node, under):
        if isinstance(node, ast.If):
            test = ast.dump(node.test)
            if test == PERSISTENT:
                arms = [(node.body, under)]
            elif test == NOT_PERSISTENT:
                arms = [(node.orelse, under)]
            else:
                visit(node.test, under)
                text = ast.unparse(node.test)
                arms = [(node.body, under + (text,)), (node.orelse, under + (f"not {text}",))]
            for arm, inner in arms:
                for s in arm:
                    visit(s, inner)
            return
        out[node] = under
        if isinstance(node, _COMPOUND):
            under = under + (_header(node),)
        for child in ast.iter_child_nodes(node):
            visit(child, under)

    for s in stmts:
        visit(s, ())
    return out


def entry_role_indices(fn):
    """The names `kernel` binds before the item loop directly from `make_warp_uniform`."""
    before = fn.body[:_item_loop(fn)]
    return {name for s in before for a in ast.walk(s) if isinstance(a, ast.Assign)
            for name, value in _pairs(a) if _calls(value, "make_warp_uniform")}


def _role_choices(loop_body, view):
    """Every role choice in the persistent view, in source order, as (index, op, number, the
    tests it sits under), with
    every other comparison that involves a role number collected separately. A role choice is
    an `if` / `elif` test of one comparison whose right side is a role number or an integer
    constant. The budget assertion is the one exemption."""
    parents = {c: p for p in ast.walk(ast.Module(body=loop_body, type_ignores=[]))
               for c in ast.iter_child_nodes(p)}
    choices, other = [], []
    for node in view:
        if not isinstance(node, ast.Compare):
            continue
        parent = parents.get(node)
        right = node.comparators[0]
        number = (right.id if isinstance(right, ast.Name) and right.id in ROLE_NUMBERS
                  else repr(right.value) if isinstance(right, ast.Constant)
                  and isinstance(right.value, int) and not isinstance(right.value, bool)
                  else None)
        if (isinstance(parent, ast.If) and parent.test is node and len(node.ops) == 1
                and number is not None):
            choices.append((ast.unparse(node.left),
                            _OPS.get(type(node.ops[0]), type(node.ops[0]).__name__),
                            number, view[node], node.lineno))
        elif _loads(node) & ROLE_NUMBERS:
            if isinstance(parent, ast.Assert) and ast.dump(node) == BUDGET_ASSERTION:
                continue
            other.append(f"{node.lineno} {ast.unparse(node)}")
    return choices, other


def check_rule(src):
    fn = _kernel(src)
    loop = fn.body[_item_loop(fn)]
    entry = entry_role_indices(fn)
    assert {"warp_idx", "warpgroup_idx"} <= entry, entry
    # The guard the rule keys on means what the kernel is built as: `persistent` is bound once in
    # `kernel`, to the constructor's flag, with no condition on the width.
    cls = _class(src)
    sets = [n for n in ast.walk(cls) if isinstance(n, ast.Assign)
            and [ast.unparse(t) for t in n.targets] == ["self.persistent"]]
    stores = [n for n in ast.walk(cls) if isinstance(n, ast.Attribute)
              and isinstance(n.ctx, ast.Store) and ast.unparse(n) == "self.persistent"]
    init = next(m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == "__init__")
    assert len(sets) == len(stores) == 1 and any(n is sets[0] for n in ast.walk(init)), (
        f"the constructor must set `self.persistent` once, found {[n.lineno for n in stores]}")
    bad = _outside_inputs(src, cls, init, sets[0].value, sets[0].targets[0])
    assert not bad, (f"line {sets[0].lineno}: `self.persistent` must be computed only from "
                     f"{sorted(PERSISTENCE_INPUTS | PERSISTENCE_BUILTINS)} and attributes set once "
                     f"from them, with nothing else writing or intercepting it; found {bad}")
    binds = _bindings(fn, "persistent")
    assert (len(binds) == 1 and isinstance(binds[0][2], ast.Name)
            and ast.dump(_assign_of(fn, binds[0][2]).value) == _expr(PERSISTENT_FLAG)), (
        f"`kernel` must bind `persistent` once, as `persistent = {PERSISTENT_FLAG}`, found "
        f"{[(line, ast.unparse(_assign_of(fn, n)) if kind == 'store' else kind)
            for line, kind, n in binds]}")

    # 0. The loop is the item loop, a persistent build gives it the item count, and nothing in an
    #    item leaves it early, so the role choices below are the ones every item makes.
    assert ast.dump(loop.iter) == _expr(ITEM_LOOP), (
        f"the item loop must be `for _item in {ITEM_LOOP}`, not `{ast.unparse(loop.iter)}`")
    counts = [n for n in _persistent_view(fn.body[:_item_loop(fn)]) if isinstance(n, ast.Assign)
              and [ast.unparse(t) for t in n.targets] == ["items_this_cta"]]
    assert counts and ast.dump(counts[-1].value) == _expr(ITEM_COUNT), (
        f"a persistent build must loop over `{ITEM_COUNT}` items, not "
        f"`{ast.unparse(counts[-1].value) if counts else None}`")
    view = _persistent_view(loop.body)
    exits = [f"{n.lineno} {type(n).__name__.lower()}" for n, under in view.items()
             if isinstance(n, (ast.Return, ast.Raise, ast.Continue, ast.Break))
             and not any(u.startswith("def ") or u == "lambda" for u in under)]
    assert not exits, f"statements that leave an item before its role choices: {exits}"

    # 1. No entry role index is read in the item loop of a persistent build.
    stale = sorted((n.lineno, n.id) for n in view
                   if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                   and n.id in entry)
    assert not stale, f"entry role indices read in the item loop: {stale}"

    # 2. The per-item names: exactly `const_expr(persistent)`, entry values and identity else.
    guards = [s for s in loop.body if isinstance(s, ast.If)
              and "warp_idx_l" in {name for a in s.body if isinstance(a, ast.Assign)
                                   for name, _ in _pairs(a)}]
    assert len(guards) == 1, f"expected one `if` that assigns warp_idx_l, found {len(guards)}"
    guard = guards[0]
    assert ast.dump(guard.test) == PERSISTENT, (
        f"the item's role indices must be chosen by exactly `cutlass.const_expr(persistent)`, "
        f"not `{ast.unparse(guard.test)}`")
    arm = {name for a in guard.body if isinstance(a, ast.Assign) for name, _ in _pairs(a)}
    assert arm == set(PER_ITEM), arm
    other = {name: value for a in guard.orelse if isinstance(a, ast.Assign)
             for name, value in _pairs(a)}
    assert {k: ast.dump(v) for k, v in other.items()} == {
        k: _expr(v) for k, v in NON_PERSISTENT.items()}, (
        "the non-persistent arm keeps the kernel-entry values and compares them directly",
        {k: ast.unparse(v) for k, v in other.items()})

    # 3. In a persistent build: one fresh read, and each per-item name bound once, exactly.
    reads = [n for n in view if isinstance(n, ast.Call)
             and ast.unparse(n.func).split(".")[-1] == "_tid_x_fresh"]
    assert len(reads) == 1, (
        f"a persistent item must read %tid.x once, found {len(reads)} reads at lines "
        f"{[n.lineno for n in reads]}")
    binders = {c: p for p in view for c in ast.iter_child_nodes(p)}
    for name, value in PER_ITEM.items():
        stores = [n for n in view if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                  and n.id == name]
        assert len(stores) == 1, (
            f"{name} must be bound once in a persistent build's item loop, "
            f"found at lines {[n.lineno for n in stores]}")
        bind = binders.get(stores[0])
        assert (isinstance(bind, ast.Assign) and bind.targets == [stores[0]]
                and ast.dump(bind.value) == _expr(value)), (
            f"line {stores[0].lineno}: {name} must be bound by exactly `{value}`")
    copies = [n for n in view if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
              and n.id == "_opaque_i32"]
    assert len(copies) == 1, (
        f"`_opaque_i32` must be read only where `_role_w` is bound, found at lines "
        f"{[n.lineno for n in copies]}")

    # 4. Every role choice, site by site, is the one the table names, and `_role_w` is called
    #    only by the checks the table gives it, one call each.
    choices, other = _role_choices(loop.body, view)
    assert not other, f"comparisons with a role number that are not role choices: {other}"
    found = [c[:4] for c in choices]
    assert found == ROLE_CHOICES, (
        "the item loop's role choices differ from ROLE_CHOICES (index, op, number, under):\n"
        + "\n".join(f"  line {c[4]}: {c[:4]}" for c in choices))
    uses = [n for n in view if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            and n.id == "_role_w"]
    assert len(uses) == sum(index == COPY for index, _, _, _ in ROLE_CHOICES), (
        f"`_role_w` must be called only by the dispatch's warp checks, found at lines "
        f"{[n.lineno for n in uses]}")


def check_helpers(src):
    """Each helper is exactly its signature and its one `return` (`HELPERS`), and its definition is
    the only binding of its name in the module, so what the kernel imports is that function."""
    tree = ast.parse(src)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name, (signature, statement) in HELPERS.items():
        assert name in fns, f"{name} is not defined in int_fold_asm.py"
        fn = fns[name]
        binds = [(line, kind) for line, kind, _ in _bindings(tree, name)]
        assert binds == [(fn.lineno, "definition")], (
            f"`{name}` must be bound only by its definition in int_fold_asm.py, found {binds}")
        assert [ast.unparse(d) for d in fn.decorator_list] == ["dsl_user_op"], name
        assert ast.unparse(fn.args) == signature, f"{name}({ast.unparse(fn.args)})"
        body = fn.body[1:] if ast.get_docstring(fn) is not None else fn.body
        assert [ast.dump(b) for b in body] == [ast.dump(ast.parse(statement).body[0])], (
            f"{name} must be exactly `{statement}`, not "
            f"`{'; '.join(ast.unparse(b) for b in body)}`")


def test_the_entry_role_indices_are_the_two_the_kernel_derives():
    assert entry_role_indices(_kernel(KERNEL.read_text())) == {"warp_idx", "warpgroup_idx"}


def test_the_kernel_chooses_every_role_from_one_fresh_read_in_persistent_builds():
    check_rule(KERNEL.read_text())


#: Ways to write an attribute or a module name, or to replace the constructor, that are not plain
#: assignments.
DYNAMIC_WRITES = {"setattr", "delattr", "__setattr__", "__delattr__", "vars", "globals", "locals"}
HOOKS = {"__setattr__", "__delattr__", "__getattribute__", "__getattr__", "__new__",
         "__init_subclass__", "__bool__", "__class_getitem__"}


def _scope_bindings(body, name):
    """Every place one scope (a module or a class body) binds `name`, through its `if`, `for`,
    `with`, `try` and `match` blocks, without entering a function, class or lambda inside it
    (whose own name counts as a binding) or a comprehension's own targets, as `_bindings`
    reports them."""
    out, todo = [], list(body)
    while todo:
        n = todo.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.name == name:
                out.append((n.lineno, "definition", n))
            continue
        if isinstance(n, ast.Lambda):
            continue
        if isinstance(n, ast.comprehension):
            # The target is the comprehension's own; a walrus in the condition still escapes.
            todo += [n.iter, *n.ifs]
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)) and n.id == name:
            out.append((n.lineno, "store", n))
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name == name:
            out.append((n.lineno, "match capture", n))
        elif isinstance(n, ast.MatchMapping) and n.rest == name:
            out.append((n.lineno, "match capture", n))
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out += [(n.lineno, "import", n) for alias in n.names
                    if (alias.asname or alias.name.split(".")[0]) == name]
        todo.extend(ast.iter_child_nodes(n))
    return sorted(out, key=lambda b: b[0])


def _outside_inputs(src, cls, init, expr, expr_target=None):
    """What `expr` reads beyond the closed list of persistence inputs (`PERSISTENCE_INPUTS`): any
    call other than to the builtins, any name not in the list, any attribute other than
    `self.<x>` set exactly once in the class, in `__init__`, from the list alone, and `self`
    used other than as such a receiver. Also reports an input that `__init__` rebinds, a builtin
    rebound at module scope; in the module, a dynamic write (`setattr`, `vars`, `globals` and the
    like), `__dict__` or `__class__` access, or any other store to an attribute named
    `persistent`; in the class, an attribute hook, `__new__` or class-level `persistent`; and an
    `__init__` bound more than once."""
    bad = []
    # Each input is a parameter of `__init__` that nothing rebinds; each builtin is bound nowhere
    # in `__init__` (a parameter included) and nowhere at module scope.
    for name in PERSISTENCE_INPUTS:
        binds = [(line, kind) for line, kind, _ in _bindings(init, name)]
        if [kind for _, kind in binds] != ["argument"]:
            bad.append(f"{name} bound in __init__ as {binds}, not only as its parameter")
    module = ast.parse(src)
    for name in PERSISTENCE_BUILTINS:
        bad += [f"{line} {name} bound in __init__ ({kind})"
                for line, kind, _ in _bindings(init, name)]
        bad += [f"{line} {name} rebound at module scope"
                for line, _, _ in _scope_bindings(module.body, name)]
    # Nothing else may write or intercept the flag, or replace the constructor: no dynamic
    # attribute writes or `__dict__` access in the module, no attribute hooks, `__new__` or
    # second `__init__` in the class, and no class-level `persistent`.
    for n in ast.walk(module):
        if isinstance(n, ast.Call) and ast.unparse(n.func).split(".")[-1] in DYNAMIC_WRITES:
            bad.append(f"{n.lineno} call {ast.unparse(n.func)}")
        elif isinstance(n, ast.Attribute) and n.attr in ("__dict__", "__class__"):
            bad.append(f"{n.lineno} {ast.unparse(n)}")
        elif (isinstance(n, ast.Attribute) and n.attr == "persistent"
              and isinstance(n.ctx, (ast.Store, ast.Del))
              and (n.lineno, n.col_offset) != (expr_target.lineno, expr_target.col_offset)):
            bad.append(f"{n.lineno} another store to {ast.unparse(n)}")
    for name in HOOKS | {"persistent"}:
        bad += [f"{line} class-level {name}" for line, _, _ in _scope_bindings(cls.body, name)]
    inits = [line for line, _, _ in _scope_bindings(cls.body, "__init__")]
    if inits != [init.lineno]:
        bad.append(f"__init__ bound at {inits}, not once")

    def leaves(e, allow_attributes):
        receivers = {id(a.value) for a in ast.walk(e) if isinstance(a, ast.Attribute)}
        for x in ast.walk(e):
            if isinstance(x, ast.Call):
                if not (isinstance(x.func, ast.Name) and x.func.id in PERSISTENCE_BUILTINS
                        and not x.keywords):
                    bad.append(f"{x.lineno} call {ast.unparse(x.func)}")
            elif isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
                if x.id == "self":
                    if id(x) not in receivers:
                        bad.append(f"{x.lineno} self used other than as `self.<x>`")
                elif x.id not in PERSISTENCE_INPUTS | PERSISTENCE_BUILTINS:
                    bad.append(f"{x.lineno} name {x.id}")
            elif isinstance(x, ast.Attribute):
                if not (isinstance(x.value, ast.Name) and x.value.id == "self"):
                    bad.append(f"{x.lineno} attribute {ast.unparse(x)}")
                elif not allow_attributes:
                    bad.append(f"{x.lineno} attribute {ast.unparse(x)}")
                else:
                    attribute(x)
            elif isinstance(x, (ast.Lambda, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom,
                                ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                                ast.Subscript, ast.Starred)):
                bad.append(f"{x.lineno} {type(x).__name__}")

    def attribute(x):
        stores = [n for n in ast.walk(cls) if isinstance(n, ast.Attribute)
                  and isinstance(n.ctx, (ast.Store, ast.Del)) and isinstance(n.value, ast.Name)
                  and n.value.id == "self" and n.attr == x.attr]
        sets = [a for a in ast.walk(init) if isinstance(a, ast.Assign)
                and [ast.unparse(t) for t in a.targets] == [f"self.{x.attr}"]]
        if len(stores) != 1 or len(sets) != 1 or sets[0].targets[0] is not stores[0]:
            bad.append(f"{x.lineno} attribute self.{x.attr} is not set exactly once, in __init__")
        else:
            leaves(sets[0].value, allow_attributes=False)

    leaves(expr, allow_attributes=True)
    return bad


def _assign_of(tree, target):
    """The assignment statement whose target is the Name node `target`, or None."""
    return next((a for a in ast.walk(tree) if isinstance(a, ast.Assign) and target in a.targets),
                ast.Assign(targets=[], value=ast.Constant(None)))


def _bindings(tree, name):
    """Every place `tree` binds `name`, as (line, kind, node): an assignment or other store, a
    definition, an argument, a `match` capture, or an import."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)) and n.id == name:
            out.append((n.lineno, "store", n))
        elif (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
              and n.name == name):
            out.append((n.lineno, "definition", n))
        elif isinstance(n, ast.arg) and n.arg == name:
            out.append((n.lineno, "argument", n))
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name == name:
            out.append((n.lineno, "match capture", n))
        elif isinstance(n, ast.MatchMapping) and n.rest == name:
            out.append((n.lineno, "match capture", n))
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out += [(n.lineno, "import", n) for alias in n.names
                    if (alias.asname or alias.name.split(".")[0]) == name]
    return out


def check_imports(src):
    """The kernel module binds each helper's name once, by importing it unrenamed from
    `int_fold_asm`, and nowhere else: no assignment, definition, argument, loop target or other
    import may shadow it."""
    tree = ast.parse(src)
    for name in HELPERS:
        binds = _bindings(tree, name)
        ok = (len(binds) == 1 and binds[0][1] == "import"
              and binds[0][2].module == "fosca._sm100.int_fold_asm"
              and [a.asname for a in binds[0][2].names if a.name == name] == [None])
        assert ok, (f"`{name}` must be bound only by its import from int_fold_asm, found "
                    f"{[(line, kind) for line, kind, _ in binds]}")


def test_the_kernel_imports_both_helpers_from_the_asm_module_and_never_rebinds_them():
    check_imports(KERNEL.read_text())


def test_the_helpers_are_side_effecting_single_moves():
    check_helpers(ASM.read_text())


#: Mutations, each of which must break the rule: the check that must reject it (the kernel's
#: rule, the helpers, or the kernel module's imports) and a list of (text, replacement), where
#: each text occurs exactly once in that file's source and all are applied.
_K, _A, _I = "kernel", "asm", "imports"
_FRESH = "tid_fresh = _tid_x_fresh() & 1023"
_WARP = "warp_idx_l = cute.arch.make_warp_uniform(tid_fresh // warp_threads)"
_WARPGROUP = "warpgroup_idx_l = cute.arch.make_warp_uniform(tid_fresh // warpgroup_threads)"
_REDUCTION = "            if warp_idx_l == reduction_warp_id:"
MUTATIONS = {
    "a dispatch reads the entry warp index": (_K, [(
        "elif _role_w(warp_idx_l) == tma_qk_warp_id:",
        "elif _role_w(warp_idx) == tma_qk_warp_id:")]),
    "a warpgroup dispatch reads the entry warpgroup index": (_K, [(
        "elif warpgroup_idx_l == correction_warpgroup_id:\n"
        "                self._correction_warpgroup(",
        "elif warpgroup_idx == correction_warpgroup_id:\n"
        "                self._correction_warpgroup(")]),
    "an init-warp test reads the entry warp index": (_K, [(
        "# Initialize Q load mbarrier\n            if warp_idx_l == init_warp_l:",
        "# Initialize Q load mbarrier\n            if warp_idx == init_warp_l:")]),
    "a width condition narrows the per-item read": (_K, [(
        "if cutlass.const_expr(persistent):\n                " + _FRESH,
        "if cutlass.const_expr(persistent and bits == 4):\n                " + _FRESH)]),
    "the warp index is derived from the entry thread read": (_K, [(
        _WARP, "warp_idx_l = cute.arch.make_warp_uniform(tidx // warp_threads)")]),
    "the reduction dispatch reads the entry warp index": (_K, [(
        _REDUCTION, "            if warp_idx == reduction_warp_id:")]),
    "the varlen choice derives the warp from the entry thread index": (_K, [(
        "if warp_idx_l == 1:", "if tidx // warp_threads == 1:")]),
    "an init-warp choice derives the warp from the entry thread index": (_K, [(
        "# Initialize Q load mbarrier\n            if warp_idx_l == init_warp_l:",
        "# Initialize Q load mbarrier\n            if tidx // warp_threads == init_warp_l:")]),
    "the early-exit MMA VP choice is replaced by `if True`": (_K, [(
        "if exit_early:\n                if warp_idx_l == mma_vp_warp_id:",
        "if exit_early:\n                if True:")]),
    "a stale value overwrites the fresh warp index": (_K, [(
        _WARP,
        _WARP + "\n"
        "                warp_idx_l = cute.arch.make_warp_uniform(tidx // warp_threads)")]),
    "the fresh read is mixed with the entry thread index": (_K, [(
        _FRESH, "tid_fresh = (_tid_x_fresh() * 0 + tidx) & 1023")]),
    "the read is not masked": (_K, [(_FRESH, "tid_fresh = _tid_x_fresh()")]),
    "a second fresh read derives the warpgroup index": (_K, [(
        _WARPGROUP,
        "warpgroup_idx_l = cute.arch.make_warp_uniform("
        "(_tid_x_fresh() & 1023) // warpgroup_threads)")]),
    "the reduction warp makes its own fresh read": (_K, [(
        _REDUCTION,
        "            if cute.arch.make_warp_uniform(_tid_x_fresh() // warp_threads)"
        " == reduction_warp_id:")]),
    "two warp checks share one copy": (_K, [
        ("            if exit_early:\n                if warp_idx_l == mma_vp_warp_id:",
         "            role_w_shared = _role_w(warp_idx_l)\n"
         "            if exit_early:\n                if warp_idx_l == mma_vp_warp_id:"),
        ("elif _role_w(warp_idx_l) == tma_qk_warp_id:", "elif role_w_shared == tma_qk_warp_id:"),
        ("elif _role_w(warp_idx_l) == tma_vo_warp_id:", "elif role_w_shared == tma_vo_warp_id:")]),
    "a warp check compares the index without its copy": (_K, [(
        "elif _role_w(warp_idx_l) == mma_kq_warp_id:", "elif warp_idx_l == mma_kq_warp_id:")]),
    "the opaque copy is removed": (_K, [(
        "                _role_w = _opaque_i32\n", "                _role_w = (lambda w: w)\n")]),
    "a width condition narrows the opaque copy": (_K, [(
        "                _role_w = _opaque_i32\n",
        "                _role_w = _opaque_i32 if bits == 2 else (lambda w: w)\n")]),
    "the opaque copy is not side-effecting": (_A, [(
        '"mov.b32 $0, $1;", "=r,r",\n        has_side_effects=True',
        '"mov.b32 $0, $1;", "=r,r",\n        has_side_effects=False')]),
    "the copy's value is changed after the move": (_A, [(
        '"mov.b32 $0, $1;", "=r,r",\n        has_side_effects=True, is_align_stack=False))',
        '"mov.b32 $0, $1;", "=r,r",\n        has_side_effects=True, is_align_stack=False)) & 0')]),
    "the copy moves another operand": (_A, [(
        "[Int32(x).ir_value(loc=loc, ip=ip)]", "[Int32(0).ir_value(loc=loc, ip=ip)]")]),
    "the item loop runs no item": (_K, [(
        "for _item in cutlass.range(items_this_cta, unroll=1):",
        "for _item in cutlass.range(0, unroll=1):")]),
    "a persistent build is given no item": (_K, [(
        "items_this_cta = cute.ceil_div(num_items - bidz, grid_z_rt)", "items_this_cta = 0")]),
    "the dispatch's first test is replaced by `if True`": (_K, [(
        "            if exit_early:\n                if warp_idx_l == mma_vp_warp_id:",
        "            if True:\n                if warp_idx_l == mma_vp_warp_id:")]),
    "every item skips its dispatch": (_K, [(
        "            if exit_early:\n                if warp_idx_l == mma_vp_warp_id:",
        "            continue\n"
        "            if exit_early:\n                if warp_idx_l == mma_vp_warp_id:")]),
    "the reduction choice is put behind a dead branch": (_K, [(
        _REDUCTION,
        "            if False:\n                pass\n"
        "            elif warp_idx_l == reduction_warp_id:")]),
    "the copy is shadowed by an identity before the loop": (_I, [(
        "        items_this_cta = 1\n",
        "        items_this_cta = 1\n        _opaque_i32 = (lambda w: w)\n")]),
    "the fresh read is shadowed by the entry read": (_I, [(
        "        items_this_cta = 1\n",
        "        items_this_cta = 1\n        _tid_x_fresh = (lambda: tidx)\n")]),
    "the fresh read is imported under its name from elsewhere": (_I, [(
        "    _tid_x_fresh,\n    _opaque_i32,\n)",
        "    _opaque_i32,\n)\nfrom cutlass import thread_idx as _tid_x_fresh")]),
    "the copy is rebound to an identity after its definition": (_A, [(
        '"mov.b32 $0, $1;", "=r,r",\n        has_side_effects=True, is_align_stack=False))\n',
        '"mov.b32 $0, $1;", "=r,r",\n        has_side_effects=True, is_align_stack=False))\n'
        "\n\n_opaque_i32 = lambda x: x\n")]),
    "the fresh read is rebound to a constant after its definition": (_A, [(
        '"mov.u32 $0, %tid.x;", "=r",\n        has_side_effects=True, is_align_stack=False))\n',
        '"mov.u32 $0, %tid.x;", "=r",\n        has_side_effects=True, is_align_stack=False))\n'
        "\n\n_tid_x_fresh = lambda: 0\n")]),
    "the kernel's persistence depends on the width": (_K, [(
        "        f4_feed = self.dequant_int2_to_f4\n        persistent = self.persistent\n",
        "        f4_feed = self.dequant_int2_to_f4\n"
        "        persistent = self.persistent and bits == 2\n")]),
    "the constructor narrows the requested flag by the width": (_K, [(
        "        self.persistent = bool(int(persistent)) and (",
        "        persistent = persistent and bits == 2\n"
        "        self.persistent = bool(int(persistent)) and (")]),
    "the constructor's persistence depends on the width through a method": (_K, [
        ("    def __init__(\n",
         "    def _narrow(self):\n        return self.bits == 2\n\n    def __init__(\n"),
        ("        self.persistent = bool(int(persistent)) and (",
         "        self.persistent = bool(int(persistent)) and self._narrow() and (")]),
    "the constructor's persistence reads an attribute a method it calls sets from the width": (_K, [
        ("    def __init__(\n", "    def _request(self, requested):\n"
         "        self._persistent_request = requested and self.bits == 2\n\n    def __init__(\n"),
        ("        self.persistent = bool(int(persistent)) and (",
         "        self._request(persistent)\n"
         "        persistent = self._persistent_request\n"
         "        self.persistent = bool(int(persistent)) and (")]),
    "a method the constructor calls narrows the flag through a local": (_K, [
        ("    def __init__(\n",
         "    def _request(self, requested):\n        narrowed = self.bits == 2\n"
         "        self._persistent_request = requested and narrowed\n\n    def __init__(\n"),
        ("        self.persistent = bool(int(persistent)) and (",
         "        self._request(persistent)\n        persistent = self._persistent_request\n"
         "        self.persistent = bool(int(persistent)) and (")]),
    "a method the constructor calls stores a width-narrowed argument": (_K, [
        ("    def __init__(\n",
         "    def _store(self, value):\n        self._persistent_request = value\n\n"
         "    def __init__(\n"),
        ("        self.persistent = bool(int(persistent)) and (",
         "        self._store(persistent and self.bits == 2)\n"
         "        persistent = self._persistent_request\n"
         "        self.persistent = bool(int(persistent)) and (")]),
    "a class-qualified call stores a width-narrowed argument": (_K, [
        ("    def __init__(\n",
         "    def _store(self, value):\n        self._persistent_request = value\n\n"
         "    def __init__(\n"),
        ("        self.persistent = bool(int(persistent)) and (",
         "        IntDecodeSm100._store(self, persistent and self.bits == 2)\n"
         "        persistent = self._persistent_request\n"
         "        self.persistent = bool(int(persistent)) and (")]),
    "a call through `cls` stores a width-narrowed argument": (_K, [
        ("    def __init__(\n",
         "    def _store(self, value):\n        self._persistent_request = value\n\n"
         "    def __init__(\n"),
        ("        self.persistent = bool(int(persistent)) and (",
         "        cls = type(self)\n        cls._store(self, persistent and self.bits == 2)\n"
         "        persistent = self._persistent_request\n"
         "        self.persistent = bool(int(persistent)) and (")]),
    "a property computes persistence from the width": (_K, [
        ("    def __init__(\n",
         "    @property\n    def _r(self):\n        return self.bits == 2\n\n    def __init__(\n"),
        ("        self.persistent = bool(int(persistent)) and (",
         "        self.persistent = bool(int(persistent)) and self._r and (")]),
    "a variadic setter stores the width for the requested flag": (_K, [
        ("    def __init__(\n",
         "    def _store(self, *args):\n        self._persistent_request = args[0]\n\n"
         "    def __init__(\n"),
        ("        self.persistent = bool(int(persistent)) and (",
         "        self._store(self.bits == 2)\n        persistent = self._persistent_request\n"
         "        self.persistent = bool(int(persistent)) and (")]),
    "a dynamic write narrows the flag after it is set": (_K, [(
        "        self.persistent_sm_count = int(persistent_sm_count)\n",
        "        setattr(self, \"persistent\", self.persistent and bits == 2)\n"
        "        self.persistent_sm_count = int(persistent_sm_count)\n")]),
    "a __dict__ write narrows the flag after it is set": (_K, [(
        "        self.persistent_sm_count = int(persistent_sm_count)\n",
        "        self.__dict__[\"persistent\"] = self.persistent and bits == 2\n"
        "        self.persistent_sm_count = int(persistent_sm_count)\n")]),
    "a class-level property replaces the flag": (_K, [(
        "    def __init__(\n",
        "    persistent = property(lambda self: self.bits == 2)\n\n    def __init__(\n")]),
    "the class attribute is replaced through the instance's class": (_K, [(
        "        self.persistent_sm_count = int(persistent_sm_count)\n",
        "        type(self).persistent = property(lambda obj: obj.bits == 2)\n"
        "        self.persistent_sm_count = int(persistent_sm_count)\n")]),
    "a vars() write narrows the flag after it is set": (_K, [(
        "        self.persistent_sm_count = int(persistent_sm_count)\n",
        "        vars(self)[\"persistent\"] = self.persistent and bits == 2\n"
        "        self.persistent_sm_count = int(persistent_sm_count)\n")]),
    "the constructor rebinds a builtin the flag is computed with": (_K, [(
        "        self.persistent = bool(int(persistent)) and (",
        "        bool = lambda value: value and bits == 2\n"
        "        self.persistent = bool(int(persistent)) and (")]),
    "a match capture in the constructor rebinds a builtin": (_K, [(
        "        self.persistent = bool(int(persistent)) and (",
        "        match (lambda value: value and bits == 2):\n            case bool:\n"
        "                pass\n"
        "        self.persistent = bool(int(persistent)) and (")]),
    "a match capture at module scope rebinds a builtin": (_K, [(
        "from fosca._sm100.int_fold_asm import (",
        "match (lambda value: value):\n    case bool:\n        pass\n\n"
        "from fosca._sm100.int_fold_asm import (")]),
    "a __bool__ hook reached through bool(self)": (_K, [
        ("    def __init__(\n",
         "    def __bool__(self):\n        return self.bits == 2\n\n    def __init__(\n"),
        ("        self.persistent = bool(int(persistent)) and (",
         "        self.persistent = bool(int(persistent)) and bool(self) and (")]),
    "the constructor's persistence depends on the width": (_K, [(
        "        self.persistent = bool(int(persistent)) and (",
        "        self.persistent = bool(int(persistent)) and bits == 2 and (")]),
    "the fresh read is not side-effecting": (_A, [(
        '"mov.u32 $0, %tid.x;", "=r",\n        has_side_effects=True',
        '"mov.u32 $0, %tid.x;", "=r",\n        has_side_effects=False')]),
}


@pytest.mark.parametrize("name", sorted(ACCEPTED_REWRITES))
def test_the_rule_accepts_a_width_independent_rewrite(name):
    src = KERNEL.read_text()
    for old, new in ACCEPTED_REWRITES[name]:
        assert src.count(old) == 1, f"rewrite site for {name!r} found {src.count(old)} times"
        src = src.replace(old, new)
    check_rule(src)


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_the_rule_rejects_a_mutation(name):
    which, edits = MUTATIONS[name]
    path, check = {_K: (KERNEL, check_rule), _A: (ASM, check_helpers),
                   _I: (KERNEL, check_imports)}[which]
    src = path.read_text()
    for old, new in edits:
        assert src.count(old) == 1, f"mutation site for {name!r} found {src.count(old)} times"
        src = src.replace(old, new)
    with pytest.raises(AssertionError):
        check(src)
