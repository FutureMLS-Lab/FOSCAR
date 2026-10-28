# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""CPU gates for IntPingPongDecodeSm90's 64-bit global offsets.

The pingpong decode indexed its global tensors with 32-bit coordinates, and a static stride
times a 32-bit coordinate is a 32-bit product, so every offset that multiplies the request, the
split or the physical page by a compile-time stride wrapped once it passed 2^31 elements (codex,
2026-09-25, round 6 findings 3 and 4; inherited from e4ab066). The production compile passes the
split and block-table strides as runtime 64-bit values, so those two products were already wide
there, though not in a compile with static strides. At HKV = 8 the channel-wise scale reads cross 2^31 at
page 524,288 with a 512-element scale block and inside page 699,050 with a 384-element one, and
the query rows cross it at batch 4096 with HQ = 4096 (HKV = 512 at the largest GQA ratio, 8) and
D = 128, and at batch 2048 at D = 256. `IntPingPongDecodeSm90.kernel` now widens that coordinate to
64 bits wherever a global tensor is indexed, on the whole-D body and on the grouped body alike.

Two things are checked here without a GPU.

  (1) The kernel source, parsed rather than imported, by a syntactic rule over the device code
      (`kernel` and the `cute.jit` methods but the host `__call__`):
        * every parameter of `kernel` is classified below, as a global tensor or not;
        * every subscript of a global tensor whose stride multiplies the request, the split or
          the page wraps that coordinate in `Int64(...)` of a plain coordinate (a name, or a name
          plus a name or a constant), or names a value bound to one, or keeps the dimension in a
          slice (`None`) that is then held to the same rule; `cute.domain_offset(coord, t)`, which
          moves a tracked tensor's origin to `coord`, is a subscript of `t` at `coord` in every
          respect, and the view it returns is held to the same rule;
        * no other coordinate of those subscripts carries a value derived, in the same method,
          from `batch_idx`, `split_idx`, a read or a view of the block table, or a name the
          method itself widens, inline or through a name bound to `Int64(...)`, at a request,
          split or page position of a global tensor or of a view of one (the combine's split
          counter `o`, for example), except the block-table column, which is the tile index
          within the request, in `mPT` or in a view of it;
        * the TMA tensors are only sliced to a kv head, `mSL` is indexed by `batch_idx` itself,
          and `mCnt` only at `cnt_index`, which `kernel` forms as
          `Int64(batch_idx) * kv_head_count + kv_head_idx`;
        * no global tensor reaches the device code under another name, as a raw pointer other
          than the counter atomic's, or through an alias.
  (2) The arithmetic at each boundary. The offset of the boundary coordinate is evaluated with
      torch int32 inputs, whose arithmetic wraps the way the device's does, widening exactly the
      positions (1) found widened in the source. It must be exact, and the same inputs without
      the widening must wrap.

WHAT THE RULE CANNOT SEE. It is syntactic, and it follows a request, split or page value only along
the routes named here: assignments, tuple elements, loop targets, the parameters of device methods,
and stores into a tensor's elements (`r[0] = page`, `r[0] += page`, `r.store(page)`,
`cute.autovec_copy(page_tensor, r)`). A store counts whether it is made through `r` itself or
through a view of `r`, inline or bound to a name: a slice, an origin offset, a recast, a load, a
tensor made on `r`'s pointer or on an offset of it (`cute.make_tensor(r.iterator.align(n),
r.layout)`), a view held in a tuple bound to a name in the same method, directly or through a
conditional (a name taken from the tuple is a view of every tensor in it), or a device method that
`r` is passed to and that stores into the elements of that parameter or of a view of it. Two routes
through tuples are not followed: a subscript of a tuple display written inline (`(r,)[0][0] =
page`), and a tensor that a device method takes out of a tuple parameter. A value read back, through
any of those views or a conversion of a load (`r.load().to(T)`), is judged by what was stored, and a
sum or difference of loaded values is plain only when both operands are, while any other arithmetic
on them is not. A tensor reached by any other route, such as the result of a call not named here
(`cute.where`, a reduction), is taken as data. The rule does not follow the value a function
returns, inside the class or out: a method that reads the block index itself and returns a flattened
product defeats it, and so does a helper outside the class.
Its taint is flow-insensitive, so a name bound to a request value anywhere in a method counts as one
everywhere in it: use distinct names. It judges a tuple as a whole, nested and conditional ones
included, so a coordinate taken from a tuple that holds a product anywhere is refused even when the
element taken is plain: name the coordinate, or unpack it from a display, instead. And it cannot see
a widened coordinate the DSL narrows again, or a stride other than the one the layout declares.
`scripts/test_decode_wide_offsets.py` decodes past the scale and query boundaries on an H100 for
those.

Run: pytest tests/test_decode_wide_offsets_cpu.py
"""
import ast
import importlib.util
import pathlib
import re

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNEL = ROOT / "src/cutedsl_decode/_sm90/int_decode_pingpong.py"
CLASS = "IntPingPongDecodeSm90"

#: Global tensor -> (rank, the coordinate positions that carry the request, split or page).
WIDE = {
    "mQ": (3, (0,)),          # [B, HQ, D]
    "mOut": (3, (0,)),        # [B, HQ, D]
    "mOp": (4, (0, 1)),       # [OUTER, B, HQ, D]
    "mLSE": (3, (0, 1)),      # [OUTER, B, HQ]
    "mPT": (2, (0,)),         # [B, maxPages]
    "mScaleRaw": (3, (0,)),   # [NP, HKV, block_tot]
}
#: Positions that may carry a split-derived value although they are not widened: the block
#: table's column is the tile index within the request, below maxPages, in `mPT` itself or in any
#: view of it (`_root_position`).
SPLIT_BOUNDED = {("mPT", 1)}
#: Read only through TMA, which takes the page as a coordinate of the copy and multiplies nothing
#: here. Their subscripts slice out one kv head's coordinate view, and nothing else.
TMA = {"mKq", "mVq", "mScale"}
#: Rank-1 and unit stride: the index is the request itself, below 65,535 by the grid limit, so it
#: is not a product.
REQUEST_INDEXED = {"mSL"}
#: Indexed only by `cnt_index`, which the kernel forms from a widened request.
COUNTER = {"mCnt"}
GLOBALS = set(WIDE) | TMA | REQUEST_INDEXED | COUNTER
#: The rest of `kernel`'s parameters: scalars, shared-memory layouts and TMA atoms.
NOT_TENSORS = {"softmax_scale", "output_scale", "merge_head_limit_dyn", "sQ_layout", "sP_layout",
               "tma_atom_k_codes", "tma_atom_v_codes", "tma_atom_quant_params"}
#: The names a request or split value starts from; a block-table read starts a page value.
BATCH_SEEDS, SPLIT_SEEDS = {"batch_idx"}, {"split_idx"}
#: The widened positions that index the split, rather than the request or the page: a name inside
#: their widening, in the tensor itself or in any view of it, is a split value (`_coordinate_seeds`).
SPLIT_POSITIONS = {("mOp", 0), ("mLSE", 0)}


def _is_int64(node):
    """`Int64(x)` or `cutlass.Int64(x)`."""
    if not (isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords):
        return False
    f = node.func
    return ((isinstance(f, ast.Name) and f.id == "Int64")
            or (isinstance(f, ast.Attribute) and f.attr == "Int64"
                and isinstance(f.value, ast.Name) and f.value.id == "cutlass"))


_PLAIN_CALLS = {"Int32", "Int64", "block_idx", "grid_dim", "thread_idx", "warp_idx",
                "make_warp_uniform", "min", "max", "range", "range_constexpr"}


#: What a device method stores into the elements of a tensor its caller passes it, carried back
#: into the caller's record of that tensor (`_out_stores`) as a stand-in value of the same kind:
#: a request value, a split value, or a product.
_MARKS = {"request": ast.Name(id="batch_idx", ctx=ast.Load()),
          "split": ast.Name(id="split_idx", ctx=ast.Load()),
          "product": ast.BinOp(left=ast.Constant(2), op=ast.Mult(), right=ast.Constant(2))}


def _flat(display):
    for e in display.elts:
        if isinstance(e, (ast.Tuple, ast.List)):
            yield from _flat(e)
        else:
            yield e


#: Per-report memo of `_definitions`, `_call_bindings`, `_derived` and `_coordinate_seeds`, keyed
#: by the method node's id; the two report entry points clear it, so an id is never reused across
#: parses within it.
_MEMO = {}


def _memo(kind, method, key, compute):
    k = (kind, id(method), key)
    if k not in _MEMO:
        _MEMO[k] = compute()
    return _MEMO[k]


def _definitions(method, marks=None):
    key = tuple(sorted((name, tuple(sorted(kinds))) for name, kinds in (marks or {}).items()))
    return _memo("defs", method, key, lambda: _definitions_of(method, marks))


def _definitions_of(method, marks=None):
    """name -> the expressions it is assigned from in `method` (a tuple unpack gives each name
    the whole right-hand side, an augmented assignment its update, a loop target its iterable).

    The values stored into a tensor's elements, by `t[i] = v`, `t[i] += v`, `t.store(v)` or
    `cute.autovec_copy(v, t)`, directly or through any view of `t` that `_storage` resolves, are
    recorded under the key `t[]`, and a local name bound to such a view (`u = t`, `u = t[None]`,
    `u = cute.domain_offset(c, t)`, `u = cute.recast_tensor(t, T)`, `u = t.load()`) shares them
    with `t` both ways, because the two name the same storage or values: an element read back is then judged by what was
    stored, both for the request, split and page taint and for whether a widened value is plain.

    A name bound to a tuple or list display (or to a conditional between displays, or to a name
    bound to either), to an element of one, unpacked from one or looping over one, is a view of
    every tensor the display holds a view of (`u = (t[None],)[0]`), because which element it names
    is not followed. `marks` (storage name -> kinds) adds the stand-ins
    `_out_stores` carries back from the device methods this one passes a tensor to."""
    defs = {}
    elements, views, bindings = {}, [], []
    for node in ast.walk(method):
        if isinstance(node, ast.Assign):
            pairs = []
            for t in node.targets:
                if (isinstance(t, (ast.Tuple, ast.List)) and isinstance(node.value, (ast.Tuple, ast.List))
                        and len(t.elts) == len(node.value.elts)):
                    pairs += list(zip(t.elts, node.value.elts))
                else:
                    pairs.append((t, node.value))
        elif isinstance(node, ast.AugAssign):
            pairs = [(node.target, ast.BinOp(left=node.target, op=node.op, right=node.value))]
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            pairs = [(node.target, node.value)]
        elif isinstance(node, (ast.For, ast.comprehension)):
            pairs = [(node.target, node.iter)]
        else:
            continue
        for target, value in pairs:
            bindings.append((target, value))
            # only names the statement binds: not the names inside a subscript it stores into
            for n in ast.walk(target):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    defs.setdefault(n.id, []).append(value)
                elif isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) and _storage(n) is not None:
                    elements.setdefault(_storage(n), []).append(value)      # `t[i] = v`, through any view of t
            if isinstance(target, ast.Name) and _view_of(value) is not None:
                views.append((target.id, _view_of(value)))                  # `u = t`, `u = t[None]`, `u = cute.recast_tensor(t, T)`, ...
    for node in ast.walk(method):
        f = node.func if isinstance(node, ast.Call) else None
        if isinstance(f, ast.Attribute) and f.attr == "store" and len(node.args) == 1:
            base = _storage(f.value)                                        # `t.store(v)`, through any view of t
        elif ast.unparse(f) == "cute.autovec_copy" if f is not None else False:
            base = _storage(node.args[-1]) if len(node.args) == 2 else None # `cute.autovec_copy(src, t)`
        else:
            continue
        if base is not None:
            elements.setdefault(base, []).append(node.args[0])

    def displays(expr, seen=frozenset()):
        """The tuple or list displays `expr` may be: itself, either branch of a conditional, or
        what a name is bound to, followed through names."""
        if isinstance(expr, (ast.Tuple, ast.List)):
            return [expr]
        if isinstance(expr, ast.IfExp):
            return displays(expr.body, seen) + displays(expr.orelse, seen)
        if isinstance(expr, ast.Name) and expr.id not in seen:
            return [d for v in defs.get(expr.id, []) for d in displays(v, seen | {expr.id})]
        return []
    for target, value in bindings:
        names = [n.id for n in ast.walk(target) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
        base = value
        while isinstance(base, ast.Subscript):
            base = base.value                                               # `u = pair[0]`, `u = pair[0][1]`
        sources = displays(base)
        for display in sources:
            for e in _flat(display):
                if _view_of(e) is not None:
                    views += [(n, _view_of(e)) for n in names]
    for storage, kinds in (marks or {}).items():
        elements.setdefault(storage, []).extend(_MARKS[k] for k in sorted(kinds))
    changed = True
    while changed:                      # a view and its tensor share their stored values
        changed = False
        for view, base in views:
            union = elements.get(view, []) + [v for v in elements.get(base, []) if v not in elements.get(view, [])]
            for name in (view, base):
                if len(elements.get(name, [])) != len(union):
                    elements[name] = list(union); changed = True
    for name, values in elements.items():
        defs[name + "[]"] = values
    return defs


def _plain_value(node, defs, seen=frozenset(), not_plain=frozenset()):
    """True when `node` is computed by additions alone from names, constants, tensor reads and
    the index intrinsics: no product, shift or division anywhere in its definition, followed
    through every assignment in the method. A product may already have wrapped before it is
    widened, so only a plain value may be."""
    if isinstance(node, ast.Constant):
        # an integer, or None (a slice marker, never a product)
        return node.value is None or (isinstance(node.value, int) and not isinstance(node.value, bool))
    def rec(n, sn=seen):
        return _plain_value(n, defs, sn, not_plain)
    if isinstance(node, ast.Name):
        if node.id in not_plain:
            return False                      # a parameter some call site passes a product
        if node.id in seen:
            return True
        return all(rec(v, seen | {node.id}) for v in defs.get(node.id, []))
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(rec(e) for e in node.elts)
    if isinstance(node, ast.Subscript):
        # an element read of a tensor is data; an element of a tuple is judged by the whole tuple
        return _subscriptable_plain(node.value, defs, seen, not_plain)
    if isinstance(node, ast.BinOp):
        return isinstance(node.op, (ast.Add, ast.Sub)) and rec(node.left) and rec(node.right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return rec(node.operand)
    if isinstance(node, ast.IfExp):
        return rec(node.body) and rec(node.orelse)
    if isinstance(node, ast.Call):
        f = node.func
        fname = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        return fname in _PLAIN_CALLS and not node.keywords and all(rec(a) for a in node.args)
    return False


def _subscriptable_plain(base, defs, seen, not_plain):
    """Whether indexing `base` gives a plain value. A tuple or list is judged AS A WHOLE, every
    element plain or none of it: a name is followed through its aliases to such displays, and a
    parameter some call site binds to a value that is not plain (`not_plain`) is refused, so
    unpacking or forwarding a tuple cannot shed a product it holds. Anything else (a tensor, a
    call's result) is taken as data, except that a tensor's elements are judged by the values stored
    into them (`_definitions`). A conversion (`x.to(T)`) holds the values it converts, and a sum or
    difference of tensor values is plain when both operands are; any other arithmetic is not."""
    while isinstance(base, ast.Call) and isinstance(base.func, ast.Attribute) and base.func.attr == "to" \
            and len(base.args) == 1:
        base = base.func.value
    if isinstance(base, ast.BinOp):
        return (isinstance(base.op, (ast.Add, ast.Sub)) and _subscriptable_plain(base.left, defs, seen, not_plain)
                and _subscriptable_plain(base.right, defs, seen, not_plain))
    if isinstance(base, ast.UnaryOp):
        return isinstance(base.op, ast.USub) and _subscriptable_plain(base.operand, defs, seen, not_plain)
    if isinstance(base, (ast.Tuple, ast.List)):
        return all(_plain_value(e, defs, seen, not_plain) for e in base.elts)
    if isinstance(base, ast.Subscript):          # an element of an element: judge the outer whole
        return _subscriptable_plain(base.value, defs, seen, not_plain)
    if isinstance(base, ast.IfExp):
        return (_subscriptable_plain(base.body, defs, seen, not_plain)
                and _subscriptable_plain(base.orelse, defs, seen, not_plain))
    if not isinstance(base, ast.Name) and _storage(base) is not None:
        # an element of a view or a load is an element of its tensor
        return _subscriptable_plain(ast.Name(id=_storage(base), ctx=ast.Load()), defs, seen, not_plain)
    if not isinstance(base, ast.Name):
        return True
    if base.id in not_plain:
        return False
    if base.id in seen:
        return True
    return (all(_subscriptable_plain(v, defs, seen | {base.id}, not_plain) for v in defs.get(base.id, []))
            and all(_plain_value(v, defs, seen | {base.id}, not_plain) for v in defs.get(base.id + "[]", [])))


def _plain_coordinate(node, defs, not_plain=frozenset()):
    """What may be widened: a name, or a name plus a name or an integer, whose definitions are
    themselves plain values (and, for a parameter, whose every call site passes one)."""
    shape = isinstance(node, ast.Name) or (
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
        and isinstance(node.left, ast.Name) and isinstance(node.right, (ast.Name, ast.Constant)))
    return shape and _plain_value(node, defs, frozenset(), not_plain)


def _call_bindings(fn, methods):
    """(callee name, parameter, argument) for every call of a device method in `fn`, positional
    arguments bound in order to the callee's positional-only and positional parameters."""
    return _memo("calls", fn, tuple(sorted(methods)), lambda: _call_bindings_of(fn, methods))


def _call_bindings_of(fn, methods):
    out = []
    for call in ast.walk(fn):
        f = call.func if isinstance(call, ast.Call) else None
        if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id == "self" and f.attr in methods):
            continue
        a = methods[f.attr].args
        positional = [x.arg for x in a.posonlyargs + a.args][1:]      # drop self
        out += [(f.attr, prm, arg) for prm, arg in zip(positional, call.args)]
        out += [(f.attr, k.arg, k.value) for k in call.keywords if k.arg]
    return out


def _not_plain_params(methods, marks=None):
    """method -> the parameters some call site passes a value that is not plain, to a fixed
    point across the device methods."""
    bad = {m: set() for m in methods}
    defs = {m: _definitions(fn, (marks or {}).get(m)) for m, fn in methods.items()}
    while True:
        grew = False
        for mname, fn in methods.items():
            for callee, prm, arg in _call_bindings(fn, methods):
                if prm not in bad[callee] and not _plain_value(arg, defs[mname], frozenset(),
                                                               frozenset(bad[mname])):
                    bad[callee].add(prm); grew = True
        if not grew:
            return bad


def _stores(method):
    out = {}
    for node in ast.walk(method):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.setdefault(node.id, []).append(node)
    return out


def _widened_names(method, defs, stores, not_plain=frozenset()):
    """Names bound exactly once, to `Int64(x)` of a plain coordinate: they count as widened."""
    return {n.targets[0].id for n in ast.walk(method)
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name) and len(stores.get(n.targets[0].id, ())) == 1
            and _is_int64(n.value) and _plain_coordinate(n.value.args[0], defs, not_plain)}


def _all_params(fn):
    a = fn.args
    return [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs] + \
        [x.arg for x in (a.vararg, a.kwarg) if x is not None]


def _is_none(node):
    return isinstance(node, ast.Constant) and node.value is None


def _device_methods(cls):
    """`kernel` and every `cute.jit` method but the host-side `__call__`."""
    out = {}
    for m in cls.body:
        if not isinstance(m, ast.FunctionDef) or m.name == "__call__":
            continue
        decs = {ast.unparse(d) for d in m.decorator_list}
        if decs & {"cute.jit", "cute.kernel"}:
            out[m.name] = m
    return out


def _origin_offset(value):
    """`cute.domain_offset(coord, t)` with a tuple `coord` and a named `t`: returns (t, coord),
    else None. It is `t` with its origin moved to `coord`, a view of the same rank whose element
    `(0, ..., 0)` is `t[coord]`, so `coord` is an index of `t` and is checked as one."""
    if (isinstance(value, ast.Call) and not value.keywords and len(value.args) == 2
            and ast.unparse(value.func) == "cute.domain_offset"
            and isinstance(value.args[0], ast.Tuple) and isinstance(value.args[1], ast.Name)):
        return value.args[1].id, value.args[0]
    return None


def _storage(expr):
    """The named tensor whose storage (or whose elements' values) `expr` reads or writes, else
    None, through any composition of: a subscript or slice (`t[i]`, `t[None]`), an attribute
    (`t.iterator`), a load (`t.load()`), an origin offset (`cute.domain_offset(c, t)`), a recast
    (`cute.recast_tensor(t, T)`, `cute.recast_ptr(p, ...)`), a pointer's alignment (`p.align(n)`)
    or offset (`p + k`), and a tensor made on a pointer (`cute.make_tensor(p, layout)`). A
    conversion of loaded values (`t.load().to(T)`) is not storage; `_subscriptable_plain` judges
    it by the values it converts."""
    while True:
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, (ast.Subscript, ast.Attribute)):
            expr = expr.value
        elif isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub)):
            expr = expr.left
        elif isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and (
                (expr.func.attr == "load" and not expr.args)
                or (expr.func.attr == "align" and len(expr.args) == 1)):
            expr = expr.func.value
        elif isinstance(expr, ast.Call) and ast.unparse(expr.func) in ("cute.make_tensor", "cute.recast_ptr") \
                and expr.args:
            expr = expr.args[0]
        elif isinstance(expr, ast.Call) and ast.unparse(expr.func) == "cute.domain_offset" and len(expr.args) == 2:
            expr = expr.args[1]
        elif isinstance(expr, ast.Call) and ast.unparse(expr.func) == "cute.recast_tensor" and expr.args:
            expr = expr.args[0]
        else:
            return None


def _view_of(value):
    """The named tensor whose storage a view-making expression names, else None: an alias, a
    pointer (`t.iterator`) or an offset of one (`p + k`), or any subscript or call `_storage`
    resolves other than an element read (a slice, a view call, a load, a tensor made on a pointer,
    or a composition of them)."""
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute) and value.attr == "iterator":
        return _storage(value)                       # a pointer to the tensor's storage
    if isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Sub)):
        return _view_of(value.left)                  # a pointer offset, `p = t.iterator + k`
    if isinstance(value, ast.Subscript) and not any(
            _is_none(e) for e in (value.slice.elts if isinstance(value.slice, ast.Tuple) else [value.slice])):
        return None                                  # an element read is a value, not a view
    return _storage(value) if isinstance(value, (ast.Subscript, ast.Call)) else None


def _derived_tensors(method, tracked_wide):
    """Tensors a method slices out of a tracked one: `g = mQ[(Int64(batch_idx), None, None)]`.

    A `None` keeps that dimension in the slice, and the slice inherits the widening duty of every
    request, split or page dimension it keeps: a slice `mLSE[(None, Int64(batch_idx), None)]`
    would still have the split as its coordinate 0. Returns name -> (rank, wide positions,
    defining Assign, root, root positions): the root is the global tensor the view was taken from,
    through any chain of views, and root position j is the root's position that the view's
    position j indexes, so a view's split positions and its block-table column are its root's
    (`_root_position`). A slice that keeps none of those dimensions is a bounded row. A view bound
    from `cute.domain_offset(coord, t)` keeps every dimension of `t`, and with them its duties.
    """
    derived = {}

    def spec_of(parent):
        """(rank, wide positions, root, root positions) of a tracked tensor or view, else None."""
        if parent in tracked_wide:
            rank, wide = tracked_wide[parent]
            return rank, wide, parent, tuple(range(rank))
        if parent in derived:
            rank, wide, _, root, root_pos = derived[parent]
            return rank, wide, root, root_pos
        return None

    changed = True
    while changed:
        changed = False
        for node in ast.walk(method):
            moved = (_origin_offset(node.value) if isinstance(node, ast.Assign) and len(node.targets) == 1
                     and isinstance(node.targets[0], ast.Name) else None)
            if moved is not None:
                spec = spec_of(moved[0])
                name = node.targets[0].id
                if spec is not None and name not in derived:
                    rank, wide, root, root_pos = spec
                    derived[name] = (rank, wide, node, root, root_pos)
                    changed = True
                continue
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Subscript)
                    and isinstance(node.value.value, ast.Name)):
                continue
            parent, name = node.value.value.id, node.targets[0].id
            spec = spec_of(parent)
            if spec is None or name in derived:
                continue
            idx = node.value.slice   # a bare index is a one-element coordinate: `row[None]`
            kept = [i for i, e in enumerate(idx.elts if isinstance(idx, ast.Tuple) else [idx]) if _is_none(e)]
            if kept:
                _, wide_p, root, root_pos = spec
                wide = tuple(j for j, i in enumerate(kept) if i in wide_p)
                # a subscript longer than the parent's rank is refused in the report; map its
                # extra positions to nothing here
                derived[name] = (len(kept), wide, node, root,
                                 tuple(root_pos[i] if i < len(root_pos) else None for i in kept))
                changed = True
    return derived


def _derived(method):
    return _memo("derived", method, (), lambda: _derived_tensors(method, WIDE))


def _root_position(name, p, derived):
    """(global tensor, position) that position `p` of the tensor or view `name` indexes, else None:
    `(name, p)` for a global tensor, and for a view the root and root position `_derived_tensors`
    recorded, so a split position and the block-table column are recognized through any view."""
    if name in WIDE:
        return name, p
    if name in derived and p < len(derived[name][4]):
        return derived[name][3], derived[name][4][p]
    return None


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _value_names(node):
    """The names an expression's VALUE is computed from. A tensor element read at an index is
    data, not a product of the index (`seq_len = mSL[batch_idx]` is a length), so the names in
    a subscript's index do not count; a block-table read is a page value and is seeded apart."""
    out, stack = set(), [node]
    while stack:
        n = stack.pop()
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Subscript):
            stack.append(n.value)
        else:
            stack.extend(ast.iter_child_nodes(n))
    return out


def _reads_block_table(node):
    """Whether `node` subscripts `mPT`: an element read or a slice, so a name bound to either is a
    request-or-page value by its definition. An origin offset of `mPT` needs no case here: the
    rule accepts it only with its request coordinate widened, and the widening seeds the names in
    that coordinate (`_coordinate_seeds`), which the view's definition names."""
    return any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "mPT"
               for n in ast.walk(node))


def _coordinate_seeds(method):
    return _memo("seeds", method, (), lambda: _coordinate_seeds_of(method))


def _coordinate_seeds_of(method):
    """(request-or-page names, split names) that `method` itself names as such: every name inside
    the widening at a widened position of a subscript or an origin offset, of a global tensor or
    of a view of one (`_derived`), whether the widening is written inline, `Int64(x)`, or through a
    name bound to one, `x64 = Int64(x)`, the spelling the rule also accepts. A position is a split
    one when the root position it indexes is in `SPLIT_POSITIONS` (`_root_position`). What indexes
    the split, the request or the page is a value of that kind wherever else the method uses it,
    so a split loop's counter, which reaches `mOp` and `mLSE` only as `Int64(o + kk)`, is a split
    value in every other coordinate too."""
    derived = _derived(method)
    bound = {}
    for n in ast.walk(method):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and _is_int64(n.value)):
            bound.setdefault(n.targets[0].id, []).append(n.value.args[0])
    bp, sp = set(), set()
    for node in ast.walk(method):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            name, idx = node.value.id, node.slice
        else:
            moved = _origin_offset(node)
            if moved is None:
                continue
            name, idx = moved
        wide = WIDE[name][1] if name in WIDE else derived[name][1] if name in derived else ()
        elts = idx.elts if isinstance(idx, ast.Tuple) else [idx]
        for p in wide:
            if p >= len(elts):
                continue
            e = elts[p]
            widened = [e.args[0]] if _is_int64(e) else bound.get(e.id, []) if isinstance(e, ast.Name) else []
            kind = sp if _root_position(name, p, derived) in SPLIT_POSITIONS else bp
            for w in widened:
                kind.update(_value_names(w))
    return bp, sp


def _tainted(method, seeds_bp=(), seeds_sp=(), marks=None):
    """(request-or-page names, split names) of one method: the seeds, the names the method widens
    as such (`_coordinate_seeds`), a read or a slice of the block table (`_reads_block_table`), and
    every name bound (by an assignment, an augmented assignment or a loop target) from an
    expression that mentions one, to a fixed point. It is flow-insensitive: a name bound to a
    request value anywhere in the method is treated as one everywhere in it."""
    own_bp, own_sp = _coordinate_seeds(method)
    bp = set(BATCH_SEEDS) | set(seeds_bp) | own_bp
    sp = set(SPLIT_SEEDS) | set(seeds_sp) | own_sp
    defs = _definitions(method, marks)
    changed = True
    while changed:
        changed = False
        for key, values in defs.items():
            name = key[:-2] if key.endswith("[]") else key      # a stored element taints its tensor
            for v in values:
                used = _value_names(v)
                if (used & bp or _reads_block_table(v)) and name not in bp:
                    bp.add(name); changed = True
                if used & sp and name not in sp:
                    sp.add(name); changed = True
    return bp, sp


def _taint_all(methods, marks=None):
    """Per method (bp, sp), with a device method's parameter seeded when any call site passes
    it a tainted value, whatever the parameter is called."""
    seeds = {m: (set(), set()) for m in methods}
    while True:
        taint = {m: _tainted(fn, *seeds[m], (marks or {}).get(m)) for m, fn in methods.items()}
        grew = False
        for mname, fn in methods.items():
            bp, sp = taint[mname]
            for callee, param, value in _call_bindings(fn, methods):
                used = _value_names(value)
                if (used & bp or _reads_block_table(value)) and param not in seeds[callee][0]:
                    seeds[callee][0].add(param); grew = True
                if used & sp and param not in seeds[callee][1]:
                    seeds[callee][1].add(param); grew = True
        if not grew:
            return taint


def _out_stores(methods):
    """method -> {storage name: kinds}: what the device methods a method calls store into the
    elements of a tensor it passes them, to a fixed point. A callee's stores into a parameter
    are stores into the argument's storage (`_storage`), so each kind of value they hold (a request
    or page value, a split value, a value that is not plain) is carried back into the caller's
    record of that storage as the stand-in `_MARKS` gives it, where the caller's views share it.
    Returns the marks with the taint (`_taint_all`) and the refused parameters (`_not_plain_params`)
    they give."""
    marks = {m: {} for m in methods}
    while True:
        taint, not_plain = _taint_all(methods, marks), _not_plain_params(methods, marks)
        defs = {m: _definitions(fn, marks[m]) for m, fn in methods.items()}
        grew = False
        for mname, fn in methods.items():
            for callee, prm, arg in _call_bindings(fn, methods):
                stored, target = defs[callee].get(prm + "[]", []), _storage(arg)
                if not stored or target is None:
                    continue
                bp, sp = taint[callee]
                kinds = set()
                if any(_value_names(v) & bp or _reads_block_table(v) for v in stored):
                    kinds.add("request")
                if any(_value_names(v) & sp for v in stored):
                    kinds.add("split")
                if not all(_plain_value(v, defs[callee], frozenset(), frozenset(not_plain[callee])) for v in stored):
                    kinds.add("product")
                have = marks[mname].setdefault(target, set())
                if not kinds <= have:
                    have |= kinds; grew = True
        if not grew:
            return marks, taint, not_plain


def global_tensor_report(src):
    """Every global-tensor use in the device code, and every one the rule refuses.

    Returns sites (tensor -> number of subscripts checked, a slice's own subscripts counted
    under the slice's name), violations (one line each), `kernel`'s parameters, and the
    counter-cell expression.
    """
    _MEMO.clear()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CLASS)
    methods = _device_methods(cls)
    kernel = methods["kernel"]
    params_of_kernel = _all_params(kernel)[1:]
    sites = {name: 0 for name in GLOBALS}
    violations = []
    counter_exprs = []
    for p in params_of_kernel:
        if p not in GLOBALS | NOT_TENSORS:
            violations.append(f"kernel: parameter {p} is not classified")

    marks, taint, not_plain = _out_stores(methods)
    for mname, m in methods.items():
        parents = {}
        for node in ast.walk(m):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        derived = _derived(m)
        wide_specs = dict(WIDE)
        wide_specs.update({name: spec[:2] for name, spec in derived.items()})
        defining = {spec[2] for spec in derived.values()}
        bp_names, sp_names = taint[mname]
        defs, stores = _definitions(m, marks[mname]), _stores(m)
        wide_names = _widened_names(m, defs, stores, frozenset(not_plain[mname]))
        tracked = GLOBALS | {"cnt_index"} | set(derived)
        for name in derived:
            sites.setdefault(name, 0)
        # a device method may take a global tensor only under its own name
        for p in _all_params(m):
            if re.fullmatch(r"m[A-Z]\w*", p) and p not in GLOBALS:
                violations.append(f"{mname}: parameter {p} is not a classified global tensor")

        def widened(e, slicing, p):
            if _is_int64(e):
                return (True if _plain_coordinate(e.args[0], defs, frozenset(not_plain[mname]))
                        else f"`{ast.unparse(e)}` widens a computed value")
            if isinstance(e, ast.Name) and e.id in wide_names:
                return True
            if slicing and _is_none(e):
                return True
            return False

        for node in ast.walk(m):
            if isinstance(node, ast.Name) and node.id in tracked and isinstance(node.ctx, ast.Store):
                parent = parents.get(node)
                if (node.id == "cnt_index" and mname == "kernel" and isinstance(parent, ast.Assign)
                        and parent.targets == [node]):
                    counter_exprs.append(parent.value)
                elif node.id in derived and parent is derived[node.id][2]:
                    pass                                   # the slice's own definition
                else:
                    violations.append(f"{mname}:{node.lineno}: rebinds {node.id}")
                continue
            if not (isinstance(node, ast.Name) and node.id in tracked and isinstance(node.ctx, ast.Load)):
                continue
            name, parent = node.id, parents.get(node)
            where = f"{mname}:{node.lineno}"
            moved = _origin_offset(parent) if isinstance(parent, ast.Call) else None
            if moved is not None and parent.args[1] is node and name in wide_specs \
                    and parents.get(parent) in defining:
                # an origin offset bound to a tracked view: its coordinate is an index of `name`
                # and never a slice, so every dimension it covers is checked as a subscript's
                idx, elts = moved[1], moved[1].elts
                rank, wide = wide_specs[name]
                sites[name] += 1
                if len(elts) != rank:
                    violations.append(f"{where}: {name} origin {ast.unparse(idx)} is not a full rank-{rank} coordinate")
                    continue
                for p, e in enumerate(elts):
                    if p in wide:
                        ok = widened(e, False, p)
                        if ok is not True:
                            violations.append(f"{where}: {name} coordinate {p} "
                                              + (ok if ok else f"`{ast.unparse(e)}` is not widened"))
                        continue
                    used = _value_names(e)
                    if used & bp_names or _reads_block_table(e):
                        violations.append(f"{where}: {name} coordinate {p} `{ast.unparse(e)}` "
                                          f"carries a request or page value unwidened")
                    elif used & sp_names and _root_position(name, p, derived) not in SPLIT_BOUNDED:
                        violations.append(f"{where}: {name} coordinate {p} `{ast.unparse(e)}` "
                                          f"carries a split value unwidened")
                continue
            if isinstance(parent, ast.Subscript) and parent.value is node:
                idx = parent.slice
                elts = idx.elts if isinstance(idx, ast.Tuple) else [idx]
                if name in wide_specs:
                    rank, wide = wide_specs[name]
                    sites[name] += 1
                    if len(elts) != rank:
                        violations.append(f"{where}: {name}[{ast.unparse(idx)}] is not a full rank-{rank} index")
                        continue
                    slicing = parents.get(parent) in defining
                    if any(_is_none(e) for e in elts) and not slicing:
                        # a view made inline (an argument, an origin offset's tensor) would escape
                        # the tracking a named view gets
                        violations.append(f"{where}: {name}[{ast.unparse(idx)}] is a slice that is not bound to a name")
                        continue
                    for p, e in enumerate(elts):
                        if p in wide:
                            ok = widened(e, slicing, p)
                            if ok is True:
                                continue
                            violations.append(f"{where}: {name} coordinate {p} "
                                              + (ok if ok else f"`{ast.unparse(e)}` is not widened"))
                            continue
                        used = _value_names(e)
                        if used & bp_names or _reads_block_table(e):
                            violations.append(f"{where}: {name} coordinate {p} `{ast.unparse(e)}` "
                                              f"carries a request or page value unwidened")
                        elif used & sp_names and _root_position(name, p, derived) not in SPLIT_BOUNDED:
                            violations.append(f"{where}: {name} coordinate {p} `{ast.unparse(e)}` "
                                              f"carries a split value unwidened")
                elif name in TMA:
                    sites[name] += 1
                    if not all(_is_none(e) or (isinstance(e, ast.Name) and e.id == "kv_head_idx")
                               for e in elts):
                        violations.append(f"{where}: {name}[{ast.unparse(idx)}] is not a kv-head slice")
                elif name in REQUEST_INDEXED:
                    sites[name] += 1
                    if not (isinstance(idx, ast.Name) and idx.id == "batch_idx"):
                        violations.append(f"{where}: {name}[{ast.unparse(idx)}] is not the request itself")
                elif name in COUNTER:
                    sites[name] += 1
                    if not (isinstance(idx, ast.Name) and idx.id == "cnt_index"):
                        violations.append(f"{where}: {name}[{ast.unparse(idx)}] is not cnt_index")
                else:
                    violations.append(f"{where}: cnt_index is indexed")
                continue
            if name == "cnt_index" and isinstance(parent, ast.Subscript) and parent.slice is node \
                    and isinstance(parent.value, ast.Name) and parent.value.id in COUNTER:
                continue
            # the counter atomic's address, `mCnt.iterator + cnt_index`, and only as a call's
            # direct argument: nothing added to it, nothing bound to it
            counter_ptr = None
            if name in COUNTER and isinstance(parent, ast.Attribute) and parent.attr == "iterator":
                counter_ptr = parents.get(parent)
            elif name == "cnt_index" and isinstance(parent, ast.BinOp):
                counter_ptr = parent
            if (isinstance(counter_ptr, ast.BinOp) and isinstance(counter_ptr.op, ast.Add)
                    and isinstance(counter_ptr.left, ast.Attribute) and counter_ptr.left.attr == "iterator"
                    and isinstance(counter_ptr.left.value, ast.Name) and counter_ptr.left.value.id in COUNTER
                    and isinstance(counter_ptr.right, ast.Name) and counter_ptr.right.id == "cnt_index"
                    and isinstance(parents.get(counter_ptr), ast.Call)
                    and counter_ptr in parents[counter_ptr].args):
                if name in COUNTER:
                    sites[name] += 1
                continue
            # passed on: only to a device method, and only into a parameter of the same name
            if isinstance(parent, ast.Call) and node in parent.args:
                f = parent.func
                target = methods.get(f.attr) if (isinstance(f, ast.Attribute)
                                                 and isinstance(f.value, ast.Name)
                                                 and f.value.id == "self") else None
                if target is not None and name not in derived:
                    pos = parent.args.index(node)
                    tparams = [a.arg for a in target.args.posonlyargs + target.args.args][1:]
                    if pos < len(tparams) and tparams[pos] == name:
                        continue
                    violations.append(f"{where}: {name} is passed to {target.name} as "
                                      f"{tparams[pos] if pos < len(tparams) else '*args'}")
                    continue
            if isinstance(parent, ast.keyword) and parent.arg == name and name not in derived:
                call = parents.get(parent)
                f = call.func if isinstance(call, ast.Call) else None
                if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                        and f.value.id == "self" and f.attr in methods):
                    continue
            violations.append(f"{where}: {name} is used as `{ast.unparse(parent)}`")
    return dict(sites=sites, violations=violations, params=params_of_kernel, counter=counter_exprs)


def _source():
    return KERNEL.read_text()


def _same_code(node, text):
    """AST equality with `text`, independent of how this interpreter's `ast.unparse` spells it."""
    parsed = ast.parse(text).body[0]
    return ast.dump(node) == ast.dump(parsed.value if isinstance(node, ast.expr) else parsed)


_COUNTER = "Int64(batch_idx) * kv_head_count + kv_head_idx"


def test_every_parameter_of_the_kernel_is_classified():
    rep = global_tensor_report(_source())
    assert set(rep["params"]) == GLOBALS | NOT_TENSORS, sorted(set(rep["params"]) ^ (GLOBALS | NOT_TENSORS))


def test_every_indexing_site_widens_the_request_split_and_page():
    rep = global_tensor_report(_source())
    assert rep["violations"] == []
    # every widened tensor is indexed somewhere, so the rule is not passing by finding nothing
    assert all(rep["sites"][name] > 0 for name in WIDE), rep["sites"]


def test_the_counter_cell_is_formed_from_a_widened_request():
    """`cnt_index = Int64(batch_idx) * kv_head_count + kv_head_idx`, assigned once, in `kernel`,
    with `kv_head_count` the grid's kv-head extent."""
    (expr,) = global_tensor_report(_source())["counter"]
    assert _same_code(expr, _COUNTER), ast.unparse(expr)
    tree = ast.parse(_source())
    kernel = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "kernel")
    stores = _stores(kernel)
    for name in ("kv_head_count", "batch_idx", "kv_head_idx", "cnt_index"):
        assert len(stores.get(name, ())) == 1, f"{name} is bound {len(stores.get(name, ()))} times"
    unpacks = [n for n in ast.walk(kernel) if isinstance(n, ast.Assign)
               and _names(n.targets[0]) & {"kv_head_count", "batch_idx", "kv_head_idx"}]
    expected = ["split_count, kv_head_count, _ = cute.arch.grid_dim()",
                "split_idx, kv_head_idx, batch_idx = cute.arch.block_idx()"]
    assert len(unpacks) == 2 and all(any(_same_code(n, e) for n in unpacks) for e in expected), \
        [ast.unparse(n) for n in unpacks]


# The negative controls. Each mutation undoes the fix, or hides a tensor, in a copy of the source.

_WIDENED_ARGS = ("batch_idx", "split_idx", "v_page_pf", "pf_page", "o + kk")
_WIDENED = re.compile(r"Int64\(((?:batch_idx|split_idx|v_page_pf|pf_page|o \+ kk))\)")


def _widened_calls(src):
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CLASS)
    return sum(1 for m in _device_methods(cls).values() for node in ast.walk(m)
               if _is_int64(node) and ast.unparse(node.args[0]) in _WIDENED_ARGS)


def _mutate(old, new):
    src = _source()
    assert src.count(old) == 1, f"the mutation target moved: {old!r}"
    return global_tensor_report(src.replace(old, new))


def test_the_rule_flags_every_site_of_the_unwidened_kernel():
    src = _source()
    widened = _widened_calls(src)
    assert widened > 0
    rep = global_tensor_report(_WIDENED.sub(r"\1", src))
    not_widened = [v for v in rep["violations"] if "is not widened" in v]
    # every Int64 the fix added is one violation, except the counter cell's, which is checked apart
    assert len(not_widened) == widened - 1
    for needle in ("mScaleRaw coordinate 0 `v_page_pf`", "mScaleRaw coordinate 0 `pf_page`",
                   "mQ coordinate 0 `batch_idx`", "mOut coordinate 0 `batch_idx`",
                   "mOp coordinate 0 `split_idx`", "mOp coordinate 1 `batch_idx`",
                   "mLSE coordinate 0 `o + kk`", "mLSE coordinate 1 `batch_idx`",
                   "mPT coordinate 0 `batch_idx`"):
        assert any(needle in v for v in not_widened), needle
    (expr,) = rep["counter"]
    assert not _same_code(expr, _COUNTER)


def test_the_rule_flags_a_tensor_passed_under_another_name():
    rep = _mutate("mQ,\n        sBias,\n        sQ,", "q_rows,\n        sBias,\n        sQ,")
    assert any("mQ is passed to _stage_q as q_rows" in v for v in rep["violations"])


_ENTRY = "        thread_idx, _, _ = cute.arch.thread_idx()\n"


def test_the_rule_flags_a_raw_pointer_and_an_alias():
    rep = _mutate(_ENTRY, _ENTRY + "        q_ptr = mQ.iterator + batch_idx\n        q_alias = mOut\n")
    joined = " | ".join(rep["violations"])
    assert "mQ is used as `mQ.iterator`" in joined and "mOut is used as" in joined


def test_the_rule_flags_an_unclassified_parameter_whatever_its_name():
    rep = _mutate("def kernel(self, mQ, mKq,", "def kernel(self, mQ, qExtra, mKq,")
    assert any("parameter qExtra is not classified" in v for v in rep["violations"])


def test_the_rule_flags_a_flattened_product_in_another_coordinate():
    """A request product moved out of the widened position, into a precomputed flat offset, and a
    product of a block-table read placed in that coordinate directly, which is a page value by
    what it reads (`_reads_block_table`) rather than by any name it mentions."""
    anchor = "        gQ = mQ[(Int64(batch_idx), None, None)]\n"
    rep = _mutate(anchor, anchor + "        q_flat = batch_idx * 65536\n        q0 = mQ[(Int64(0), 0, q_flat)]\n")
    assert any("mQ coordinate 2 `q_flat` carries a request or page value unwidened" in v
               for v in rep["violations"])
    rep = _mutate(anchor, anchor + "        q0 = mQ[(Int64(0), 0, mPT[(Int64(batch_idx), 0)] * 128)]\n")
    assert any("mQ coordinate 2 `mPT[Int64(batch_idx), 0] * 128` carries a request or page value unwidened" in v
               for v in rep["violations"]), rep["violations"]


def test_the_rule_flags_a_product_inside_the_widening():
    rep = _mutate("gQ = mQ[(Int64(batch_idx), None, None)]",
                  "gQ = mQ[(Int64(batch_idx * 65536 // 65536), None, None)]")
    assert any("widens a computed value" in v for v in rep["violations"])


def test_the_rule_flags_a_counter_formula_that_drops_a_term():
    for bad in ("Int64(batch_idx) * 0 + kv_head_idx", "Int64(batch_idx) * kv_head_count + 0"):
        src = _source().replace("cnt_index = Int64(batch_idx) * kv_head_count + kv_head_idx",
                                f"cnt_index = {bad}", 1)
        (expr,) = global_tensor_report(src)["counter"]
        assert not _same_code(expr, _COUNTER)


def test_the_rule_flags_arithmetic_on_the_counter_pointer():
    rep = _mutate("mCnt.iterator + cnt_index, Int32(1)",
                  "(mCnt.iterator + cnt_index) + batch_idx, Int32(1)")
    assert any("is used as" in v for v in rep["violations"])


_DRAIN_PAGE = "gVDrainPage = mScaleRaw[(Int64(v_page_pf), None, None)]"
_DRAIN_FRAG = 'gVDrainFrag = cute.domain_offset((kv_head_idx, self.qoff["v_ch"] + dl_p), gVDrainPage)'


def test_the_rule_holds_an_origin_offset_to_the_subscript_rule():
    """`cute.domain_offset(coord, t)` is a subscript of `t` at `coord`: an unwidened page there is
    refused, a page value moved into another coordinate is refused, whether it is a name bound to
    one or a block-table read placed there directly, a coordinate short of the rank is refused, and
    the view it returns is tracked, so a raw pointer taken from it is refused as well."""
    rep = _mutate(_DRAIN_PAGE, "gVDrainPage = cute.domain_offset((v_page_pf, 0, 0), mScaleRaw)")
    assert any("mScaleRaw coordinate 0 `v_page_pf` is not widened" in v for v in rep["violations"])
    rep = _mutate(_DRAIN_FRAG, _DRAIN_FRAG.replace('self.qoff["v_ch"] + dl_p', "v_page_pf * 512"))
    assert any("gVDrainPage coordinate 1 `v_page_pf * 512` carries a request or page value unwidened" in v
               for v in rep["violations"])
    rep = _mutate(_DRAIN_FRAG, _DRAIN_FRAG.replace('self.qoff["v_ch"] + dl_p', "mPT[(Int64(batch_idx), 0)] * 4096"))
    assert any("gVDrainPage coordinate 1 `mPT[Int64(batch_idx), 0] * 4096` carries a request or page value unwidened" in v
               for v in rep["violations"]), rep["violations"]
    rep = _mutate(_DRAIN_FRAG, _DRAIN_FRAG.replace('(kv_head_idx, self.qoff["v_ch"] + dl_p)', "(kv_head_idx,)"))
    assert any("gVDrainPage origin (kv_head_idx,) is not a full rank-2 coordinate" in v for v in rep["violations"])
    rep = _mutate(_DRAIN_FRAG, _DRAIN_FRAG + "\n                            drain_ptr = gVDrainFrag.iterator")
    assert any("gVDrainFrag is used as `gVDrainFrag.iterator`" in v for v in rep["violations"])


def test_the_rule_tracks_every_view_it_admits():
    """A slice made inline, as an origin offset's tensor or a call's argument, is refused, because
    the view it makes would escape tracking; and a rank-one slice `row[None]` is a tracked view like
    any other, so a page product moved into an origin offset of it is refused (codex, 2026-09-28).
    The inline slice is refused whether it slices a view or a global tensor, so the drain's two
    views written as one statement, with a page product in the origin's channel coordinate, are
    refused. Each inline slice, of the global tensor and of the view, is refused as well when it is
    passed as the argument of a device method that indexes it with a page product. An origin offset
    made inline, of a global tensor or of a view, is refused for the same reason, because its
    subscript would escape tracking, and so is an origin offset bound to two names at once, because a
    view is tracked only under a single name."""
    inline = _DRAIN_FRAG.replace("gVDrainPage)", "gVDrainPage[(None, None)])")
    rep = _mutate(_DRAIN_FRAG, inline)
    assert any(v.split(": ", 1)[1].startswith("gVDrainPage[") and v.endswith("is a slice that is not bound to a name")
               for v in rep["violations"]), rep["violations"]
    # the page's slice of the global tensor made inline, as the origin offset's tensor
    one_statement = ("gVDrainFrag = cute.domain_offset((kv_head_idx, v_page_pf * 4096), "
                     "mScaleRaw[(Int64(v_page_pf), None, None)])")
    src = _source()
    assert src.count(_DRAIN_PAGE) == 1 and src.count(_DRAIN_FRAG) == 1
    rep = global_tensor_report(src.replace(_DRAIN_PAGE, "").replace(_DRAIN_FRAG, one_statement))
    assert any(v.endswith(": mScaleRaw[(Int64(v_page_pf), None, None)] is a slice that is not bound to a name")
               for v in rep["violations"]), rep["violations"]
    indent = " " * 28
    # each inline slice passed as a device method's argument, which the method indexes with a page
    # product
    method = ("    @cute.jit\n    def _drain_read(self, gView, kv_head_idx, v_page_pf):\n"
              "        return gView[(kv_head_idx, v_page_pf * 4096)]\n\n")
    assert src.count(_CONSUMER_DEF) == 1
    with_method = src.replace(_CONSUMER_DEF, method + _CONSUMER_DEF)
    for sliced in ("mScaleRaw[(Int64(v_page_pf), None, None)]", "gVDrainPage[(None, None)]"):
        call = f"self._drain_read({sliced}, kv_head_idx, v_page_pf)"
        rep = global_tensor_report(with_method.replace(_DRAIN_FRAG, f"{_DRAIN_FRAG}\n{indent}rVDrain[0] = {call}.to(Float32)"))
        assert any(v.endswith(f": {sliced} is a slice that is not bound to a name") for v in rep["violations"]), \
            (sliced, rep["violations"])
    # an inline origin offset whose subscript leaves the page unwidened at the page position, and
    # one whose subscript carries a page product in the channel coordinate
    for tensor, moved, index in (
            ("mScaleRaw", "cute.domain_offset((Int64(v_page_pf), 0, 0), mScaleRaw)", "(v_page_pf, kv_head_idx, 0)"),
            ("gVDrainPage", "cute.domain_offset((kv_head_idx, 0), gVDrainPage)", "(0, v_page_pf * 4096)")):
        rep = _mutate(_DRAIN_FRAG, f"{_DRAIN_FRAG}\n{indent}rVDrain[0] = {moved}[{index}].to(Float32)")
        assert any(v.endswith(f": {tensor} is used as `{moved}`") for v in rep["violations"]), (moved, rep["violations"])
    # an origin offset of the view bound to two names, one of them indexed with a page product
    moved = "cute.domain_offset((kv_head_idx, 0), gVDrainPage)"
    rep = _mutate(_DRAIN_FRAG, f"{_DRAIN_FRAG}\n{indent}gVDrainA = gVDrainB = {moved}"
                               f"\n{indent}rVDrain[0] = gVDrainA[(0, v_page_pf * 4096)].to(Float32)")
    assert any(v.endswith(f": gVDrainPage is used as `{moved}`") for v in rep["violations"]), rep["violations"]
    rank_one = (_DRAIN_FRAG + f"\n{indent}gVDrainRow = gVDrainFrag[(0, None)]\n{indent}gVDrainAlias = gVDrainRow[None]"
                f"\n{indent}gVDrainBad = cute.domain_offset((v_page_pf * 4096,), gVDrainAlias)"
                f"\n{indent}rVDrain[0] = gVDrainBad[0].to(Float32)")
    rep = _mutate(_DRAIN_FRAG, rank_one)
    assert any("gVDrainAlias coordinate 0 `v_page_pf * 4096` carries a request or page value unwidened" in v
               for v in rep["violations"]), rep["violations"]


def test_an_origin_offset_of_a_wide_tensor_keeps_its_wide_positions():
    """A view bound from `cute.domain_offset(coord, t)` keeps every dimension of `t` and the widening
    duty of each, so its page position must be widened when the view is indexed, as `t`'s own would
    be: a plain value and a page value there are each refused as unwidened, and the same index
    widened is accepted. A value widened at the view's page position is a page value wherever else
    the method uses it, as it would be at `t`'s own: `kv_head_idx` widened there is refused at the
    kv-head position of the fragment's origin."""
    indent = " " * 24
    moved = "gVDrainBase = cute.domain_offset((Int64(v_page_pf), 0, 0), mScaleRaw)"
    for coord in ("kv_head_idx", "v_page_pf"):
        rep = _mutate(_DRAIN_PAGE, f"{moved}\n{indent}gVDrainPage = gVDrainBase[({coord}, None, None)]")
        assert any(f"gVDrainBase coordinate 0 `{coord}` is not widened" in v for v in rep["violations"]), \
            (coord, rep["violations"])
    rep = _mutate(_DRAIN_PAGE, f"{moved}\n{indent}zero_row = Int32(0)"
                               f"\n{indent}gVDrainPage = gVDrainBase[(Int64(zero_row), None, None)]")
    assert not rep["violations"], rep["violations"]
    rep = _mutate(_DRAIN_PAGE, f"{moved}\n{indent}gVDrainPage = gVDrainBase[(Int64(kv_head_idx), None, None)]")
    assert any("gVDrainPage coordinate 0 `kv_head_idx` carries a request or page value unwidened" in v
               for v in rep["violations"]), rep["violations"]


def test_a_page_value_parked_in_a_tensor_element_is_followed():
    """A page value stored into a register tensor and read back is still a page value, however it
    is stored (an element store, an augmented one, `.store` of another tensor's load, directly or
    through a slice, an `autovec_copy` into it, a store through a local view, origin offset, recast
    or alias), so a product of it in an origin offset's coordinate or a subscript's is
    refused; and a product stored and read back is not a plain value to widen (codex, 2026-09-28).
    A view the page is stored through may also be selected from a tuple nested in another. The page
    may also be stored through a tensor made on a recast of the storage's pointer or on that pointer
    less an offset, and through a view selected from a list. A product formed on the loaded values
    is not plain either, negated or not, while the plain page read back and negated is accepted. A
    sum of loaded values is plain only when both operands are, so the stored product read back
    through a sum, on either side, is refused, and the plain page read back the same way is
    accepted."""
    indent = " " * 28
    uses = ("gVDrainBad = cute.domain_offset((0, rPage[0] * 4096), gVDrainFrag)\n"
            f"{indent}rVDrain[0] = gVDrainBad[(0, 0)].to(Float32)",
            "rVDrain[0] = gVDrainPage[(kv_head_idx, rPage[0] * 4096)].to(Float32)")
    parks = ("rPage[0] = v_page_pf",
             "rPage[0] += v_page_pf",
             f"rPageSrc = cute.make_rmem_tensor(1, Int32)\n{indent}rPageSrc[0] = v_page_pf\n{indent}rPage.store(rPageSrc.load())",
             f"rPageSrc = cute.make_rmem_tensor(1, Int32)\n{indent}rPageSrc[0] = v_page_pf\n{indent}rPage[None].store(rPageSrc.load())",
             f"rPageView = rPage[None]\n{indent}rPageView[0] = v_page_pf",
             f"rPageAlias = rPage\n{indent}rPageAlias[0] = v_page_pf",
             f"rPageShift = cute.domain_offset((0,), rPage)\n{indent}rPageShift[0] = v_page_pf",
             f"rPageSrc = cute.make_rmem_tensor(1, Int32)\n{indent}rPageSrc[0] = v_page_pf\n{indent}cute.autovec_copy(rPageSrc, rPage)",
             f"rPageWords = cute.recast_tensor(rPage, Int32)\n{indent}rPageWords[0] = v_page_pf",
             "cute.recast_tensor(rPage, Int32)[0] = v_page_pf",
             "cute.domain_offset((0,), rPage)[0] = v_page_pf",
             # a tensor rebuilt on the storage's pointer, bound or inline
             f"rPageMade = cute.make_tensor(rPage.iterator.align(4), rPage.layout)\n{indent}rPageMade[0] = v_page_pf",
             "cute.make_tensor(rPage.iterator, rPage.layout)[0] = v_page_pf",
             "cute.make_tensor(rPage.iterator + 0, rPage.layout)[0] = v_page_pf",
             f"rPagePtr = rPage.iterator\n{indent}cute.make_tensor(rPagePtr, rPage.layout)[0] = v_page_pf",
             # a tensor made on a recast of that pointer, inline or bound, or on the pointer less
             # an offset
             "cute.make_tensor(cute.recast_ptr(rPage.iterator, dtype=Int32), rPage.layout)[0] = v_page_pf",
             f"rPageRecast = cute.recast_ptr(rPage.iterator, dtype=Int32)\n{indent}cute.make_tensor(rPageRecast, rPage.layout)[0] = v_page_pf",
             "cute.make_tensor(rPage.iterator - 0, rPage.layout)[0] = v_page_pf",
             # a view selected from a tuple, from a list or from a tuple nested in one, unpacked
             # from one, or a loop over one
             f"rPageViews = (rPage[None],)\n{indent}rPageAlias = rPageViews[0]\n{indent}rPageAlias[0] = v_page_pf",
             f"rPageViews = [rPage[None]]\n{indent}rPageAlias = rPageViews[0]\n{indent}rPageAlias[0] = v_page_pf",
             f"rPageNest = ((rPage[None],),)\n{indent}rPageAlias = rPageNest[0][0]\n{indent}rPageAlias[0] = v_page_pf",
             f"rPagePair = (rPage[None], rPage)\n{indent}rPageA, rPageB = rPagePair\n{indent}rPageB[0] = v_page_pf",
             f"for rPageAlias in (rPage[None],):\n{indent}    rPageAlias[0] = v_page_pf",
             # a pointer offset either way bound to a name, and a tuple chosen by a condition
             f"rPagePtr = rPage.iterator + 0\n{indent}cute.make_tensor(rPagePtr, rPage.layout)[0] = v_page_pf",
             f"rPagePtr = rPage.iterator - 0\n{indent}cute.make_tensor(rPagePtr, rPage.layout)[0] = v_page_pf",
             f"rPageOther = cute.make_rmem_tensor(1, Int32)\n{indent}rPagePair = (rPage,) if cutlass.const_expr(D == 128) "
             f"else (rPageOther,)\n{indent}rPageAlias = rPagePair[0]\n{indent}rPageAlias[0] = v_page_pf")
    for park in parks:
        for use in uses:
            parked = f"{_DRAIN_FRAG}\n{indent}rPage = cute.make_rmem_tensor(1, Int32)\n{indent}{park}\n{indent}{use}"
            rep = _mutate(_DRAIN_FRAG, parked)
            assert any("`rPage[0] * 4096` carries a request or page value unwidened" in v for v in rep["violations"]), \
                (park, use, rep["violations"])
    widened = (f"{_DRAIN_FRAG}\n{indent}rPage = cute.make_rmem_tensor(1, Int32)\n{indent}rPage[0] = v_page_pf * 4096"
               f"\n{indent}page_offset = rPage[0]\n{indent}gVDrainBad = mScaleRaw[(Int64(page_offset), None, None)]"
               f"\n{indent}rVDrain[0] = gVDrainBad[(0, 0)].to(Float32)")
    rep = _mutate(_DRAIN_FRAG, widened)
    assert any("mScaleRaw coordinate 0 `Int64(page_offset)` widens a computed value" in v for v in rep["violations"]), \
        rep["violations"]
    # the same product read back through a view made inline, or through a load, inline or named,
    # converted or not
    for read in ("page_offset = cute.domain_offset((0,), rPage)[0]", "page_offset = rPage.load()[0]",
                 f"rPageRegs = rPage.load()\n{indent}page_offset = rPageRegs[0]",
                 "page_offset = rPage.load().to(Int32)[0]",
                 f"rPageRegs = rPage.load().to(Int32)\n{indent}page_offset = rPageRegs[0]"):
        rep = _mutate(_DRAIN_FRAG, widened.replace("page_offset = rPage[0]", read))
        assert any("mScaleRaw coordinate 0 `Int64(page_offset)` widens a computed value" in v for v in rep["violations"]), \
            (read, rep["violations"])
    # a plain page stored, read back as it is or negated, and the product formed on the loaded
    # values, negated or not
    plain = widened.replace("rPage[0] = v_page_pf * 4096", "rPage[0] = v_page_pf")
    assert _mutate(_DRAIN_FRAG, plain)["violations"] == []
    negated = plain.replace("page_offset = rPage[0]", "page_offset = (-rPage.load())[0]")
    assert _mutate(_DRAIN_FRAG, negated)["violations"] == []
    for read in (f"rPageRegs = rPage.load() * 4096\n{indent}page_offset = rPageRegs[0]",
                 "page_offset = (rPage.load() * 4096).to(Int32)[0]",
                 f"rPageRegs = (rPage.load() + 1) * 4096\n{indent}page_offset = rPageRegs[0]",
                 "page_offset = (-(rPage.load() * 4096))[0]"):
        rep = _mutate(_DRAIN_FRAG, plain.replace("page_offset = rPage[0]", read))
        assert any("mScaleRaw coordinate 0 `Int64(page_offset)` widens a computed value" in v for v in rep["violations"]), \
            (read, rep["violations"])
    # a sum of loaded values, with the stored value on either side: refused when the product is
    # stored, accepted when the plain page is
    for read in ("page_offset = (rPage.load() + 0)[0]",
                 f"rZero = cute.make_rmem_tensor(1, Int32)\n{indent}rZero[0] = Int32(0)"
                 f"\n{indent}page_offset = (rZero.load() + rPage.load())[0]"):
        rep = _mutate(_DRAIN_FRAG, widened.replace("page_offset = rPage[0]", read))
        assert any("mScaleRaw coordinate 0 `Int64(page_offset)` widens a computed value" in v for v in rep["violations"]), \
            (read, rep["violations"])
        assert _mutate(_DRAIN_FRAG, plain.replace("page_offset = rPage[0]", read))["violations"] == [], read


_CONSUMER_DEF = "    @cute.jit\n    def _consumer_wg(self"


def test_a_page_value_a_device_method_stores_into_its_callers_tensor_is_followed():
    """A device method that stores into the elements of a tensor its caller passes it stores into
    the caller's tensor: a page value it stores taints the caller's tensor, directly or through a
    view on either side or a second call, and so does a block-table element it reads itself and
    stores; and a product it stores is not a plain value, neither for the caller to widen nor,
    read back and passed on, for a second device method to widen (codex, 2026-09-28). The same
    holds for a product the caller forms and passes to a method that stores its parameter as it
    is. A split value stored the same way is covered by
    `test_a_split_value_is_followed_along_the_page_values_routes`."""
    indent = " " * 28
    stashers = {
        "direct": "dst[0] = page",
        "through a view": "dst_view = dst[None]\n        dst_view[0] = page",
        "by a second call": "self._stash_again(dst, page)",
    }
    again = "    @cute.jit\n    def _stash_again(self, out, value):\n        out[0] = value\n\n"
    use = "rVDrain[0] = gVDrainPage[(kv_head_idx, rPage[0] * 4096)].to(Float32)"
    for label, body in stashers.items():
        method = f"    @cute.jit\n    def _stash_page(self, dst, page):\n        {body}\n\n" + again
        for arg in ("rPage", "rPage[None]"):
            call = (f"{_DRAIN_FRAG}\n{indent}rPage = cute.make_rmem_tensor(1, Int32)"
                    f"\n{indent}self._stash_page({arg}, v_page_pf)\n{indent}{use}")
            src = _source()
            assert src.count(_DRAIN_FRAG) == 1 and src.count(_CONSUMER_DEF) == 1
            rep = global_tensor_report(src.replace(_DRAIN_FRAG, call).replace(_CONSUMER_DEF, method + _CONSUMER_DEF))
            assert any("`rPage[0] * 4096` carries a request or page value unwidened" in v for v in rep["violations"]), \
                (label, arg, rep["violations"])
    # the callee reads the page from the block table itself: no name it stores is tainted, the
    # block-table read it stores is (`_reads_block_table`)
    method = ("    @cute.jit\n    def _stash_page(self, dst, mPT, batch_idx):\n"
              "        dst[0] = mPT[(Int64(batch_idx), 0)]\n\n")
    call = (f"{_DRAIN_FRAG}\n{indent}rPage = cute.make_rmem_tensor(1, Int32)"
            f"\n{indent}self._stash_page(rPage, mPT, batch_idx)\n{indent}{use}")
    rep = global_tensor_report(_source().replace(_DRAIN_FRAG, call).replace(_CONSUMER_DEF, method + _CONSUMER_DEF))
    assert any("`rPage[0] * 4096` carries a request or page value unwidened" in v for v in rep["violations"]), \
        rep["violations"]
    # the stored product, read back by the caller and widened
    method = "    @cute.jit\n    def _stash_page(self, dst, page):\n        dst[0] = page * 4096\n\n"
    call = (f"{_DRAIN_FRAG}\n{indent}rPage = cute.make_rmem_tensor(1, Int32)\n{indent}self._stash_page(rPage, v_page_pf)"
            f"\n{indent}page_offset = rPage[0]\n{indent}gVDrainBad = mScaleRaw[(Int64(page_offset), None, None)]"
            f"\n{indent}rVDrain[0] = gVDrainBad[(0, 0)].to(Float32)")
    src = _source().replace(_DRAIN_FRAG, call)
    rep = global_tensor_report(src.replace(_CONSUMER_DEF, method + _CONSUMER_DEF))
    assert any("mScaleRaw coordinate 0 `Int64(page_offset)` widens a computed value" in v for v in rep["violations"]), \
        rep["violations"]
    # and the same method storing the plain page is accepted
    plain = method.replace("page * 4096", "page")
    assert global_tensor_report(src.replace(_CONSUMER_DEF, plain + _CONSUMER_DEF))["violations"] == []
    # the method storing its parameter as it is, passed a product by its caller: what it stores is
    # not plain either, because that parameter is refused (`_not_plain_params`)
    assert src.count("self._stash_page(rPage, v_page_pf)") == 1
    passed = src.replace("self._stash_page(rPage, v_page_pf)", "self._stash_page(rPage, v_page_pf * 4096)")
    rep = global_tensor_report(passed.replace(_CONSUMER_DEF, plain + _CONSUMER_DEF))
    assert any("mScaleRaw coordinate 0 `Int64(page_offset)` widens a computed value" in v for v in rep["violations"]), \
        rep["violations"]
    # the stored product, read back by the caller and passed to a second device method that widens
    # it, is refused there; the plain page stored and passed on the same way is accepted
    reader = "    @cute.jit\n    def _read_at(self, mScaleRaw, off):\n        return mScaleRaw[(Int64(off), 0, 0)]\n\n"
    call = (f"{_DRAIN_FRAG}\n{indent}rPage = cute.make_rmem_tensor(1, Int32)\n{indent}self._stash_page(rPage, v_page_pf)"
            f"\n{indent}page_offset = rPage[0]\n{indent}rVDrain[0] = self._read_at(mScaleRaw, page_offset).to(Float32)")
    src = _source().replace(_DRAIN_FRAG, call)
    rep = global_tensor_report(src.replace(_CONSUMER_DEF, method + reader + _CONSUMER_DEF))
    assert any(v.startswith("_read_at:") and "mScaleRaw coordinate 0 `Int64(off)` widens a computed value" in v
               for v in rep["violations"]), rep["violations"]
    assert global_tensor_report(src.replace(_CONSUMER_DEF, plain + reader + _CONSUMER_DEF))["violations"] == []


def test_a_view_passed_to_a_device_method_is_refused():
    """A view is not handed to a device method, positionally or by keyword, even as a parameter of
    its own name, because inside the method that parameter is not tracked: a page product the
    method puts in the view's channel coordinate would escape the rule."""
    indent = " " * 28
    method = ("    @cute.jit\n    def _drain_read(self, gVDrainPage, kv_head_idx, v_page_pf):\n"
              "        return gVDrainPage[(kv_head_idx, v_page_pf * 4096)]\n\n")
    src = _source()
    assert src.count(_DRAIN_FRAG) == 1 and src.count(_CONSUMER_DEF) == 1
    src = src.replace(_CONSUMER_DEF, method + _CONSUMER_DEF)
    for call, used_as in (
            ("self._drain_read(gVDrainPage, kv_head_idx, v_page_pf)",
             "self._drain_read(gVDrainPage, kv_head_idx, v_page_pf)"),
            ("self._drain_read(gVDrainPage=gVDrainPage, kv_head_idx=kv_head_idx, v_page_pf=v_page_pf)",
             "gVDrainPage=gVDrainPage")):
        rep = global_tensor_report(src.replace(_DRAIN_FRAG, f"{_DRAIN_FRAG}\n{indent}rVDrain[0] = {call}.to(Float32)"))
        assert any(v.endswith(f": gVDrainPage is used as `{used_as}`") for v in rep["violations"]), \
            (call, rep["violations"])


def test_an_origin_offset_carrying_a_split_value_is_refused():
    rep = _mutate(_DRAIN_FRAG, _DRAIN_FRAG.replace('self.qoff["v_ch"] + dl_p', 'split_idx * 4096 + self.qoff["v_ch"] + dl_p'))
    assert any("gVDrainPage coordinate 1 `split_idx * 4096 + self.qoff['v_ch'] + dl_p` carries a split value unwidened" in v
               for v in rep["violations"]), rep["violations"]


def test_a_split_value_is_followed_along_the_page_values_routes():
    """A split value is followed along the routes a page value is, so a flattened product of it is
    refused wherever it lands: through a name bound from it, a register tensor it is stored into
    and read back from, the parameter of a device method it is passed to, and a register tensor of
    the caller's that a device method stores it into. The same code carrying `kv_head_idx`, which
    is not a split value, is accepted, so each refusal is the split value's."""
    indent = " " * 28
    read_rsplit = "rVDrain[0] = gVDrainPage[(kv_head_idx, rSplit[0] * 4096)].to(Float32)"
    routes = {
        "a name bound from it": ("", f"split_off = VALUE * 4096\n{indent}"
                                     "rVDrain[0] = gVDrainPage[(kv_head_idx, split_off)].to(Float32)",
                                 "_consumer_wg:", "gVDrainPage coordinate 1 `split_off`"),
        "a register tensor": ("", f"rSplit = cute.make_rmem_tensor(1, Int32)\n{indent}rSplit[0] = VALUE\n{indent}{read_rsplit}",
                              "_consumer_wg:", "gVDrainPage coordinate 1 `rSplit[0] * 4096`"),
        "a device method's parameter": (
            "    @cute.jit\n    def _read_split_scale(self, mScaleRaw, s, v_page_pf):\n"
            "        return mScaleRaw[(Int64(v_page_pf), 0, s * 4096)]\n\n",
            "rVDrain[0] = self._read_split_scale(mScaleRaw, VALUE, v_page_pf).to(Float32)",
            "_read_split_scale:", "mScaleRaw coordinate 2 `s * 4096`"),
        "a device method's store into the caller's tensor": (
            "    @cute.jit\n    def _stash_split(self, dst, split):\n        dst[0] = split\n\n",
            f"rSplit = cute.make_rmem_tensor(1, Int32)\n{indent}self._stash_split(rSplit, VALUE)\n{indent}{read_rsplit}",
            "_consumer_wg:", "gVDrainPage coordinate 1 `rSplit[0] * 4096`"),
    }
    for label, (methods, code, where, needle) in routes.items():
        src = _source()
        assert src.count(_DRAIN_FRAG) == 1 and src.count(_CONSUMER_DEF) == 1
        src = src.replace(_CONSUMER_DEF, methods + _CONSUMER_DEF)
        rep = global_tensor_report(src.replace(_DRAIN_FRAG, f"{_DRAIN_FRAG}\n{indent}{code.replace('VALUE', 'split_idx')}"))
        assert any(v.startswith(where) and f"{needle} carries a split value unwidened" in v for v in rep["violations"]), \
            (label, rep["violations"])
        rep = global_tensor_report(src.replace(_DRAIN_FRAG, f"{_DRAIN_FRAG}\n{indent}{code.replace('VALUE', 'kv_head_idx')}"))
        assert rep["violations"] == [], (label, rep["violations"])


_COMBINE_PASS1 = ("rLBlk[kk] = mLSE[(Int64(o + kk), Int64(batch_idx), head)] if o + kk < split_count "
                  "else Float32(-1e30)")
#: Pass 2 runs once per 128-channel pass, one level deeper than pass 1, at the pass's `channel`.
_COMBINE_PASS2 = ("rLBlk[kk] = mLSE[(Int64(o + kk), Int64(batch_idx), head)] if inb else Float32(-1e30)\n"
                  + " " * 36 + "rOBlk[kk] = mOp[(Int64(o + kk), Int64(batch_idx), head, channel)] if inb "
                  "else Float32(0.0)")
_COMBINE_ENTRY = " " * 24 + "global_max = Float32(-1e30)\n" + " " * 24 + "o = Int32(0)\n"


def _combine(pass1, pass2, entry=""):
    """The source with the fused combine's two split loops' reads replaced, and `entry` bound
    before the first."""
    src = _source()
    assert src.count(_COMBINE_PASS1) == 1 and src.count(_COMBINE_PASS2) == 1 and src.count(_COMBINE_ENTRY) == 1
    return (src.replace(_COMBINE_ENTRY, entry + _COMBINE_ENTRY)
            .replace(_COMBINE_PASS1, pass1).replace(_COMBINE_PASS2, pass2))


def test_a_split_counter_widened_elsewhere_is_a_split_value():
    """The combine's split counter `o` reaches the split position only inside a widening, so moved
    into the channel coordinate as a flattened split offset, `o * 2^27`, it is refused. That holds
    whether the counter is widened inline, `Int64(o + kk)`, through a name bound to that widening,
    at the split position of a request slice of `mLSE` and `mOp`, whose coordinate 0 still
    indexes the split, or in the origin offset of such a slice, so that it reaches the split
    position through a view of a view and not through any subscript; and the same reads with
    nothing flattened are accepted. It holds as well when the counter is widened only as the bare
    index of a rank-one column of `mLSE` that keeps nothing but the split, and `mOp`'s request
    slice is read at split 0."""
    indent, indent2 = " " * 32, " " * 36    # pass 1, and pass 2 inside the 128-channel pass loop
    old = "rOBlk[kk] = mOp[(Int64(o + kk), Int64(batch_idx), head, channel)] if inb else Float32(0.0)"
    rep = _mutate(old, old.replace("Int64(o + kk)", "Int64(kk)").replace("head, channel", "head, o * 134217728 + channel"))
    assert any("mOp coordinate 3 `o * 134217728 + channel` carries a split value unwidened" in v
               for v in rep["violations"]), rep["violations"]
    flat = "o * 134217728 + channel"
    bound = ("split_row_a = Int64(o + kk)\n"
             f"{indent}rLBlk[kk] = mLSE[(split_row_a, Int64(batch_idx), head)] if o + kk < split_count else Float32(-1e30)",
             "split_row_b = Int64(o + kk)\n"
             f"{indent2}rLBlk[kk] = mLSE[(split_row_b, Int64(batch_idx), head)] if inb else Float32(-1e30)\n"
             f"{indent2}rOBlk[kk] = mOp[(split_row_b, Int64(batch_idx), head, CHANNEL)] if inb else Float32(0.0)", "",
             "mOp coordinate 3")
    views = ("rLBlk[kk] = gLseReq[(Int64(o + kk), head)] if o + kk < split_count else Float32(-1e30)",
             "rLBlk[kk] = gLseReq[(Int64(o + kk), head)] if inb else Float32(-1e30)\n"
             f"{indent2}rOBlk[kk] = gOpReq[(Int64(o + kk), head, CHANNEL)] if inb else Float32(0.0)",
             " " * 24 + "gLseReq = mLSE[(None, Int64(batch_idx), None)]\n"
             + " " * 24 + "gOpReq = mOp[(None, Int64(batch_idx), None, None)]\n",
             "gOpReq coordinate 2")
    offsets = ("gLseAt = cute.domain_offset((Int64(o + kk), 0), gLseReq)\n"
               f"{indent}rLBlk[kk] = gLseAt[(Int64(zero_split), head)] if o + kk < split_count else Float32(-1e30)",
               "gLseAt2 = cute.domain_offset((Int64(o + kk), 0), gLseReq)\n"
               f"{indent2}gOpAt = cute.domain_offset((Int64(o + kk), 0, 0), gOpReq)\n"
               f"{indent2}rLBlk[kk] = gLseAt2[(Int64(zero_split), head)] if inb else Float32(-1e30)\n"
               f"{indent2}rOBlk[kk] = gOpAt[(Int64(zero_split), head, CHANNEL)] if inb else Float32(0.0)",
               " " * 24 + "zero_split = Int32(0)\n" + views[2],
               "gOpAt coordinate 2")
    column = ("rLBlk[kk] = gLseCol[Int64(o + kk)] if o + kk < split_count else Float32(-1e30)",
              "rLBlk[kk] = gLseCol[Int64(o + kk)] if inb else Float32(-1e30)\n"
              f"{indent2}rOBlk[kk] = gOpReq[(Int64(zero_split), head, CHANNEL)] if inb else Float32(0.0)",
              " " * 24 + "zero_split = Int32(0)\n" + " " * 24 + "gLseCol = mLSE[(None, Int64(batch_idx), head)]\n"
              + " " * 24 + "gOpReq = mOp[(None, Int64(batch_idx), None, None)]\n",
              "gOpReq coordinate 2")
    for pass1, pass2, entry, where in (bound, views, offsets, column):
        src = _combine(pass1, pass2.replace("CHANNEL", "channel"), entry)
        assert global_tensor_report(src)["violations"] == [], (where, global_tensor_report(src)["violations"])
        rep = global_tensor_report(_combine(pass1, pass2.replace("CHANNEL", flat), entry))
        assert any(f"{where} `{flat}` carries a split value unwidened" in v for v in rep["violations"]), \
            (where, rep["violations"])


def test_a_page_value_read_and_passed_to_a_method_is_followed():
    """A page read from the block table and handed to a device method seeds that method's
    parameter, so a product of it there is refused: an element read passed inline, a read
    through a slice, which is a block-table read itself, or a read through an origin offset,
    whose request coordinate is widened inline or by a name bound to `Int64(...)` beforehand,
    either of which seeds the names it widens."""
    call = "                heads_per_kv,\n                merge_head_limit_dyn,\n                kv_head_idx,\n                mCnt,"
    anchor = "        with annotate.range(\"epilogue\"):\n"
    pre_widened = ("origin_row = Int32(0)\n        origin_row64 = Int64(origin_row)\n"
                   "        gPageTable = cute.domain_offset((origin_row64, 0), mPT)")
    for view, read in ((None, "mPT[(Int64(batch_idx), 0)]"),
                       ("gPageTable = mPT[(Int64(batch_idx), None)]", "gPageTable[0]"),
                       ("gPageTable = cute.domain_offset((Int64(zero_row), 0), mPT)", "gPageTable[(Int64(batch_idx), 0)]"),
                       (pre_widened, "gPageTable[(Int64(batch_idx), 0)]")):
        src = _source()
        assert src.count(call) == 1 and src.count(anchor) == 1
        if view is not None:
            src = src.replace(anchor, f"        zero_row = Int32(0)\n        {view}\n" + anchor)
        src = src.replace(call, call.replace("                kv_head_idx,\n", f"                {read},\n"))
        rep = global_tensor_report(src)
        assert any(v.startswith("_merge_and_store:") and "carries a request or page value unwidened" in v
                   for v in rep["violations"]), (view, rep["violations"])


def test_a_read_through_a_view_of_the_block_table_is_a_page_value():
    """A page read through a view of `mPT`, an origin offset or a slice, is a page value, so a page
    product that reaches another coordinate through it is refused (codex, 2026-09-28). A slice is
    a block-table read itself (`_reads_block_table`); an origin offset is one because the
    widening of its request coordinate seeds the names in it, whether inline or, in the third
    view, through a name bound to `Int64(...)` beforehand, whose name is distinct from the drain's
    `zero_row`, so no seed of the drain's widening reaches the view. The fourth view is a slice of
    an origin offset of `mPT`, a view of a view, again under a name distinct from `zero_row`. The
    fifth is the other order, an origin offset of a slice of `mPT`, moved to the CTA's first tile
    and read at the tile's place in the CTA's range, so a split value sits both in the offset's
    coordinate and in its subscript. The same views with nothing moved are accepted: the tile
    index keeps the block-table column's exemption through a view, and through a view of a view,
    because each is looked up at the root position: for the fifth view's offset coordinate, the
    root position of the slice it moves, and for its subscript, the root the moved view records."""
    page_read = "v_page_pf = mPT[(Int64(batch_idx), global_tile_idx)]"
    frag = _DRAIN_FRAG.replace('self.qoff["v_ch"] + dl_p', 'v_page_pf * 4096 + self.qoff["v_ch"] + dl_p')
    indent = " " * 24
    pre_widened = (f"origin_row = Int32(0)\n{indent}origin_row64 = Int64(origin_row)\n"
                   f"{indent}gPageTable = cute.domain_offset((origin_row64, 0), mPT)")
    two_level = (f"origin_row = Int32(0)\n{indent}gPageRows = cute.domain_offset((Int64(origin_row), 0), mPT)\n"
                 f"{indent}gPageTable = gPageRows[(Int64(batch_idx), None)]")
    split_moved = (f"gPageTable = mPT[(Int64(batch_idx), None)]\n{indent}cta_first_tile = split_idx * tiles_per_cta\n"
                   f"{indent}gPageSplit = cute.domain_offset((cta_first_tile,), gPageTable)")
    for view, read in (("gPageTable = cute.domain_offset((Int64(zero_row), 0), mPT)",
                        "v_page_pf = gPageTable[(Int64(batch_idx), global_tile_idx)]"),
                       ("gPageTable = mPT[(Int64(batch_idx), None)]", "v_page_pf = gPageTable[global_tile_idx]"),
                       (pre_widened, "v_page_pf = gPageTable[(Int64(batch_idx), global_tile_idx)]"),
                       (two_level, "v_page_pf = gPageTable[global_tile_idx]"),
                       (split_moved, "v_page_pf = gPageSplit[global_tile_idx - cta_first_tile]")):
        src = _source()
        assert src.count(page_read) == 1 and src.count(_DRAIN_FRAG) == 1 and src.count(_DRAIN_PAGE) == 1
        src = src.replace(page_read, f"zero_row = Int32(0)\n{indent}{view}\n{indent}{read}")
        assert global_tensor_report(src)["violations"] == [], (view, global_tensor_report(src)["violations"])
        src = (src.replace(_DRAIN_PAGE, "gVDrainPage = mScaleRaw[(Int64(zero_row), None, None)]")
               .replace(_DRAIN_FRAG, frag))
        rep = global_tensor_report(src)
        assert any("coordinate 1 `v_page_pf * 4096 + self.qoff['v_ch'] + dl_p` carries a request or page "
                   "value unwidened" in v for v in rep["violations"]), (view, rep["violations"])


def test_the_rule_flags_a_tma_slice_that_is_not_a_kv_head():
    rep = _mutate("mScale[(None, kv_head_idx, None)]",
                  "mScale[(None, batch_idx * 8 + kv_head_idx, None)]")
    assert any("is not a kv-head slice" in v for v in rep["violations"])


def test_the_rule_accepts_the_equivalent_spellings_of_a_widening():
    """`cutlass.Int64(x)`, and a name bound once to `Int64(x)`, widen as `Int64(x)` does."""
    anchor = "        gQ = mQ[(Int64(batch_idx), None, None)]\n"
    for new in ("        gQ = mQ[(cutlass.Int64(batch_idx), None, None)]\n",
                "        b64 = Int64(batch_idx)\n        gQ = mQ[(b64, None, None)]\n"):
        assert _mutate(anchor, new)["violations"] == []


def test_the_rule_flags_a_widened_name_whose_product_already_wrapped():
    """A name bound to a product, negated or not, is not a plain value to widen, because the
    product may already have wrapped, while a name bound to the negated request itself is accepted.
    The negated forms need the name: written inside `Int64(...)`, they are refused by the shape a
    widened coordinate must have before their value is judged."""
    anchor = "        gQ = mQ[(Int64(batch_idx), None, None)]\n"
    for q_batch in ("batch_idx * 524288 // 524288", "-(batch_idx * 524288 // 524288)"):
        rep = _mutate(anchor, f"        q_batch = {q_batch}\n        gQ = mQ[(Int64(q_batch), None, None)]\n")
        assert any("widens a computed value" in v for v in rep["violations"]), (q_batch, rep["violations"])
    assert _mutate(anchor, "        q_batch = -batch_idx\n        gQ = mQ[(Int64(q_batch), None, None)]\n")["violations"] == []


def test_the_rule_follows_a_request_value_into_a_loop_target():
    anchor = "        gQ = mQ[(Int64(batch_idx), None, None)]\n"
    rep = _mutate(anchor, anchor + "        for q_flat in cutlass.range(batch_idx * 524288, "
                  "batch_idx * 524288 + 1, 1):\n            q0 = mQ[(Int64(kv_head_idx), 0, q_flat)]\n")
    assert any("mQ coordinate 2 `q_flat` carries a request or page value unwidened" in v
               for v in rep["violations"])


def test_the_rule_follows_a_request_value_into_a_renamed_parameter():
    """`_stage_q` receives the request as `batch_idx`; renamed, the value must still be tracked."""
    src = _source()
    renamed = src.replace("        PG,\n        batch_idx,\n        heads_per_kv,\n        io_thread_idx,",
                          "        PG,\n        req,\n        heads_per_kv,\n        io_thread_idx,", 1)
    assert renamed != src, "the _stage_q signature moved; update this mutation"
    renamed = renamed.replace("        gQ = mQ[(Int64(batch_idx), None, None)]\n",
                              "        gQ = mQ[(Int64(req), None, None)]\n"
                              "        q0 = mQ[(Int64(req), 0, req * 65536)]\n", 1)
    rep = global_tensor_report(renamed)
    assert any("coordinate 2 `req * 65536` carries a request or page value unwidened" in v
               for v in rep["violations"])


def test_the_counter_operands_must_be_bound_once():
    src = _source().replace("        cnt_index = Int64(batch_idx)",
                            "        kv_head_count *= 0\n        cnt_index = Int64(batch_idx)", 1)
    tree = ast.parse(src)
    kernel = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "kernel")
    assert len(_stores(kernel)["kv_head_count"]) == 2   # which the counter test refuses


def test_the_rule_flags_a_keyword_only_parameter():
    rep = _mutate("               tma_atom_k_codes, tma_atom_v_codes, tma_atom_quant_params):",
                  "               tma_atom_k_codes, tma_atom_v_codes, tma_atom_quant_params, *, qExtra=None):")
    assert any("parameter qExtra is not classified" in v for v in rep["violations"])


def test_the_widened_alias_counts_in_the_arithmetic_as_well():
    src = _source().replace("        gQ = mQ[(Int64(batch_idx), None, None)]\n",
                            "        b64 = Int64(batch_idx)\n        gQ = mQ[(b64, None, None)]\n", 1)
    assert global_tensor_report(src)["violations"] == []
    assert _wide_positions_of(src, "mQ") == {0}


def test_the_rule_follows_a_wrapped_product_into_a_callee():
    """A product passed to a device method is not a plain value there, so the method's widening of
    that parameter is refused, whether it is written inline or through a name bound to
    `Int64(...)`, which then does not count as widened."""
    call = "                self._stage_q(\n                    D,\n                    PG,\n                    batch_idx,"
    wrapped = "                self._stage_q(\n                    D,\n                    PG,\n                    batch_idx * 524288 // 524288,"
    rep = _mutate(call, wrapped)
    assert any("_stage_q" in v and "widens a computed value" in v for v in rep["violations"])
    anchor = "        gQ = mQ[(Int64(batch_idx), None, None)]\n"
    src = _source()
    assert src.count(call) == 1 and src.count(anchor) == 1
    bound = src.replace(call, wrapped).replace(anchor, "        b64 = Int64(batch_idx)\n        gQ = mQ[(b64, None, None)]\n")
    rep = global_tensor_report(bound)
    assert any(v.startswith("_stage_q:") and v.endswith(": mQ coordinate 0 `b64` is not widened")
               for v in rep["violations"]), rep["violations"]


def test_the_rule_looks_inside_a_tuple_built_in_the_method():
    anchor = "        gQ = mQ[(Int64(batch_idx), None, None)]\n"
    rep = _mutate(anchor, "        q_coords = (batch_idx * 524288 // 524288, kv_head_idx)\n"
                          "        q_batch = q_coords[0]\n        gQ = mQ[(Int64(q_batch), None, None)]\n")
    assert any("widens a computed value" in v for v in rep["violations"])


def test_the_rule_follows_a_request_value_into_a_positional_only_parameter():
    src = _source()
    method = "    @cute.jit\n    def _stage_q(\n"
    assert src.count(method) == 1
    src = src.replace(method, "    @cute.jit\n    def _read_query_element(self, req, /, mQ, kv_head_idx):\n"
                              "        return mQ[(Int64(kv_head_idx), 0, req * 65536)]\n\n" + method, 1)
    anchor = "        gQ = mQ[(Int64(batch_idx), None, None)]\n"
    assert src.count(anchor) == 1
    src = src.replace(anchor, anchor + "        q0 = self._read_query_element(batch_idx, mQ, kv_head_idx)\n", 1)
    rep = global_tensor_report(src)
    assert any("_read_query_element" in v and "carries a request or page value unwidened" in v
               for v in rep["violations"])


def test_the_rule_follows_a_tuple_through_a_parameter_and_an_alias():
    src = _source()
    method = "    @cute.jit\n    def _stage_q(\n"
    src = src.replace(method, "    @cute.jit\n    def _read_query_row(self, coords, mQ):\n"
                              "        q_coords = coords\n        q_batch = q_coords[0]\n"
                              "        return mQ[(Int64(q_batch), 0, 0)]\n\n" + method, 1)
    anchor = "        gQ = mQ[(Int64(batch_idx), None, None)]\n"
    src = src.replace(anchor, anchor + "        q0 = self._read_query_row((batch_idx * 524288 // 524288, "
                              "kv_head_idx), mQ)\n", 1)
    rep = global_tensor_report(src)
    assert any("_read_query_row" in v and "widens a computed value" in v for v in rep["violations"])


def test_the_rule_accepts_forwarding_through_a_positional_only_signature():
    src = _source()
    method = "    @cute.jit\n    def _stage_q(\n"
    src = src.replace(method, "    @cute.jit\n    def _read_query_element(self, req, /, mQ, kv_head_idx):\n"
                              "        return mQ[(Int64(req), kv_head_idx, 0)]\n\n" + method, 1)
    anchor = "        gQ = mQ[(Int64(batch_idx), None, None)]\n"
    src = src.replace(anchor, anchor + "        q0 = self._read_query_element(batch_idx, mQ, kv_head_idx)\n", 1)
    assert global_tensor_report(src)["violations"] == []


def test_a_tuple_cannot_shed_its_product_by_unpacking_or_forwarding():
    """A tuple is judged whole, so unpacking it or passing it on keeps the product it holds."""
    src = _source()
    method = "    @cute.jit\n    def _stage_q(\n"
    src = src.replace(method, "    @cute.jit\n    def _read_query_row(self, coords, mQ):\n"
                              "        return self._read_query_row_again(coords, mQ)\n\n"
                              "    @cute.jit\n    def _read_query_row_again(self, coords, mQ):\n"
                              "        q_batch, q_head = coords\n"
                              "        return mQ[(Int64(q_batch), 0, 0)]\n\n" + method, 1)
    anchor = "        gQ = mQ[(Int64(batch_idx), None, None)]\n"
    src = src.replace(anchor, anchor + "        q0 = self._read_query_row((batch_idx * 524288 // 524288, "
                              "kv_head_idx), mQ)\n", 1)
    rep = global_tensor_report(src)
    assert any("_read_query_row_again" in v and "widens a computed value" in v for v in rep["violations"])


def test_a_nested_or_conditional_tuple_cannot_shed_its_product():
    anchor = "        gQ = mQ[(Int64(batch_idx), None, None)]\n"
    for defn, take in (("((batch_idx * 524288 // 524288, kv_head_idx),)", "q_coords[0][0]"),
                       ("((batch_idx * 524288 // 524288, kv_head_idx) if cutlass.const_expr(D == 128)"
                        " else (batch_idx, kv_head_idx))", "q_coords[0]")):
        rep = _mutate(anchor, f"        q_coords = {defn}\n        q_batch = {take}\n"
                              "        gQ = mQ[(Int64(q_batch), None, None)]\n")
        assert any("widens a computed value" in v for v in rep["violations"]), defn


def test_a_slice_marker_in_a_tuple_is_plain():
    anchor = "        gQ = mQ[(Int64(batch_idx), None, None)]\n"
    rep = _mutate(anchor, "        q_coords = (batch_idx, None, None)\n        q_batch = q_coords[0]\n"
                          "        gQ = mQ[(Int64(q_batch), None, None)]\n")
    assert rep["violations"] == []


def test_a_tuple_holding_a_product_is_refused_whole():
    """The documented conservative case: the element taken is plain, the tuple is not."""
    anchor = "        gQ = mQ[(Int64(batch_idx), None, None)]\n"
    rep = _mutate(anchor, "        q_coords = (batch_idx, kv_head_idx * heads_per_kv)\n"
                          "        q_batch = q_coords[0]\n        gQ = mQ[(Int64(q_batch), None, None)]\n")
    assert any("widens a computed value" in v for v in rep["violations"])


def test_the_rule_accepts_a_plain_tuple_unpacking():
    anchor = "        gQ = mQ[(Int64(batch_idx), None, None)]\n"
    rep = _mutate(anchor, "        q_batch, q_head = batch_idx, kv_head_idx\n"
                          "        gQ = mQ[(Int64(q_batch), None, None)]\n")
    assert rep["violations"] == []


# ---------------------------------------------------------------------------------------------
# The arithmetic at each boundary
# ---------------------------------------------------------------------------------------------

def _i32(x):
    # One-element tensors, not 0-dim ones: torch leaves a 0-dim tensor out of type promotion,
    # while the device promotes an Int64 coordinate against an Int32 term to Int64.
    return torch.tensor([x], dtype=torch.int32)


def _strides(shape):
    out, acc = [], 1
    for extent in reversed(shape):
        out.append(acc)
        acc *= extent
    return out[::-1]


def _offset(shape, coord, widen):
    """C-contiguous element offset with 32-bit coordinates, the positions in `widen` 64-bit.

    Every stride is the compile-time value a static layout carries. The production compile
    gives the split and block-table strides as runtime values, and those are 64-bit, so its
    split and block-table products were already wide; the rule does not depend on that."""
    total = None
    for pos, (c, s) in enumerate(zip(coord, _strides(shape))):
        c = _i32(c).to(torch.int64) if pos in widen else _i32(c)
        # a stride too large for 32 bits is a 64-bit value on the device too
        term = c * (torch.tensor([s], dtype=torch.int64) if s >= 1 << 31 else s)
        total = term if total is None else total + term
    return total


def _wide_positions(name):
    """The positions the SOURCE widens at every site of `name` (asserted equal to WIDE above).

    A `None` counts: it keeps the dimension in a slice, which the report then holds to the same
    duty, so the offset the slice's own subscript adds is still formed from a widened value."""
    return _wide_positions_of(_source(), name)


def _wide_positions_of(src, name):
    _MEMO.clear()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CLASS)
    seen = None
    methods = _device_methods(cls)
    marks, _, not_plain = _out_stores(methods)
    for mname, m in methods.items():
        defs, stores = _definitions(m, marks[mname]), _stores(m)
        np_m = frozenset(not_plain[mname])
        wide_names = _widened_names(m, defs, stores, np_m)
        for node in ast.walk(m):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and node.value.id == name and isinstance(node.slice, ast.Tuple)):
                duty = WIDE[name][1] if name in WIDE else ()
                here = {i for i, e in enumerate(node.slice.elts)
                        if (_is_int64(e) and _plain_coordinate(e.args[0], defs, np_m))
                        or (isinstance(e, ast.Name) and e.id in wide_names)
                        or (_is_none(e) and i in duty)}
                seen = here if seen is None else seen & here
    return seen or set()


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _first_scale_read_past(hkv, qoff, D=128):
    """(page, kv_head, element) of the first channel-section element the kernel reads at or past
    element 2^31 of a C-contiguous [NP, HKV, block_tot] pool. The kernel reads the (s, z) pair of
    every channel, [k_ch, k_ch + 2D) and [v_ch, v_ch + 2D)."""
    bt = qoff["block_tot"]
    read = sorted(e for base in (qoff["k_ch"], qoff["v_ch"]) if base is not None
                  for e in range(base, base + 2 * D))
    page = (1 << 31) // (hkv * bt) - 1
    while True:
        for hk in range(hkv):
            for e in read:
                if (page * hkv + hk) * bt + e >= 1 << 31:
                    return page, hk, e
        page += 1


def _scale_pairs():
    env = _load_by_path("_quant_envelope_for_wide_offsets", ROOT / "scripts/_quant_envelope.py")
    ql = _load_by_path("_quant_layout_for_wide_offsets", ROOT / "src/cutedsl_decode/quant_layout.py")
    out = []
    for kq in env.SM90_K_SET:
        for vq in env.SM90_V_SET:
            qoff = ql.quant_block_offsets(kq, vq, 64, 128)
            if qoff["k_ch"] is not None or qoff["v_ch"] is not None:
                out.append((kq, vq, qoff))
    return out


def test_the_source_widens_exactly_the_declared_positions():
    for name, (_, wide) in WIDE.items():
        assert _wide_positions(name) == set(wide), name


@pytest.mark.parametrize("kq,vq,qoff", _scale_pairs(), ids=lambda x: str(x) if not isinstance(x, dict) else "")
def test_the_first_channel_scale_read_past_2_31_is_addressed_where_it_is(kq, vq, qoff):
    """Every pair with a channel section: the first element read at or past 2^31 is exact with
    the widened page and wraps negative without it. The page is 524,288 with a 512-element block
    and 699,050 with a 384-element one."""
    hkv = 8
    page, hk, e = _first_scale_read_past(hkv, qoff)
    assert page == {512: 524_288, 384: 699_050}[qoff["block_tot"]]
    shape = (page + 1, hkv, qoff["block_tot"])
    exact = (page * hkv + hk) * qoff["block_tot"] + e
    wide = _wide_positions("mScaleRaw")
    assert int(_offset(shape, (page, hk, e), wide)) == exact
    narrow = _offset(shape, (page, hk, e), ())
    assert narrow.dtype == torch.int32 and int(narrow) == exact - (1 << 32) < 0


def test_the_384_block_boundary_falls_inside_page_699050():
    """With a 384-element block the page is 3,072 elements, so element 2^31 is element 2,048 of
    page 699,050: kv-head 5's channel section, which starts at element 128 of its block."""
    assert 699_050 * 8 * 384 == (1 << 31) - 2048
    for kq, vq, qoff in _scale_pairs():
        if qoff["block_tot"] == 384:
            assert _first_scale_read_past(8, qoff) == (699_050, 5, 128)


def test_page_2_20_of_a_512_block_wraps_onto_page_0_without_the_widening():
    """At a 512-element block and HKV = 8 a page is 4,096 elements, so page 2^20 is element 2^32:
    in 32 bits it reads page 0's scales, a silent wrong answer rather than a fault."""
    shape = ((1 << 20) + 1, 8, 512)
    assert int(_offset(shape, (1 << 20, 0, 0), ())) == 0
    assert int(_offset(shape, (1 << 20, 0, 0), _wide_positions("mScaleRaw"))) == 1 << 32


#: (tensor, shape, boundary coordinate): the first coordinate of each that reaches element 2^31,
#: every one launchable (request and kv-head below 65,535).
_ROW_BOUNDARIES = [
    ("mQ", (4097, 4096, 128), (4096, 0, 0)),              # batch 4096 at HQ = 4096, D = 128
    ("mOut", (4097, 4096, 128), (4096, 0, 0)),
    ("mOp", (2, 4097, 4096, 128), (0, 4096, 0, 0)),       # the request term
    ("mOp", (65, 8192, 32, 128), (64, 0, 0, 0)),          # the split term: 64 * 8192 * 32 * 128
    ("mLSE", (65, 8192, 4096), (64, 0, 0)),               # the split term
    ("mLSE", (1, 32769, 65536), (0, 32768, 0)),           # the request term at HQ = 65,536
    ("mPT", (32769, 65536), (32768, 0)),                  # 65,536 pages per request
    # D = 256, the grouped body: the output rows are twice as long, so each row term crosses
    # 2^31 at half the coordinate
    ("mQ", (2049, 4096, 256), (2048, 0, 0)),              # batch 2048 at HQ = 4096, D = 256
    ("mOut", (2049, 4096, 256), (2048, 0, 0)),
    ("mOp", (2, 2049, 4096, 256), (0, 2048, 0, 0)),       # the request term
    ("mOp", (33, 8192, 32, 256), (32, 0, 0, 0)),          # the split term: 32 * 8192 * 32 * 256
]


@pytest.mark.parametrize("name,shape,coord", _ROW_BOUNDARIES,
                         ids=[f"{n}-{c}" for n, _, c in _ROW_BOUNDARIES])
def test_each_row_boundary_is_addressed_where_it_is(name, shape, coord):
    strides = _strides(shape)
    exact = sum(c * s for c, s in zip(coord, strides))
    assert exact == 1 << 31
    assert all(c < e for c, e in zip(coord, shape))
    assert int(_offset(shape, coord, _wide_positions(name))) == exact
    assert int(_offset(shape, coord, ())) == -(1 << 31)
    # the element just before is below the boundary, so it does not wrap even unwidened
    before, rem = [], exact - 1
    for s in strides:
        before.append(rem // s)
        rem %= s
    assert int(_offset(shape, before, ())) == exact - 1


def test_the_counter_cell_boundary():
    """cnt_index = batch * HKV + kv_head: at HKV = 65,535, batch 32,768 and kv-head 32,768 is 2^31."""
    batch, hkv, kvh = 32_768, 65_535, 32_768
    assert batch * hkv + kvh == 1 << 31
    assert int(_i32(batch).to(torch.int64) * hkv + _i32(kvh)) == 1 << 31
    assert int(_i32(batch) * hkv + _i32(kvh)) == -(1 << 31)
