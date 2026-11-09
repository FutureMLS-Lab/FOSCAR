# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: MIT
"""Lane maps of the Blackwell kernel's dequant streams, costed in the shared bank model.

The dequant warpgroup of `IntDecodeSm100` runs a K stream on warps 0-1 and a V stream on
warps 2-3. A stream thread (0 to 63) loads 16-byte code chunks from its code ring into
registers and deposits the expanded operand bytes into its operand ring, which the tensor core
reads. This file enumerates every one of those shared-memory instructions, all 32 lanes of each
stream warp, for INT2 and INT4, K and V, at head tiles 4 and 8, at all eight operand-ring start
rows (address bits 7-9, the bits a K-major swizzle reads) and all eight code-ring placements
modulo 1,024, and costs each one in `tests/_smem_banks.py`, the suite's measured bank model.

THE RULE IT ENFORCES. Every code load and every deposit is at ncu's ideal, with one documented
floor: the INT2 deposits write the padded e2m1 operand the `kind::f8f6f4` MMA reads from shared
memory, 8 payload bytes in every 16, so their payload reaches only 16 of the 32 banks, a
wavefront carries at most 64 payload bytes, and a warp's 256 payload bytes take 4 wavefronts
against ncu's ideal of 2. ncu measures exactly that, 4/2, in all 80 INT2 kernels.

WHAT MAKES IT THE KERNEL'S ARITHMETIC, NOT A COPY OF IT. The module imports the DSL, so the
source is read instead:

- the INT4 deposit addresses come from the kernel's own `IntDecodeSm100.deposit_lane_offsets`,
  executed from the source, called with the arguments `_dequant_warpgroup` passes it;
- every code-load address is evaluated from the kernel's own `load_src` calls: the code view
  each call passes, the index each stream thread slices that view with (`dq_tidx`, `i4vt`,
  `vt`, each from its own assignment), the run or slot index of the call, the
  `[None, code_idx]` slice `load_src` reads from, and the word count it reads. Every call on a
  shipped stream's view is costed, so a changed slice index changes the costed addresses;
- every INT2 deposit address is evaluated from the kernel's own `sKf4b[...]` and `sVf4b[...]`
  store subscripts: each index is the kernel's expression, the thread's slots (`sf_rows`,
  `sf_pan0`, `v_row0`, `v_pan00`) come from their own assignments, the page (`pm`, `pk`) from
  the tile loop's own unpacking of `j`, the stage from the operand-ring handle the store
  indexes, and the width from the `cute.make_layout` the store is issued through. Every
  subscript of either view is a costed store, so a changed store index changes the costed
  addresses;
- every other quantity (chunk sizes, run counts, page and run offsets) and every view stride
  (`codeKfS4_w`, `codeVfS4_w`, `codeKfS_w`, `codeVfS_w`, `sKf4b`, `sVf4b`) is the expression
  in `_dequant_warpgroup`, evaluated at the kernel's one page size (64) and head_dim 128 from
  the inputs `kernel` hands it;
- `_dequant_warpgroup` does not read `q_head_tile`, so the maps cannot depend on the head tile.
  Both tiles are still enumerated, because the head-tile-keyed INT4 V map (full channel rows at
  tile 8, whose loads measured 8/4) is exactly what this rule retired.

THE ALIGNED K/V LAYOUT. On the main feeds V is stored exactly like K, so the V stream runs the
K stream's map over its own rings: the same code view shape, the same chunk ownership
`t + 64 q`, and at INT4 the same `deposit_lane_offsets` geometry into an MN_SW128 V stage whose
bytes and swizzle are K_SW128's. The tests below hold the V stream to K's geometry rather than
to a second one.

WHAT IT DOES NOT COVER. The INT2/E4M3 ablation (`dequant_int2_to_f4=0`, which the fold
ablation also takes) is not a shipped path: its V stream reads token columns at a 32-byte
stride, which this model costs at 8/4, and nothing here enforces it. The softmax warpgroup's
scale-record reads are not dequant-stream instructions.

THE MODEL IS ANCHORED BELOW, in both directions: the half-row V deposit on the retired
128-byte V image costs 8/4 at exactly the ring rows where ncu measured 8/4 (and 4/4 at row 6,
the paper configuration's), the retired full-row V load costs 8/4 as ncu measured, and the
pre-#95 token-row V load (one 64-byte token row per thread) costs 16/4 as ncu measured.
"""
from __future__ import annotations

import ast
import builtins
import collections
import copy
import itertools
import pathlib
import re

import pytest

from ._smem_banks import wavefronts

ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNEL = ROOT / "src/foscar/_sm100/int_fold_decode.py"

#: The inputs `kernel` hands `_dequant_warpgroup`, at page 64 and head_dim 128 (the only page
#: size the kernel accepts, `kernel_contract.py`; the only head dim the wrapper accepts).
#: `code_stage_bytes` is one code page (`page_size * mma_tile_k * bits // 8`), and
#: `kv_stage_bytes` one 128 x 128 e4m3 operand stage. The `tiles_*` are the shipped sequence
#: tile, 256 (two MMA tiles of tokens), at head_dim 128 (one of channels); a sequence tile of
#: 128 runs the first half of the same tile loop.
PAGE, MMA_TILE = 64, 128
INPUTS = {
    "page_size": PAGE, "mma_tile_m": MMA_TILE, "mma_tile_k": MMA_TILE,
    "pages_m": MMA_TILE // PAGE, "pages_k": MMA_TILE // PAGE,
    "kv_stage_bytes": MMA_TILE * MMA_TILE, "warpgroup_threads": 128, "k_stages": 3,
    "tiles_sm": 2, "tiles_sk": 2, "tiles_dk": 1, "tiles_dm": 1,
}

#: Operand-ring starts: each of the eight rows, and the four starts production allocates
#: (0x700, 0x780, 0x980 and 0x1180 in `_anchor_fold_outputs.INT4_OPERAND_RING_START`).
OPERAND_RING_STARTS = [0x400 + 128 * r for r in range(8)] + [0x700, 0x780, 0x980, 0x1180]
#: Code-ring starts: TMA-filled, so 128-byte aligned; each placement modulo 1,024.
CODE_RING_STARTS = [0x20000 + 128 * r for r in range(8)]
HEAD_TILES = (4, 8)
#: Stages enumerated per ring. Every stage base is the ring start plus a multiple of the
#: stage size, 16,384 B for the operand ring and one code page for the code ring.
STAGES = range(3)


# ---------------------------------------------------------------------------------------------
# The kernel source: `_dequant_warpgroup`'s expressions and the deposit helper
# ---------------------------------------------------------------------------------------------

def _class():
    tree = ast.parse(KERNEL.read_text())
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "IntDecodeSm100")


def _method(name):
    return next(m for m in _class().body if isinstance(m, ast.FunctionDef) and m.name == name)


@pytest.fixture(scope="module")
def dequant():
    return _method("_dequant_warpgroup")


@pytest.fixture(scope="module")
def helper():
    """`IntDecodeSm100.deposit_lane_offsets`, executed from the kernel source with builtins
    as its only globals. It is plain arithmetic, so the kernel runs the same code on DSL
    Int32 values that this runs on Python ints."""
    fn = next(m for m in _class().body if isinstance(m, ast.FunctionDef)
              and m.name == "deposit_lane_offsets")
    assert [ast.unparse(d) for d in fn.decorator_list] == ["staticmethod"]
    fn = copy.deepcopy(fn)
    fn.decorator_list = []
    namespace: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(KERNEL), "exec"),
         {"__builtins__": builtins}, namespace)
    return namespace["deposit_lane_offsets"]


def _assignment(fn, name):
    """The one assignment of `name` in `fn`, which must be unique."""
    hits = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name]
    assert len(hits) == 1, (name, [h.lineno for h in hits])
    return hits[0]


def _eval(node, env):
    return eval(compile(ast.Expression(node), str(KERNEL), "eval"),
                {"__builtins__": {"min": min}}, dict(env))


#: The scalars `_dequant_warpgroup` derives, in the order it derives them.
DERIVED = ("words", "words_pt", "out_bytes", "stage_words", "kf_chunk_words", "kf_runs",
           "kf_cpr", "kf_out_bytes", "dep_bytes", "i4s_threads", "stream_threads")
INT4_DERIVED = ("kf_row_bytes", "k_page_off", "k_run_off", "vf_row_bytes", "v_page_off",
                "v_run_off")


def _geometry(dequant, bits):
    """Every scalar the stream maps use, evaluated from the kernel's own expressions."""
    env = dict(INPUTS, bits=bits, f4_feed=bits == 2,
               code_stage_bytes=PAGE * MMA_TILE * bits // 8)
    for name in DERIVED + (INT4_DERIVED if bits == 4 else ()):
        env[name] = _eval(_assignment(dequant, name).value, env)
    return env


def _helper_calls(dequant):
    calls = [n for n in ast.walk(dequant) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "self.deposit_lane_offsets"]
    return {ast.unparse(c.args[0]): c for c in calls}


def _sw(x, bits):
    """The K-major swizzle S<bits, 4, 3> on an absolute shared-memory byte address."""
    return x ^ (((x >> 7) & ((1 << bits) - 1)) << 4)


# ---------------------------------------------------------------------------------------------
# The code loads, as the kernel slices and issues them
# ---------------------------------------------------------------------------------------------

#: The code view each shipped stream loads from, and the (width, tensor) it serves. The one
#: other view `load_src` reads, `codeV_w`, is the INT2/E4M3 ablation's (module docstring).
LOAD_VIEWS = {"codeKfS4_w": (4, "K"), "codeVfS4_w": (4, "V"),
              "codeKfS_w": (2, "K"), "codeVfS_w": (2, "V")}
ABLATION_LOAD_VIEWS = {"codeV_w"}
#: Each code view's base: the code ring for K, and the V ring past the K ring's stages.
CODE_VIEW_BASES = {"K": "sCode_w", "V": "sCode_w + self.code_stages_k * stage_words"}

#: A load site: its line, width and tensor, its stream (the `dq_tidx` bound and whether the
#: site is below it), the constexpr loops its view reads, its view, and its word-count argument.
LoadSite = collections.namedtuple("LoadSite", "line bits tensor stream loops view words")


def _assignments(fn):
    """`fn`'s single-name assignments by name, for the many lookups below."""
    index = collections.defaultdict(list)
    for n in ast.walk(fn):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)):
            index[n.targets[0].id].append(n)
    return index


def _one(index, name):
    hits = index.get(name, [])
    assert len(hits) == 1, (name, [h.lineno for h in hits])
    return hits[0].value


class _Scope(dict):
    """The names a kernel expression reads: the ones given, and any other from its one
    assignment in `_dequant_warpgroup`, evaluated on first use. A V thread's `i4vt`, `vt`
    and `v_row0` are therefore the kernel's own expressions of its `dq_tidx`. A name in `local`
    (a store's per-iteration names, `_loop_bindings`) is evaluated from that binding instead."""

    def __init__(self, index, names, local=None):
        super().__init__(names)
        self._index = index
        self._local = local or {}

    def __missing__(self, name):
        if name in self._local:
            node = self._local[name]
            assert node is not None, f"{name} is unpacked from a value this test cannot split"
        else:
            node = _one(self._index, name)
        self[name] = self.eval(node)
        return self[name]

    def eval(self, node):
        return eval(compile(ast.Expression(node), str(KERNEL), "eval"),
                    {"__builtins__": {"min": min}}, self)


def _view_offset(index, node, scope):
    """(u32-word offset from the view's base, strides of the modes left, the view's name) of a
    code-view expression: a `cute.make_tensor` view, a name bound to a slice of one, or a
    subscript of either. A None index keeps its mode; any other is evaluated in `scope`."""
    if isinstance(node, ast.Name):
        value = _one(index, node.id)
        if isinstance(value, ast.Subscript):
            return _view_offset(index, value, scope)
        assert ast.unparse(value.func) == "cute.make_tensor", node.id
        layout = value.args[1]
        assert ast.unparse(layout.func) == "cute.make_layout", node.id
        (stride,) = [k.value for k in layout.keywords if k.arg == "stride"]
        return 0, list(scope.eval(stride)), node.id
    assert isinstance(node, ast.Subscript), ast.unparse(node)
    offset, strides, view = _view_offset(index, node.value, scope)
    indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    assert len(indices) == len(strides), ast.unparse(node)
    kept = []
    for i, stride in zip(indices, strides):
        if isinstance(i, ast.Constant) and i.value is None:
            kept.append(stride)
        else:
            offset += stride * scope.eval(i)
    return offset, kept, view


def _load_src(dequant):
    """`load_src`'s parameters, the default of its word count, the slice of the view it reads
    from, and the expression of the words it reads. It reads contiguous u32 words starting at
    that slice's first element (its `.iterator`)."""
    fn = next(n for n in ast.walk(dequant)
              if isinstance(n, ast.FunctionDef) and n.name == "load_src")
    params = [a.arg for a in fn.args.args]
    (default,) = fn.args.defaults
    parents = {c: n for n in ast.walk(fn) for c in ast.iter_child_nodes(n)}
    (read,) = [n for n in ast.walk(fn) if isinstance(n, ast.Subscript)
               and isinstance(n.value, ast.Name) and n.value.id == params[0]]
    assert isinstance(parents[read], ast.Attribute) and parents[read].attr == "iterator"
    (layout,) = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and ast.unparse(n.func) == "cute.make_layout"]
    (width,) = layout.args
    return params, default, read.slice, width


def _enclosing(parents, node, reads):
    """The stream of the `if dq_tidx < <stream threads>:` branch `node` sits in, as (bound,
    whether in the body), and the enclosing constexpr loops whose variables are in `reads`,
    outermost first, as (variable, range arguments)."""
    streams, loops = [], []
    while node in parents:
        parent = parents[node]
        if (isinstance(parent, ast.If)
                and re.fullmatch(r"dq_tidx < \w+", ast.unparse(parent.test))):
            streams.append((parent.test.comparators[0].id, node in parent.body))
        if isinstance(parent, ast.For) and parent.target.id in reads:
            assert ast.unparse(parent.iter.func) == "cutlass.range_constexpr", (
                ast.unparse(parent))
            loops.append((parent.target.id, parent.iter.args))
        node = parent
    (stream,) = streams
    return stream, loops[::-1]


def _load_sites(dequant, index):
    """Every `load_src` call in `_dequant_warpgroup` on a shipped stream's code view. A site's
    stream is the branch of `if dq_tidx < <stream threads>:` it sits in (the K stream in the
    body, the V stream in the else), and its loops are the enclosing constexpr loops whose
    variables its view expression reads."""
    parents = {c: n for n in ast.walk(dequant) for c in ast.iter_child_nodes(n)}
    sites, views = [], set()
    for call in ast.walk(dequant):
        if not (isinstance(call, ast.Call) and ast.unparse(call.func) == "load_src"):
            continue
        assert not call.keywords and len(call.args) in (3, 4), ast.unparse(call)
        view = call.args[0]
        root = view
        while not isinstance(root, ast.Name) or isinstance(_one(index, root.id), ast.Subscript):
            root = root.value if isinstance(root, ast.Subscript) else _one(index, root.id)
        views.add(root.id)
        if root.id in ABLATION_LOAD_VIEWS:
            continue
        bits, tensor = LOAD_VIEWS[root.id]
        assert ast.unparse(_one(index, root.id).args[0]) == CODE_VIEW_BASES[tensor], root.id
        # The code index is a code-ring pipeline handle's stage.
        assert isinstance(call.args[1], ast.Attribute) and call.args[1].attr == "index"
        reads = {n.id for n in ast.walk(view) if isinstance(n, ast.Name)}
        (bound, in_body), loops = _enclosing(parents, call, reads)
        assert ("K" if in_body else "V") == tensor, (call.lineno, root.id)
        words = call.args[3] if len(call.args) == 4 else None
        sites.append(LoadSite(call.lineno, bits, tensor, (bound, in_body), loops, view, words))
    assert views == set(LOAD_VIEWS) | ABLATION_LOAD_VIEWS, views
    return sites


def _load_offsets(dequant, bits):
    """Per shipped load site of one width: (site, {(dq_tidx, code stage, loop values): byte
    offset from its code ring's start}), every offset evaluated from the kernel's slices and
    `load_src` call, and the load checked to read one 16-byte chunk."""
    index = _assignments(dequant)
    assert ast.unparse(_one(index, "sCode_w")) == "cute.recast_ptr(sCode.iterator, dtype=Int32)"
    params, default, read, width = _load_src(dequant)
    g = _geometry(dequant, bits)
    out = []
    for site in _load_sites(dequant, index):
        if site.bits != bits:
            continue
        words = _Scope(index, g).eval(site.words or default)
        assert 4 * eval(compile(ast.Expression(width), str(KERNEL), "eval"),
                        {"__builtins__": {}}, {params[3]: words}) == 16, site.line
        bound, in_body = site.stream
        n = g[bound]
        threads = range(0, n) if in_body else range(n, g["warpgroup_threads"])
        loop_values = list(itertools.product(
            *[range(*[_Scope(index, g).eval(a) for a in args]) for _name, args in site.loops]))
        load = ast.Subscript(value=site.view, slice=read, ctx=ast.Load())
        table = {}
        for t in threads:
            for s in STAGES:
                for lv in loop_values:
                    scope = _Scope(index, dict(g, dq_tidx=t, **{params[1]: s},
                                               **dict(zip([nm for nm, _a in site.loops], lv))))
                    offset, kept, _view_name = _view_offset(index, load, scope)
                    # One mode is left, the chunk's words, contiguous as `load_src` reads them.
                    assert kept == [1], (site.line, kept)
                    table[(t, s, lv)] = 4 * offset
        out.append((site, table))
    return out


def _enumerate_loads(dequant, bits):
    """Yield (what, (actual, ideal)) for every code load of one width: each shipped load site,
    code-ring placement, stream warp, stage and value of the site's loops."""
    stage_b = _geometry(dequant, bits)["code_stage_bytes"]
    for site, table in _load_offsets(dequant, bits):
        warps = sorted({t // 32 for t, _s, _lv in table})
        loop_values = sorted({lv for _t, _s, lv in table})
        for code in CODE_RING_STARTS:
            ring = code if site.tensor == "K" else code + 8 * stage_b   # past the K code stages
            for w in warps:
                for s in STAGES:
                    for lv in loop_values:
                        acc = [(ln, ring + table[(32 * w + ln, s, lv)]) for ln in range(32)]
                        yield ((f"{site.tensor} load", site.line, code, w, s, lv),
                               wavefronts(acc, 16, "ld"))


# ---------------------------------------------------------------------------------------------
# The deposits, as the kernel computes them
# ---------------------------------------------------------------------------------------------

def _int4_maps(dequant, helper):
    """The INT4 geometry, and per tensor the arguments the kernel passes the deposit helper."""
    g = _geometry(dequant, 4)
    calls = _helper_calls(dequant)
    assert sorted(calls) == ["dq_tidx", "i4vt"]
    args = {}
    for tid, (ring, tensor) in {"dq_tidx": ("k_ring", "K"), "i4vt": ("v_ring", "V")}.items():
        call = calls[tid]
        assert ast.unparse(call.args[1]) == f"{ring}.toint() >> 7", ast.unparse(call.args[1])
        args[tensor] = [_eval(a, g) for a in call.args[2:]]
    # The page and run offsets each deposit adds, and the uniform offsets the helper checks.
    assert args["K"] == [g["kf_row_bytes"], g["kf_out_bytes"],
                         (g["kv_stage_bytes"], g["k_page_off"], g["k_run_off"])]
    assert args["V"] == [g["vf_row_bytes"], g["kf_out_bytes"],
                         (g["kv_stage_bytes"], g["k_stages"] * g["kv_stage_bytes"],
                          g["v_page_off"], g["v_run_off"])]
    return g, args


def _enumerate_int4(dequant, helper, head_tile):
    """Yield (what, (actual, ideal)) for every INT4 dequant-stream instruction."""
    del head_tile                 # the maps do not read it (pinned below)
    yield from _enumerate_loads(dequant, 4)
    g, args = _int4_maps(dequant, helper)
    for ring in OPERAND_RING_STARTS:
        for tensor, runs, base in (
                ("K", 2 * g["kf_runs"], ring),
                ("V", 2 * g["kf_runs"], ring + g["k_stages"] * g["kv_stage_bytes"])):
            row_bytes, run_bytes, uniform = args[tensor]
            page_off = g["k_page_off"] if tensor == "K" else g["v_page_off"]
            run_off = g["k_run_off"] if tensor == "K" else g["v_run_off"]
            lane = {t: helper(t, ring >> 7, row_bytes, run_bytes, uniform) for t in range(64)}
            for warp in (0, 1):
                for s in STAGES:
                    for p in range(2):
                        for q in range(runs):
                            for m in range(run_bytes // 16):
                                acc = [(ln, base + g["kv_stage_bytes"] * s + page_off * p
                                        + run_off * q + lane[32 * warp + ln][m])
                                       for ln in range(32)]
                                yield ((f"{tensor} deposit", ring, warp, s, p, q, m),
                                       wavefronts(acc, 16, "st"))


# ---------------------------------------------------------------------------------------------
# The INT2 deposits, as the kernel indexes and issues them
# ---------------------------------------------------------------------------------------------

#: The operand view each INT2 (f4 feed) stream deposits into, the tensor it serves, and its
#: pinned base: the operand ring for K, and the V ring past the K ring's stages. Both are
#: recast to bytes, so their strides are byte strides.
DEPOSIT_VIEWS = {"sKf4b": "K", "sVf4b": "V"}
DEPOSIT_VIEW_BASES = {
    "K": "cute.recast_ptr(tAsK.iterator, dtype=Uint8)",
    "V": "cute.recast_ptr(tAsK.iterator, dtype=Uint8) + k_stages * kv_stage_bytes",
}
#: The name the stage index is evaluated under: a pipeline handle's stage is a runtime value,
#: so it is replaced by each enumerated stage, as `load_src`'s code stage is.
STAGE = "__stage__"

#: A deposit site: its line, view and tensor, its stream (as `LoadSite`), the constexpr loops
#: and the per-iteration bindings its subscript reads, its subscript with the stage index
#: replaced by `STAGE`, the operand-block index of the handle it stores through, and its width.
DepositSite = collections.namedtuple(
    "DepositSite", "line view tensor stream loops local subscript block width")


def _bound_names(node):
    """(name, value or None) for every name `node` binds by assignment; a tuple unpacking of
    anything but a tuple display binds its names to None."""
    if isinstance(node, ast.Assign):
        pairs = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                pairs.append((target.id, node.value))
            elif isinstance(target, ast.Tuple):
                values = (node.value.elts if isinstance(node.value, ast.Tuple)
                          and len(node.value.elts) == len(target.elts)
                          else [None] * len(target.elts))
                pairs += [(t.id, v) for t, v in zip(target.elts, values)
                          if isinstance(t, ast.Name)]
        return pairs
    if isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
        return [(node.target.id, None)] if isinstance(node.target, ast.Name) else []
    return []


def _loop_bindings(parents, node):
    """The names bound per iteration of the loops enclosing `node`, each from the assignment
    before it in one of those loop bodies, the nearest one winning; and the outermost of those
    loops."""
    local, outer = {}, None
    while node in parents:
        parent = parents[node]
        if isinstance(parent, ast.For) and node in parent.body:
            outer = parent
            for stmt in reversed(parent.body[:parent.body.index(node)]):
                for name, value in _bound_names(stmt):
                    local.setdefault(name, value)
        node = parent
    return local, outer


def _reads(node, local):
    """Every name `node` reads, directly or through the per-iteration bindings it reads."""
    seen, todo = set(), [node]
    while todo:
        for n in ast.walk(todo.pop()):
            if isinstance(n, ast.Name) and n.id not in seen:
                seen.add(n.id)
                if local.get(n.id) is not None:
                    todo.append(local[n.id])
    return seen


def _deposit_sites(dequant, index):
    """Every subscript of `sKf4b` and `sVf4b` in `_dequant_warpgroup`. Each must be the
    destination of a store, `cute.autovec_copy(<registers>, dst)` with `dst =
    cute.make_tensor(<view>[...].iterator.align(16), cute.make_layout(<width>))`, whose stage
    index is its own tensor's operand-ring handle `<k|v>_kv_handles[<block>].index`."""
    parents = {c: n for n in ast.walk(dequant) for c in ast.iter_child_nodes(n)}
    sites = []
    for name in ast.walk(dequant):
        if not (isinstance(name, ast.Name) and name.id in DEPOSIT_VIEWS
                and isinstance(name.ctx, ast.Load)):
            continue
        tensor = DEPOSIT_VIEWS[name.id]
        sub = parents[name]
        assert isinstance(sub, ast.Subscript) and sub.value is name, name.lineno
        aligned = parents[parents[parents[sub]]]
        make = parents[aligned]
        assign = parents[make]
        assert ast.unparse(aligned) == f"{ast.unparse(sub)}.iterator.align(16)", sub.lineno
        assert (isinstance(make, ast.Call) and ast.unparse(make.func) == "cute.make_tensor"
                and make.args[0] is aligned and len(make.args) == 2), sub.lineno
        layout = make.args[1]
        assert (ast.unparse(layout.func) == "cute.make_layout" and len(layout.args) == 1
                and not layout.keywords), sub.lineno
        assert isinstance(assign, ast.Assign) and assign.value is make, sub.lineno
        (dst,) = assign.targets
        body = parents[assign].body
        copies = [st for st in body[body.index(assign) + 1:] if isinstance(st, ast.Expr)
                  and isinstance(st.value, ast.Call)
                  and ast.unparse(st.value.func) == "cute.autovec_copy"
                  and ast.unparse(st.value.args[1]) == dst.id]
        assert len(copies) == 1, (sub.lineno, "the view is not stored through")
        indices = list(sub.slice.elts)
        stages = [k for k, i in enumerate(indices)
                  if isinstance(i, ast.Attribute) and i.attr == "index"]
        assert len(stages) == 1, sub.lineno
        handle = indices[stages[0]].value
        assert (isinstance(handle, ast.Subscript)
                and ast.unparse(handle.value) == f"{tensor.lower()}_kv_handles"), sub.lineno
        indices[stages[0]] = ast.copy_location(ast.Name(id=STAGE, ctx=ast.Load()),
                                               indices[stages[0]])
        subscript = ast.Subscript(value=name, slice=ast.Tuple(elts=indices, ctx=ast.Load()),
                                  ctx=ast.Load())
        local, outer = _loop_bindings(parents, sub)
        reads = _reads(sub, local)
        # A per-iteration name the store reads is bound once in the tile loop, so the binding
        # evaluated here is the only one that can reach the store.
        count = collections.Counter(nm for n in ast.walk(outer) for nm, _v in _bound_names(n))
        assert all(count[nm] == 1 for nm in reads & local.keys()), (sub.lineno, count)
        stream, loops = _enclosing(parents, sub, reads)
        assert ("K" if stream[1] else "V") == tensor, (sub.lineno, name.id)
        sites.append(DepositSite(sub.lineno, name.id, tensor, stream, loops, local, subscript,
                                 handle.slice, layout.args[0]))
    assert sorted(site.view for site in sites) == sorted(DEPOSIT_VIEWS), sites
    return sites


#: The deposits of one site: the byte offset of its view's base from the operand ring's start,
#: the store width in bytes, and {(dq_tidx, stage, loop values): (byte offset from the view's
#: base, operand block)}.
Deposits = collections.namedtuple("Deposits", "site base width table")


def _deposit_offsets(dequant):
    """Per INT2 deposit site, every address it stores to, evaluated from its subscript."""
    index = _assignments(dequant)
    g = _geometry(dequant, 2)
    out = []
    for site in _deposit_sites(dequant, index):
        view = _one(index, site.view)
        assert ast.unparse(view.args[0]) == DEPOSIT_VIEW_BASES[site.tensor], site.view
        base = (0 if site.tensor == "K" else _Scope(index, g).eval(view.args[0].right))
        width = _Scope(index, g, site.local).eval(site.width)
        bound, in_body = site.stream
        n = g[bound]
        threads = range(n) if in_body else range(n, g["warpgroup_threads"])
        names = [nm for nm, _a in site.loops]
        loop_values = list(itertools.product(
            *[range(*[_Scope(index, g).eval(a) for a in args]) for _n, args in site.loops]))
        table = {}
        for t in threads:
            for s in STAGES:
                for lv in loop_values:
                    scope = _Scope(index, dict(g, dq_tidx=t, **{STAGE: s}, **dict(zip(names, lv))),
                                   site.local)
                    offset, kept, _view_name = _view_offset(index, site.subscript, scope)
                    # One mode is left, the chunk's bytes, contiguous as the copy writes them.
                    assert kept == [1], (site.line, kept)
                    table[(t, s, lv)] = (offset, scope.eval(site.block))
        out.append(Deposits(site, base, width, table))
    return out


def _enumerate_int2(dequant, head_tile):
    """Yield (what, (actual, ideal)) for every INT2 (f4 feed) dequant-stream instruction: each
    load as `_enumerate_loads`, and each deposit site at every operand-ring start, stream warp,
    stage and value of the site's loops."""
    del head_tile
    yield from _enumerate_loads(dequant, 2)
    g = _geometry(dequant, 2)
    for site, base, width, table in _deposit_offsets(dequant):
        assert width == g["dep_bytes"] == 8, (site.line, width)
        warps = sorted({t // 32 for t, _s, _lv in table})
        loop_values = sorted({lv for _t, _s, lv in table})
        for ring in OPERAND_RING_STARTS:
            for w in warps:
                for s in STAGES:
                    for lv in loop_values:
                        acc = [(ln, ring + base + table[(32 * w + ln, s, lv)][0])
                               for ln in range(32)]
                        yield ((f"{site.tensor} deposit", site.line, ring, w, s, lv),
                               wavefronts(acc, width, "st"))


# ---------------------------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("head_tile", HEAD_TILES)
def test_every_int4_load_and_deposit_is_at_its_ideal(dequant, helper, head_tile):
    costs = collections.Counter()
    for what, cost in _enumerate_int4(dequant, helper, head_tile):
        assert cost == (4, 4), (what, cost)
        costs[what[0]] += 1
    # 2 load sites (a tile's first code page and the prefetch of the next) x 8 code placements
    # x 2 warps x 3 stages x 4 runs per load; 12 ring starts x 2 warps x 3 stages x 2 pages x
    # 4 runs x 2 halves per deposit.
    assert costs == {"K load": 384, "V load": 384, "K deposit": 1152, "V deposit": 1152}


@pytest.mark.parametrize("head_tile", HEAD_TILES)
def test_every_int2_load_is_at_its_ideal_and_every_deposit_at_the_padded_format_floor(
        dequant, head_tile):
    costs = collections.Counter()
    for what, cost in _enumerate_int2(dequant, head_tile):
        if what[0].endswith("load"):
            assert cost == (4, 4), (what, cost)
        else:
            # The floor: 32 lanes x 8 payload bytes, on 16 of the 32 banks.
            assert cost == (4, 2), (what, cost)
        costs[what[0]] += 1
    # 2 load sites x 8 code placements x 2 warps x 3 stages x 2 slots per load; 12 ring starts
    # x 2 warps x 3 stages x 4 code pages per tile x 2 slots x 4 words per deposit.
    assert costs == {"K load": 192, "V load": 192, "K deposit": 2304, "V deposit": 2304}


def test_the_int2_floor_is_the_physical_minimum_of_the_padded_format():
    """8 payload bytes at the start of every 16-byte chunk sit on 16 of the 32 banks, whatever
    the chunk's address, so a wavefront carries at most 16 x 4 = 64 payload bytes and a warp's
    32 x 8 = 256 need 4. Checked over every 16-byte-aligned chunk address in two rows."""
    banks = {((16 * chunk + b) // 4) % 32 for chunk in range(16) for b in range(8)}
    assert len(banks) == 16
    assert 32 * 8 // (len(banks) * 4) == 4


def test_the_dequant_streams_do_not_read_the_head_tile(dequant):
    """What makes the two head tiles run one map: the dequant warpgroup never reads
    `q_head_tile`. Until 2026-09-30 it chose the INT4 V map by it."""
    reads = [n.lineno for n in ast.walk(dequant) if isinstance(n, ast.Attribute)
             and n.attr == "q_head_tile"]
    assert reads == []


def test_the_v_stream_takes_the_k_streams_map(dequant, helper):
    """V stored like K means the V stream runs K's map: each width's V code view has the K
    view's shape and strides over its own ring, and the INT4 V deposit has K's row, run, page
    and run-offset geometry."""
    index = _assignments(dequant)
    for k_view, v_view in (("codeKfS4_w", "codeVfS4_w"), ("codeKfS_w", "codeVfS_w")):
        k_layout, v_layout = (_one(index, n).args[1] for n in (k_view, v_view))
        k_text = ast.unparse(k_layout).replace("code_stages", "self.code_stages_v")
        assert ast.unparse(v_layout) == k_text, (k_view, v_view)
    g, args = _int4_maps(dequant, helper)
    assert args["V"][:2] == args["K"][:2]
    assert (g["vf_row_bytes"], g["v_page_off"], g["v_run_off"]) == (
        g["kf_row_bytes"], g["k_page_off"], g["k_run_off"])


def test_each_code_page_is_loaded_once(dequant):
    """At every load site the stream's threads and the site's loops read every 16-byte chunk of
    a code page exactly once, at both widths, so the loads costed above are the partition and
    not a subset of it; and stage s reads the same chunks one code page further on."""
    for bits in (4, 2):
        stage_b = _geometry(dequant, bits)["code_stage_bytes"]
        sites = _load_offsets(dequant, bits)
        assert sorted(site.tensor for site, _table in sites) == ["K", "K", "V", "V"], bits
        for site, table in sites:
            starts = sorted(off for (_t, s, _lv), off in table.items() if s == 0)
            assert starts == list(range(0, stage_b, 16)), (bits, site.line)
            for (t, s, lv), off in table.items():
                assert off == table[(t, 0, lv)] + s * stage_b, (bits, site.line, t, s, lv)


#: The INT2 operand image the `kind::f8f6f4` MMA reads: the 16-byte chunk of (token row r,
#: 16-channel panel c) sits at byte 2048 c + 16 r of the stage, for K (the K-major panel image)
#: and for V (the same bytes read MN-major, V stored like K). Page p of an operand block holds
#: tokens [64 p, 64 p + 64).
INT2_PAGE_CHUNKS = {
    t: (lambda p: [2048 * c + 16 * (64 * p + r) for r in range(64) for c in range(8)])
    for t in ("K", "V")
}


def test_each_int2_code_page_deposits_its_slice_of_the_operand_image_once(dequant):
    """At both deposit sites, the stream's threads and the site's slot and word loops write the
    8 payload bytes of every 16-byte chunk of their code page's slice of the operand image
    exactly once, so the deposits costed above are the partition and not a subset of it. Code
    page j of a tile is page j mod 2 of operand block j div 2, the block whose stage handle the
    store indexes, and stage s writes the same chunks one stage further on."""
    g = _geometry(dequant, 2)
    deposits = _deposit_offsets(dequant)
    assert sorted(d.site.tensor for d in deposits) == ["K", "V"]
    for site, _base, _width, table in deposits:
        pages = g["pages_m"] if site.tensor == "K" else g["pages_k"]
        jpos = [name for name, _args in site.loops].index("j")
        chunks = collections.defaultdict(list)
        for (t, s, lv), (off, block) in table.items():
            j = lv[jpos]
            assert block == j // pages, (site.line, t, s, lv, block)
            assert off == table[(t, 0, lv)][0] + s * g["kv_stage_bytes"], (site.line, t, s, lv)
            if s == 0:
                chunks[j].append(off)
        assert len(chunks) > pages, (site.line, sorted(chunks))
        for j, offs in chunks.items():
            assert sorted(offs) == sorted(INT2_PAGE_CHUNKS[site.tensor](j % pages)), (
                site.line, j)


# ---------------------------------------------------------------------------------------------
# The INT4 deposit helper writes the operand image the tensor core reads
# ---------------------------------------------------------------------------------------------

#: The operand byte of (row r, byte b of the page's slice of that row) in page p of a stage,
#: as each DSL layout places it (printed on B200): K_SW128
#: `((128,32),1,4,S):((128,1),0,32,16384)` puts token row 64 p + r at 128 (64 p + r) + b, and
#: the V layout, MN_SW128 (`make_smem_layout_a` for the MN-major V operand),
#: `((128,32),1,4,S):((1,128),0,4096,16384)`, puts channel b of token 64 p + r at the same
#: byte. The swizzle is taken on the absolute address.
LOGICAL = {
    "K": (3, lambda p, r, b: 128 * (64 * p + r) + b),
    "V": (3, lambda p, r, b: 128 * (64 * p + r) + b),
}


@pytest.mark.parametrize("tensor", sorted(LOGICAL))
def test_each_int4_deposit_lands_where_the_operand_layout_puts_its_byte(
        dequant, helper, tensor):
    """For every thread, ring start, stage, page, run and 16-byte half, the kernel's address
    (stage base + page offset + run offset + lane offset) is the swizzled address of the byte
    the layout puts there. Run q of thread t is code chunk c = t + 64 q, which expands to
    page row c // cpr at byte 32 (c % cpr)."""
    g, args = _int4_maps(dequant, helper)
    row_bytes, run_bytes, uniform = args[tensor]
    bits, logical = LOGICAL[tensor]
    cpr = row_bytes // run_bytes
    page_off = g["k_page_off"] if tensor == "K" else g["v_page_off"]
    run_off = g["k_run_off"] if tensor == "K" else g["v_run_off"]
    runs = 2 * g["kf_runs"]
    n = 0
    for ring in OPERAND_RING_STARTS:
        for t in range(64):
            lane = helper(t, ring >> 7, row_bytes, run_bytes, uniform)
            for s in STAGES:
                for p in range(2):
                    for q in range(runs):
                        c = t + 64 * q
                        for m in range(run_bytes // 16):
                            want = _sw(ring + g["kv_stage_bytes"] * s + logical(
                                p, c // cpr, run_bytes * (c % cpr) + 16 * m), bits)
                            got = (ring + g["kv_stage_bytes"] * s + page_off * p
                                   + run_off * q + lane[m])
                            assert got == want, (tensor, hex(ring), t, s, p, q, m)
                            n += 1
    assert n == len(OPERAND_RING_STARTS) * 64 * len(STAGES) * 2 * 4 * 2


@pytest.mark.parametrize("tensor", sorted(LOGICAL))
def test_one_pages_int4_deposits_write_its_slice_once(dequant, helper, tensor):
    g, args = _int4_maps(dequant, helper)
    row_bytes, run_bytes, uniform = args[tensor]
    run_off = g["k_run_off"] if tensor == "K" else g["v_run_off"]
    page_off = g["k_page_off"] if tensor == "K" else g["v_page_off"]
    runs = 2 * g["kf_runs"]
    for ring in OPERAND_RING_STARTS:
        units = collections.Counter(
            (ring + run_off * q + off) // 16
            for t in range(64)
            for off in helper(t, ring >> 7, row_bytes, run_bytes, uniform)
            for q in range(runs))
        assert set(units.values()) == {1} and len(units) == page_off // 16, (tensor, ring)
        assert min(units) == ring // 16


@pytest.mark.parametrize(("args", "match"), [
    ((0, 0, 96, 32, ()), "no K-major swizzle spans 96"),
    ((0, 0, 128, 24, ()), "does not tile"),
    ((0, 0, 64, 48, ()), "does not tile"),
    ((0, 0, 128, 32, (16384, 512)), "not a multiple of the swizzle period 1024"),
    ((0, 0, 64, 32, (256,)), "not a multiple of the swizzle period 512"),
    ((0, 0, 64, 32, (512.0,)), "not a multiple"),
    ((0, 0, 128.0, 32, ()), "positive int"),
    ((0, 0, 128, 0, ()), "positive int"),
])
def test_the_helper_refuses_a_geometry_it_does_not_model(helper, args, match):
    with pytest.raises(ValueError, match=match):
        helper(*args)


def test_the_deposits_add_the_offsets_the_helper_was_checked_against(dequant):
    """The deposit address is the stage base, the lane offset and the page and run offsets the
    helper's period check saw, and nothing else; and the stage base is a whole stage."""
    exprs = {ast.unparse(n) for n in ast.walk(dequant) if isinstance(n, ast.BinOp)}
    assert "k_dst + (pm * k_page_off + _q * k_run_off)" in exprs
    assert "v_dst + (pk * v_page_off + q * v_run_off)" in exprs
    for ptr, lane, index in (("k_stage", "k_lane", "k_kv_handles[kb].index"),
                             ("v_stage", "v_lane", "v_kv_handles[vb].index")):
        assert f"{ptr} + {lane}[_m]" in exprs
        base = {"k_stage": "k_ring", "v_stage": "v_base"}[ptr]
        assert ast.unparse(_assignment(dequant, ptr).value) == (
            f"{base} + cute.assume({index} * kv_stage_bytes, divby=kv_stage_bytes)")
    assert ast.unparse(_assignment(dequant, "v_base").value) == (
        "v_ring + k_stages * kv_stage_bytes")


def test_the_int4_v_stages_take_the_layouts_own_swizzle_on_the_host_and_on_the_device(
        dequant):
    """The image the helper addresses is the one the MMA reads: the host builds the V layout
    with `make_smem_layout_a` alone (MN_SW128 for the MN-major V operand, no override), the
    device's V view keeps the ring's 128-byte swizzle and asserts that it is the layout's, and
    the V deposit asserts that this swizzle spans the token row it addresses."""
    source = KERNEL.read_text()
    assert "K_SW64" not in source
    kernel = _method("kernel")
    views = [n for n in ast.walk(kernel) if isinstance(n, ast.Assign)
             and ast.unparse(n.targets[0]) == "tAsV"]
    assert len(views) == 1
    assert ast.unparse(views[0].value) == "cute.make_tensor(tAsK.iterator, smem_layout_v_mma.outer)"
    asserts = [ast.unparse(n.test) for n in ast.walk(kernel) if isinstance(n, ast.Assert)]
    assert ("tAsV.iterator.type.swizzle_type == smem_layout_v_mma.inner.type" in asserts)
    deq_asserts = [ast.unparse(n.test) for n in ast.walk(dequant) if isinstance(n, ast.Assert)]
    assert any("16 << swz_v.num_bits == vf_row_bytes" in a for a in deq_asserts)


# ---------------------------------------------------------------------------------------------
# Negative controls: the model sees the conflicts ncu measured before this rule
# ---------------------------------------------------------------------------------------------

def test_the_retired_half_row_map_on_the_128_byte_v_image_conflicts_where_ncu_measured():
    """Before 2026-09-30 the INT4 V stage carried the 128-byte swizzle. The same chunk map
    (row c // 2, token half c % 2, 128-byte rows with the page at byte 64 pk) is 8/4 at ring
    rows 0, 1, 3, 4, 5 and 7, and 4/4 at rows 2 and 6. ncu measured 8/4 in the six INT4 pairs
    whose ring starts at rows 0, 1, 3, 4 and 7, and 4/4 at row 6 (the token-128 pair, the
    paper configuration)."""
    conflicted = set()
    for row in range(8):
        av = 0x10000 + 128 * row
        for warp in (0, 1):
            for pk in (0, 1):
                for q in range(4):
                    for m in (0, 1):
                        acc = []
                        for ln in range(32):
                            c = 32 * warp + ln + 64 * q
                            acc.append((ln, _sw(av + 128 * (c // 2) + 64 * pk
                                                + 32 * (c % 2) + 16 * m, 3)))
                        cost = wavefronts(acc, 16, "st")
                        assert cost in ((4, 4), (8, 4)), cost
                        if cost == (8, 4):
                            conflicted.add(row)
    assert conflicted == {0, 1, 3, 4, 5, 7}


def test_the_retired_full_row_v_load_is_8_over_4():
    """The retired head-tile-8 V map loaded channel row t' + 64 q whole, 32 bytes as two
    16-byte loads, so each quarter warp read 256 bytes at a 32-byte stride: 8/4 at every code
    placement, as ncu measured in all 20 INT4 pairs at head tile 8."""
    for code in CODE_RING_STARTS:
        for warp in (0, 1):
            for q in (0, 1):
                for h in (0, 1):
                    acc = [(ln, code + 32 * (32 * warp + ln + 64 * q) + 16 * h)
                           for ln in range(32)]
                    assert wavefronts(acc, 16, "ld") == (8, 4)


def test_the_pre_95_token_row_v_load_is_16_over_4():
    """The earlier MN-major INT4 V stream gave each thread one whole 64-byte token row,
    loaded as four 16-byte loads, so each quarter warp read 16 bytes at a 64-byte lane stride,
    on 2 of the 8 16-byte bank groups: 16/4 at every code placement. ncu measured
    16 wavefronts per execution against 4 at each of the 16 V-stream LDS.128 sites of
    that layout. The aligned layout reads the same token
    rows with the K stream's chunk ownership, which `test_every_int4_load_and_deposit_is_at_
    its_ideal` holds at 4/4."""
    for code in CODE_RING_STARTS:
        for warp in (0, 1):
            for h in range(4):
                acc = [(ln, code + 64 * (32 * warp + ln) + 16 * h) for ln in range(32)]
                assert wavefronts(acc, 16, "ld") == (16, 4)


def test_a_swizzle_key_taken_from_the_ring_start_misplaces_the_deposits(dequant, helper):
    """The key is the ABSOLUTE address's row bits. Told row 0 instead, the helper still
    agrees with the operand layout where the ring's own row bits are 0 in the key's width,
    and nowhere else, so the placement test above can fail."""
    g, args = _int4_maps(dequant, helper)
    for tensor in ("K", "V"):
        row_bytes, run_bytes, uniform = args[tensor]
        bits, logical = LOGICAL[tensor]
        cpr = row_bytes // run_bytes
        for row in range(8):
            ring = 0x400 + 128 * row
            wrong = 0
            for t in range(64):
                lane = helper(t, 0, row_bytes, run_bytes, uniform)
                for m in range(run_bytes // 16):
                    want = _sw(ring + logical(0, t // cpr, run_bytes * (t % cpr) + 16 * m),
                               bits)
                    wrong += ring + lane[m] != want
            assert (wrong == 0) == (row % (1 << bits) == 0), (tensor, row, wrong)
