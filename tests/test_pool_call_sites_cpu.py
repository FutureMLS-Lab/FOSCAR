# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""Every pool builder call states its K store order, and every pool-unpacking
reference inverts it.

WHY A CALL-SITE GATE AND NOT A DEFAULT. `make_pool_problem(k_store_perm=None)`
means LOGICAL order, so the argument cannot carry a "resolve it for me"
sentinel the way `v_store_perm=None` does: K's identity order and its absent
argument are the same value. A caller that omits it therefore gets logical
order silently, which since cutedsl-decode PR #55 disagrees with what an int2
f4 token-wise-K kernel reads, and the disagreement produces finite,
plausible-looking output with every K channel taken from the wrong slot inside
its 16-channel group. Nothing raises.

That is exactly what happened: commit 820fb5f flipped the kernel to the
permuted read and eight call sites kept building logical-order pools, including
the flex numerics gate, the public-wrapper equivalence gate and the bit-identity
anchor. Two of those three compare a kernel against another kernel over the same
bytes, so they could not see it by construction. Measured on B200 GPU 7 at
pristine 820fb5f, the flex gate's bits=2 token-wise-K cells read rel_l2 of 1.031
against a 5e-3 bar while bits=4 sat at 1.77e-3. This file is the gate those
gates could not be.

IT PARSES RATHER THAN GREPS, and the difference is not stylistic. An earlier
revision matched call names textually and missed
`from _fold_harness import make_pool_problem as make_problem`, which
`scripts/_e2e_fold_check.py` uses: the one call site an aliasing import hides is
the one a textual scan cannot see, so the gate would have reported coverage it
did not have. `ast.parse` reads source only and imports nothing, so this still
runs on a CPU box with no CUDA.

WHAT THIS RULE CANNOT DO, stated because a gate believed stronger than it is is
worse than a weak one. It detects OMISSION: a scope that unpacks raw codes and
never indexes them by a permutation. It cannot detect an index expression that
EVALUATES to identity, and no syntactic rule can, because that is a value
property. Codex demonstrated `codes[..., k_store_perm.sort().values]`, which sorts
a permutation into identity and satisfies this rule;
`test_the_rule_cannot_see_a_value` pins that as a known limitation rather than
leaving it to be rediscovered. Nor can the rule require the index to name an
inverse: int2's sigma is SELF-INVERSE, so `codes[..., k_store_perm]` is correct
there, while int4's t8 (2026-08-15) is NOT, so the same expression on an int4
unpack is wrong at six of eight channels — a VALUE property this syntactic rule
cannot see. `test_kperm_sm100_cpu.test_int4_reference_inverts_the_perm` is the
round-trip that catches an int4 forward-index regression.

WHAT CATCHES THE REST is the numerics gate, and that is measured rather than hoped:
a K pool read in the wrong order reads rel_l2 near 1.0 against an independent
reference, which is what `scripts/test_fold_quant_flex_sm100.py` reported at
pristine 820fb5f, 1.031 against a 5e-3 bar. The layering is deliberate. This file
catches at build time, on CPU, the omission that has actually happened eight times
in this feature's history; the flex, validate and accuracy gates catch any ordering
wrong in value, on a GPU, loudly.
"""
from __future__ import annotations

import ast
import io
import pathlib
import subprocess
import tokenize

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Builders whose K order is a pool contract rather than a tuning knob.
BUILDERS = ("make_pool_problem", "make_pool_problem_flex")

#: References that unpack mKq and therefore have to invert the stored order.
#:
#: `unpack_rows_lut` IS TRACKED, and it is the one that matters most. It is the
#: raw code unpacker, so a caller who reaches for it is doing by hand what
#: `dequant_pools_flex` does with the order threaded through, and nothing about
#: the call announces that. `test_fold_quant_flex_sm100.ref_channel_k_flex`
#: unpacked K with it and was the single reason channel-wise K could not carry the
#: store permutation until 2026-08-13, so leaving it untracked would let that fix
#: be reverted with this gate still green.
UNPACKERS = ("dequant_pools_flex", "ref_paged_attention_fp8q_flex")

#: Raw unpackers, which do NOT take the order: a caller unpacks and then inverts
#: the result, as `repack_sm100.dequant_pools_lut` does. So the accounting is a
#: property of the enclosing FUNCTION rather than of the call, and the rule is that the
#: function must either take `k_store_perm` as a parameter or apply
#: `_k_store_perm_inverse`. Both are read from the AST, never from text: at the
#: time this was written `ref_channel_k_flex` mentioned `k_store_perm` three
#: times IN ITS DOCSTRING while doing nothing about it, so a textual rule would
#: have called the one broken site clean.
RAW_UNPACKERS = ("unpack_rows_lut",)

#: What counts as a function accounting for the order. This is a NAMING
#: convention and it is enforced as one: the repo calls a K permutation
#: `k_store_perm`, `k_perm`, `kperm` or `_kp`, and `scripts/_e2e_fold_check.py`
#: builds the inverse inline from `_kp` rather than calling the helper. A scope
#: that handles the order under some novel name must either rename to match or
#: carry the logical marker, which is a deliberate cost: a gate that accepted any
#: plausible-looking name would accept a scope that only mentions one.
ACCOUNTS_SUBSTRINGS = ("k_store_perm", "k_perm", "kperm")
ACCOUNTS_EXACT = ("_kp",)

#: A call site may instead DECLARE that logical order is what it wants. Some
#: genuinely do: the unpermuted arm of a paired comparison is unpermuted on
#: purpose, and a pack/unpack round trip has no stored order on either side. The
#: marker must carry a reason after the colon, so the claim is auditable rather
#: than a mute silencer.
MARKER = "k-order: logical"

#: How far above a call a declaration may sit. A declaration never reaches past
#: another call of the same family, so a marker cannot excuse its neighbour: in
#: tests/test_kperm_sm100_cpu.py the logical arm's marker sits a few lines above
#: the PERMUTED arm's call, and an earlier revision let it excuse that call too,
#: which would have retired the one property that paired test exists to hold.
MARKER_REACH = 6


def _sources():
    """Every TRACKED Python file in the repository.

    Enumerated from `git ls-files` rather than from a directory list, following
    `tests/test_one_timing_path_cpu.py`. A hand-written list of directories is a
    hole with a schedule: a builder call added under `src/` or a new top-level
    directory would bypass a gate that only walked `scripts/`, `baselines/`
    and `tests/`, and nothing would say so.
    """
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.py"],
                         capture_output=True, text=True, check=True)
    for rel in out.stdout.split("\n"):
        rel = rel.strip()
        if not rel:
            continue
        path = ROOT / rel
        if path.is_file():
            yield rel, path.read_text(encoding="utf-8")


def _marker_lines(text: str) -> set[int]:
    """Line numbers of `# k-order: logical <reason>` comments.

    Read through `tokenize`, so a marker written inside a string literal is not
    mistaken for a declaration.
    """
    out: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type != tokenize.COMMENT:
                continue
            body = tok.string.lstrip("#").strip()
            if body.startswith(MARKER) and body[len(MARKER):].strip(" -:"):
                out.add(tok.start[0])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


def _aliases(tree: ast.AST, names: tuple[str, ...]) -> dict[str, str]:
    """Local name -> tracked name, following `import X as Y`."""
    out = {n: n for n in names}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if a.name in names and a.asname:
                    out[a.asname] = a.name
    return out


def _calls(text: str, names: tuple[str, ...], keyword: str = "k_store_perm"):
    """Yield (tracked_name, lineno, states_the_order) for every call.

    Covers `f(...)`, `mod.f(...)` and aliased imports. A call that forwards
    `**kwargs` is NOT credited, because whether the order is in that dict is
    exactly what a reader cannot see. `keyword` names the argument that states
    the order: `k_store_perm` for the SM100 builders, `order` for the SM90 ones.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    alias = _aliases(tree, names)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        local = (func.id if isinstance(func, ast.Name)
                 else func.attr if isinstance(func, ast.Attribute) else None)
        tracked = alias.get(local)
        if tracked is None:
            continue
        yield tracked, node.lineno, any(
            kw.arg == keyword for kw in node.keywords)


def _unstated(names: tuple[str, ...]):
    """Call sites that neither state the order nor declare logical order."""
    bad = []
    for rel, text in _sources():
        markers = _marker_lines(text)
        calls = sorted(_calls(text, names), key=lambda c: c[1])
        for i, (tracked, lineno, stated) in enumerate(calls):
            if stated:
                continue
            # A declaration reaches back MARKER_REACH lines, but never past the
            # previous call of this family.
            floor = lineno - MARKER_REACH
            if i:
                floor = max(floor, calls[i - 1][1] + 1)
            if any(floor <= m <= lineno for m in markers):
                continue
            bad.append(f"{rel}:{lineno}: {tracked}")
    return bad


@pytest.mark.parametrize("builder", BUILDERS)
def test_every_pool_build_states_its_k_order(builder):
    bad = _unstated((builder,))
    assert not bad, (
        f"{len(bad)} {builder} call site(s) do not state k_store_perm, so they "
        "build a LOGICAL-order K pool by default. That is wrong for EVERY int4 "
        "kernel (t8 shuffle, 2026-08-15) and for any int2 f4 kernel at either "
        "K axis, and it fails silently. Pass k_pool_perm(bits, kkw), "
        f"or write `# {MARKER} <reason>` directly above the call if logical "
        "order is what this site wants:\n  " + "\n  ".join(bad))


def _perm_mentioned(node) -> bool:
    """Does this expression name a K permutation, other than as a constant?"""
    def aware(n) -> bool:
        name = (n.id if isinstance(n, ast.Name)
                else n.attr if isinstance(n, ast.Attribute) else None)
        return name is not None and (
            name in ACCOUNTS_EXACT
            or any(s in name for s in ACCOUNTS_SUBSTRINGS))
    return any(aware(n) for n in ast.walk(node))


def _unpacked_names(fn: ast.AST, local_names: set[str]) -> set[str]:
    """Names bound to the result of a raw unpack inside `fn`."""
    out = set()
    for node in ast.walk(fn):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None:
            continue
        calls = [n for n in ast.walk(value)
                 if isinstance(n, ast.Call)
                 and ((isinstance(n.func, ast.Name) and n.func.id in local_names)
                      or (isinstance(n.func, ast.Attribute)
                          and n.func.attr in local_names))]
        if not calls:
            continue
        targets = ([node.target] if isinstance(node, ast.AnnAssign)
                   else node.targets)
        for tgt in targets:
            for n in ast.walk(tgt):
                if isinstance(n, ast.Name):
                    out.add(n.id)
    return out


def _reorders_the_unpacked_codes(fn: ast.AST, local_names: set[str]) -> bool:
    """Do the UNPACKED CODES themselves get reordered by a permutation?

    This is a dataflow question and the two syntactic versions before it were both
    defeated. The first accepted any function that NAMED a perm, so an unused
    parameter passed. The second accepted a perm appearing in any call or
    subscript, and codex broke it by executing `validate(k_store_perm)` and
    `g(k_store_perm=None)`, neither of which touches the codes. What has to be
    true is narrower: the value that came out of the unpack must be indexed by
    something perm-derived.

    Two forms count, and both require the codes AND the perm in one expression:
    ONLY a subscript counts, and its result must rebind the codes or be returned.
    A call taking both the codes and a perm was accepted until codex broke it with
    `validate(codes, k_store_perm)`, and an unused `_ = codes[..., perm]` passed
    until codex broke that too. An exotic reordering such as
    `torch.index_select` must therefore carry the logical marker: a deliberate
    false negative, in exchange for no false positives in a rule whose whole
    purpose is catching a silent wrong answer. A raw unpack whose result is never bound to a name cannot be
    reordered at all, so it fails.
    """
    names = _unpacked_names(fn, local_names)

    def is_unpack_call(node) -> bool:
        return (isinstance(node, ast.Call)
                and ((isinstance(node.func, ast.Name)
                      and node.func.id in local_names)
                     or (isinstance(node.func, ast.Attribute)
                         and node.func.attr in local_names)))

    def is_codes(node) -> bool:
        # The unpacked codes: a name bound to them, or the call result itself.
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id in names:
                return True
            if is_unpack_call(n):
                return True
        return False

    # ONLY a subscript counts. A call taking both the codes and a perm was
    # accepted until codex broke it with `validate(codes, k_store_perm)`, which
    # co-occurs and reorders nothing. Indexing is the one form that cannot be
    # faked, so an exotic reordering such as `torch.index_select` must carry the
    # logical marker or be rewritten as a subscript. That is a deliberate
    # false-negative in exchange for no false positives, because this rule exists
    # to catch a silent wrong answer.
    # THE RESULT MUST FLOW ONWARD, which is the last hole codex found: an unused
    # `_ = codes[..., perm]` reorders nothing and passed. So the subscript has to
    # REBIND the codes or be returned. Rebinding covers every form in this tree
    # (`qk = qk[..., inv]`, `codes = codes[..., inv]`, and
    # `codes = unpack(...)[..., inv]`, whose target is itself the unpacked name).
    for node in ast.walk(fn):
        subs = []
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = ([node.target] if isinstance(node, ast.AnnAssign)
                       else node.targets)
            rebinds = any(isinstance(x, ast.Name) and x.id in names
                          for tgt in targets for x in ast.walk(tgt))
            if rebinds and node.value is not None:
                subs = [n for n in ast.walk(node.value)
                        if isinstance(n, ast.Subscript)]
        elif isinstance(node, ast.Return) and node.value is not None:
            subs = [n for n in ast.walk(node.value)
                    if isinstance(n, ast.Subscript)]
        for s in subs:
            if is_codes(s.value) and _perm_mentioned(s.slice):
                return True
    return False


@pytest.mark.parametrize("unpacker", RAW_UNPACKERS)
def test_every_raw_unpack_scope_accounts_for_the_k_order(unpacker):
    """A raw unpack must sit in a scope that knows about the stored order.

    This is the gate that would have caught the one live hole:
    `test_fold_quant_flex_sm100.ref_channel_k_flex` unpacks K by hand and never
    inverted sigma, which was the sole reason channel-wise K could not carry the
    store permutation until 2026-08-13.
    """
    bad = []
    for rel, text in _sources():
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        markers = _marker_lines(text)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Follow `import X as Y` here too. Aliasing is the failure this
            # whole file was rewritten for, so the raw rule must not be the one
            # place that ignores it.
            local_names = {k for k, v in _aliases(tree, (unpacker,)).items()
                           if v == unpacker}
            hits = [n for n in ast.walk(fn)
                    if isinstance(n, ast.Call)
                    and ((isinstance(n.func, ast.Name)
                          and n.func.id in local_names)
                         or (isinstance(n.func, ast.Attribute)
                             and n.func.attr in local_names))]
            if not hits or _reorders_the_unpacked_codes(fn, local_names):
                continue
            for h in hits:
                if any(h.lineno - MARKER_REACH <= m <= h.lineno for m in markers):
                    continue
                bad.append(f"{rel}:{h.lineno}: {fn.name}() unpacks K raw")
    assert not bad, (
        f"{len(bad)} raw {unpacker} call(s) sit in a scope that neither takes "
        "k_store_perm nor applies _k_store_perm_inverse, so they assume stored "
        "order IS logical order. That is a false green the moment the pool is "
        f"permuted. Thread the order, or write `# {MARKER} <reason>`:\n  "
        + "\n  ".join(bad))


@pytest.mark.parametrize("unpacker", UNPACKERS)
def test_every_pool_unpack_inverts_the_k_order(unpacker):
    bad = _unstated((unpacker,))
    assert not bad, (
        f"{len(bad)} {unpacker} call site(s) do not invert the K store order. A "
        "reference that ignores it agrees with a kernel reading a permuted pool "
        "as logical order, which is a FALSE GREEN rather than a failure. Pass "
        f"k_store_perm, or write `# {MARKER} <reason>`:\n  " + "\n  ".join(bad))


def test_the_scan_sees_the_call_sites_it_is_meant_to():
    """A scan that matched nothing would report success forever.

    The floor is per family and counts PARSED calls, so a docstring that happens
    to name a builder cannot pad it. Raise these numbers when call sites are
    added deliberately; a fall means the scan broke, not that the code got safer.
    """
    found = {n: sum(1 for _rel, text in _sources()
                    for _c in _calls(text, (n,)))
             for n in BUILDERS + UNPACKERS}
    # Measured 2026-08-12, exact rather than approximate: a floor set below the
    # real count is slack the next omission hides in. `make_pool_problem_flex`
    # went 7 -> 6 when `scripts/probes/probe_k_perm.py` was deleted with the
    # kwarg its arms needed, and this floor is what noticed, which is the
    # behaviour wanted: an inventory change should require a deliberate edit here
    # rather than pass silently. `make_pool_problem` went 8 -> 6 on 2026-09-14,
    # when `scripts/_g3_dump_check.py` (two call sites) was deleted with
    # `dump_stage`.
    floors = {"make_pool_problem": 6, "make_pool_problem_flex": 6,
              "dequant_pools_flex": 9, "ref_paged_attention_fp8q_flex": 1}
    short = {n: (found[n], f) for n, f in floors.items() if found[n] < f}
    assert not short, f"the scan lost call sites (found, floor): {short}"


def _tracked_for_indirection():
    # RAW_UNPACKERS included: `u = unpack_rows_lut; u(...)` bypassed raw-call
    # detection entirely while the prohibition covered only the other two
    # families, so the one rule that most needs it was the one that lacked it.
    # The SM90 pool helpers too: `pack = repack_K_rs; pack(...)` hides a call from the SM90
    # order gate, and `convert = code_rows_to_order; convert(rows, "chunk16")` from the
    # literal gate (codex, 2026-09-25).
    return set(BUILDERS + UNPACKERS + RAW_UNPACKERS + SM90_POOL_HELPERS)


def _indirect_uses(text: str) -> list[str]:
    """`lineno: what` for every use of a tracked function other than a call or an import."""
    tracked = _tracked_for_indirection()
    bad = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return bad
    # No exemption for the DEFINING module: `pack_k = repack_K_rs` there re-exports the
    # function under a name no gate knows (codex, 2026-09-25). A `def` is not a Name node,
    # so defining a tracked function is never flagged.
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)):
            called.add(id(node.func))
    # A tracked function under an `import X as Y` name is still that function, and
    # `mod.X` is too: codex rebound both (`writer = pk`, `writer = r.repack_K_rs`) past a
    # rule that knew only the canonical bare name (2026-09-25).
    local = {k for k, v in _aliases(tree, tuple(tracked)).items() if v in tracked}
    # An SM90 helper may not be imported under another name at all. The scan resolves an
    # alias within one file, but a module that aliases one re-exports it under the alias,
    # and a second module importing that alias is invisible to every gate here (codex's
    # `from pool_helpers import pack_k`, 2026-09-25). No file needs the alias.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if a.name in SM90_POOL_HELPERS and a.asname and a.asname != a.name:
                    bad.append(f"{node.lineno}: imports {a.name} as {a.asname}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in tracked:
            if id(node) not in called:
                bad.append(f"{node.lineno}: .{node.attr} used as a value")
            continue
        if not isinstance(node, ast.Name) or node.id not in local:
            continue
        if id(node) in called:
            continue
        if isinstance(node.ctx, ast.Store):
            bad.append(f"{node.lineno}: rebinds {node.id}")
            continue
        # A bare load that is not the callee of a call: passed as a value.
        bad.append(f"{node.lineno}: {node.id} used as a value")
    return bad


def test_the_builders_are_never_reached_indirectly():
    """A tracked builder is imported, defined, or called, and never passed around.

    The scan resolves `import X as Y`, and it cannot resolve
    `builder = make_pool_problem; builder(...)`, nor a `functools.partial`, nor a
    dispatch dict, nor a re-export. Chasing those through a static analysis would
    be a losing race, so the indirection is FORBIDDEN instead: this fails if a
    tracked name is ever used as a value rather than called. That converts an
    invisible hole into a loud test failure at the moment someone introduces it,
    which is the only version of this property a reader can trust.
    """
    bad = [f"{rel}:{use}" for rel, text in _sources() for use in _indirect_uses(text)]
    # An `import X` statement is not a Name node, so imports never appear here.
    assert not bad, (
        "a pool builder or unpacker is reached indirectly, which makes the "
        "call-site gates above blind to it. Call it directly, or teach the scan "
        "to follow this form and delete this test:\n  " + "\n  ".join(bad))


def test_an_aliased_import_is_still_seen():
    """The hole this file was rewritten to close.

    `scripts/_e2e_fold_check.py` imports the builder under another name. It
    states its K order today, so the gate cannot prove itself on that file;
    this asserts the SCAN resolves the alias, which is what makes that statement
    load-bearing rather than lucky.
    """
    src = (ROOT / "scripts/_e2e_fold_check.py").read_text(encoding="utf-8")
    assert "make_pool_problem as make_problem" in src, (
        "the aliasing import this test pins is gone; re-point or delete it")
    seen = list(_calls(src, ("make_pool_problem",)))
    assert seen, "the alias resolver stopped seeing the aliased call"
    assert all(stated for _n, _l, stated in seen), (
        "the aliased call site stopped stating k_store_perm")


@pytest.mark.parametrize("case,src,reorders", [
    ('unused parameter',
     'def f(mKq, bits, D, k_store_perm=None):\n    c = unpack_rows_lut(mKq, bits, D)\n    return c\n', False),
    ('docstring mention only',
     'def f(mKq, bits, D):\n    """k_store_perm is not inverted here"""\n    c = unpack_rows_lut(mKq, bits, D)\n    return c\n', False),
    ('perm passed somewhere else',
     'def f(mKq, bits, D, k_store_perm=None):\n    c = unpack_rows_lut(mKq, bits, D)\n    validate(k_store_perm)\n    return c\n', False),
    ('keyword whose value is a constant',
     'def f(mKq, bits, D):\n    c = unpack_rows_lut(mKq, bits, D)\n    g(k_store_perm=None)\n    return c\n', False),
    ('unpack returned directly',
     'def f(mKq, bits, D):\n    return unpack_rows_lut(mKq, bits, D)\n', False),
    ('subscript on the unpacked codes',
     'def f(mKq, bits, D, k_store_perm=None):\n    c = unpack_rows_lut(mKq, bits, D)\n    return c[..., _k_store_perm_inverse(k_store_perm).long()]\n', True),
    ('codex loophole: a call taking both, reordering nothing',
     'def f(mKq, bits, D, k_store_perm=None):\n    codes = unpack_rows_lut(mKq, bits, D)\n    validate(codes, k_store_perm)\n    return codes\n', False),
    ('codex: a subscript whose result is never used',
     'def f(mKq, bits, D, k_store_perm=None):\n    codes = unpack_rows_lut(mKq, bits, D)\n    _ = codes[..., k_store_perm]\n    return codes\n', False),
    ('returned directly rather than rebound',
     'def f(mKq, bits, D, kperm=None):\n    c = unpack_rows_lut(mKq, bits, D)\n    return c[..., kperm]\n', True),
    ('codex false negative: subscript applied inline to the call result',
     'def f(mKq, bits, D, k_store_perm=None):\n    codes = unpack_rows_lut(mKq, bits, D)[..., _k_store_perm_inverse(k_store_perm)]\n    return codes\n', True),
])
def test_the_rule_demands_the_codes_be_reordered(case, src, reorders):
    """Three versions of this rule have been defeated, each more subtly.

    Version one accepted a function that merely NAMED a perm, so an unused
    parameter passed. Version two accepted a perm appearing in any call, and codex
    broke it by EXECUTING `validate(k_store_perm)` and `g(k_store_perm=None)`,
    neither of which touches the codes. Both are pinned below as negatives,
    because the property is not that a perm is nearby: it is that the UNPACKED
    CODES are indexed by one.
    """
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef))
    assert _reorders_the_unpacked_codes(fn, {"unpack_rows_lut"}) is reorders, case


def test_the_rule_cannot_see_a_value():
    """A KNOWN LIMITATION, pinned so it cannot be mistaken for coverage.

    `k_store_perm.sort().values` is identity, so this leaves the codes in stored
    order while satisfying the rule. Codex found it on the fourth attack. It is not
    fixable syntactically: whether an index expression is a permutation or the
    identity is a property of its VALUE, and this file never evaluates anything.

    It is also not a realistic omission. The failure this gate exists for is a
    scope that forgets to reorder, which has happened eight times in this feature's
    history; sorting a permutation into identity is something one has to intend. An
    ordering that is wrong in value is caught by the numerics gates instead, at
    rel_l2 near 1.0 against an independent reference.

    If this assertion ever starts FAILING, the rule got stronger and this test
    should be deleted rather than adjusted.
    """
    src = ("def f(mKq, bits, D, k_store_perm=None):\n"
           "    codes = unpack_rows_lut(mKq, bits, D)\n"
           "    codes = codes[..., k_store_perm.sort().values]\n"
           "    return codes\n")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef))
    assert _reorders_the_unpacked_codes(fn, {"unpack_rows_lut"}) is True, (
        "the rule now rejects an identity-valued index; delete this test")


#: SM90 packers. Their `order` is the READING kernel's code-pool order
#: (`_sm90/serving_cache.py`, CODE POOL ORDERS): IntPingPongDecodeSm90 reads chunk16, and
#: the row order holds the same bytes per block, so a pool in the wrong order reads wrong
#: at int4 and raises nothing if its shape was also reshaped.
#: `order` is keyword-only with no default, so an omission is a TypeError when the call
#: runs; most callers only run on an H100, and this finds the omission on a CPU.
SM90_PACKERS = ("repack_K_rs", "repack_V_rs")

#: Where the string "chunk16" may appear as a value: the module that defines the orders,
#: the kernel that declares which one it reads, and the packer's own host gate, which
#: tests the order itself rather than writing a pool for a reader. Every other writer asks
#: `IntPingPongDecodeSm90.code_pool_order`, the predicate the kernel exposes.
CHUNK16_LITERAL_OK = ("src/cutedsl_decode/_sm90/serving_cache.py",
                      "src/cutedsl_decode/_sm90/int_decode_pingpong.py",
                      "scripts/probes/repack_rs.py")
#: Every helper that takes an order, because an order passed to it is what the literal gate
#: reads. The order-free geometry (`code_row_bytes`, `code_row_chunks`, `code_v_half_bytes`)
#: carries no order to hide.
SM90_POOL_HELPERS = SM90_PACKERS + ("code_rows_to_order", "code_block_shape",
                                    "code_pool_shape", "code_block_word_strides",
                                    "code_chunk_offset", "code_word_offset",
                                    "require_code_pool")


@pytest.mark.parametrize("packer", SM90_PACKERS)
def test_every_sm90_pack_states_its_code_pool_order(packer):
    bad = [f"{rel}:{lineno}" for rel, text in _sources()
           for _t, lineno, stated in _calls(text, (packer,), keyword="order")
           if not stated]
    assert not bad, (
        f"{len(bad)} {packer} call site(s) do not pass order=. Pass the reading "
        "kernel's order, IntPingPongDecodeSm90.code_pool_order, or \"row\" for the "
        "thread-row layout itself:\n  " + "\n  ".join(bad))


def _chunk16_literals(text: str):
    """Line of every string constant that is exactly "chunk16".

    Every constant, not only a pool-helper argument: `ORDER = "chunk16"` followed by
    `repack_K_rs(..., order=ORDER)` states the order through a name, and so does a dict
    value or a default, which a rule that looked only at call arguments let through
    (review, 2026-09-25). A docstring or error message that mentions the order is a longer
    string and does not match."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "chunk16":
            yield node.lineno


def test_no_sm90_pool_writer_restates_the_chunk16_order():
    """A writer asks the kernel's predicate; it does not spell its answer.

    A literal "chunk16" keeps working while the predicate agrees, and stops being the
    reader's order the day the predicate changes."""
    bad = [f"{rel}:{lineno}" for rel, text in _sources()
           if rel not in CHUNK16_LITERAL_OK and not rel.startswith("tests/")
           for lineno in _chunk16_literals(text)]
    assert not bad, (
        "the chunk16 order is spelled outside the modules that define and declare it; "
        "ask IntPingPongDecodeSm90.code_pool_order instead:\n  " + "\n  ".join(bad))


def test_the_sm90_scan_sees_the_packer_calls():
    """Floors measured 2026-09-25: a fall means the scan broke, not that code got safer."""
    found = {n: sum(1 for _rel, text in _sources()
                    for _c in _calls(text, (n,), keyword="order"))
             for n in SM90_PACKERS}
    floors = {"repack_K_rs": 12, "repack_V_rs": 10}
    short = {n: (found[n], f) for n, f in floors.items() if found[n] < f}
    assert not short, f"the scan lost call sites (found, floor): {short}"


@pytest.mark.parametrize("src", [
    # codex's aliased import (2026-09-25); the alias is also refused by `_indirect_uses`
    "from repack_rs import repack_K_rs as pack_k\nkq = pack_k(q, m, 4, order=\"chunk16\")\n",
    # the order held in a name (review, 2026-09-25)
    "from repack_rs import repack_K_rs\nORDER = \"chunk16\"\n"
    "kq = repack_K_rs(q, m, 4, order=ORDER)\n",
    # and in a default
    "def build(q, m, order=\"chunk16\"):\n    return repack_K_rs(q, m, 4, order=order)\n",
])
def test_the_chunk16_literal_is_seen_however_it_reaches_the_call(src):
    assert list(_chunk16_literals(src)), src


def test_the_sm90_order_gate_follows_an_aliased_import():
    """`_calls` resolves `import X as Y`, so an aliased packer call is still counted."""
    src = "from repack_rs import repack_K_rs as pack_k\nkq = pack_k(q, m, 4, order=o)\n"
    assert list(_calls(src, SM90_PACKERS, keyword="order")) == [("repack_K_rs", 2, True)]


def test_a_message_that_names_the_order_is_not_a_literal():
    src = "raise ValueError(\"reads the 'chunk16' order\")\nx = f\"{o!r} vs chunk16\"\n"
    assert not list(_chunk16_literals(src))


@pytest.mark.parametrize("src", [
    "from repack_rs import repack_K_rs as pk\nwriter = pk\nwriter(q, m, 4, order=o)\n",
    "import repack_rs as r\nwriter = r.repack_K_rs\nwriter(q, m, 4, order=o)\n",
    "from repack_rs import repack_V_rs\nf = functools.partial(repack_V_rs, order=o)\n",
    "from repack_rs import repack_K_rs as pack_k\n",
    "from x.serving_cache import code_rows_to_order\nconvert = code_rows_to_order\n"
    "pool = convert(rows, \"chunk16\")\n",
    "def repack_K_rs(k, m, bits, *, order):\n    pass\npack_k = repack_K_rs\n",
])
def test_an_sm90_helper_reached_indirectly_is_refused(src):
    """Codex's round-2, -3 and -4 bypasses of the prohibition, pinned (2026-09-25).

    What no syntactic scan can see, and so what this does not claim: a helper reached
    through a computed `getattr`, a dynamically generated import, or runtime data flow."""
    assert _indirect_uses(src), src
