#!/usr/bin/env python3
# Copyright (c) 2026 Donglin Zhuang
# SPDX-License-Identifier: MIT
"""CPU gates for the constructor contract the two decode kernels share.

`kernel_contract.py` exists because the same argument name would otherwise mean different
things across the two kernels. `sdtype` is the case that bites: SM100 asserts
`sdtype in ("f32", "bf16")` and SM90 asserts `sdtype in (torch.float32,
torch.bfloat16)`, the same rule in two vocabularies, so a caller otherwise has to know
which kernel it is talking to before it can name a scale dtype. `page_size == 64`
and `bits in (2, 4)` are each asserted twice with different messages, which is the
same defect one step quieter: two copies free to drift, which is why
`quant_block_offsets` lives in one file.

These gates need no GPU, because the contract is host-side by construction: nothing here
is traced and nothing here touches kernel state.

Run: pytest tests/test_kernel_contract_cpu.py
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _contract():
    """Load the module BY PATH. The package `__init__` imports torch and cutlass, and
    this file's whole claim is that the contract needs neither."""
    spec = importlib.util.spec_from_file_location(
        "kernel_contract", _SRC / "foscar" / "kernel_contract.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_module_is_torch_free():
    """Loaded with no torch import of its own, like `quant_layout.py`."""
    src = (_SRC / "foscar" / "kernel_contract.py").read_text()
    assert "import torch" not in src, (
        "the contract is imported by both kernel modules and must stay torch-free")


@pytest.mark.parametrize("spelling,want", [
    ("f32", "f32"),
    ("bf16", "bf16"),
])
def test_string_spellings_pass_through(spelling, want):
    assert _contract().normalize_sdtype(spelling) == want


def test_torch_spellings_are_accepted():
    """The torch spellings normalize alongside the string ones."""
    torch = pytest.importorskip("torch")
    n = _contract().normalize_sdtype
    assert n(torch.float32) == "f32"
    assert n(torch.bfloat16) == "bf16"


@pytest.mark.parametrize("bad", ["f16", "float32", 32, None, ()])
def test_unknown_sdtype_is_refused_by_name(bad):
    """The message names the value it got, not only that the value was refused."""
    with pytest.raises(AssertionError) as e:
        _contract().normalize_sdtype(bad)
    assert repr(bad) in str(e.value)


def test_sdtype_is_not_folded_into_the_shape_check():
    """SM100 passes `sdtype=None` meaning "resolve it from the K axis", and that
    resolution needs quant fields the shape check runs before. Folding the two together
    would reject None on 77 of 115 anchor cells."""
    import inspect
    sig = inspect.signature(_contract().GQAPagedDecode._validate_shape)
    assert "sdtype" not in sig.parameters, (
        "the shape check must not demand an sdtype; each kernel normalizes its own "
        f"where it has one. Got {list(sig.parameters)}")


def test_the_shared_shape_rules_hold():
    _contract().validate_problem(page_size=64, head_dim=64, q_head_tile=8, bits=2)
    _contract().validate_problem(page_size=64, head_dim=128, q_head_tile=8, bits=2)
    _contract().validate_problem(page_size=64, head_dim=256, q_head_tile=1, bits=4)


@pytest.mark.parametrize("kw,needle", [
    (dict(page_size=32, head_dim=128, q_head_tile=8, bits=2), "tile KV at 64"),
    (dict(page_size=64, head_dim=100, q_head_tile=8, bits=2), "multiple of 64"),
    (dict(page_size=64, head_dim=32, q_head_tile=8, bits=2), "multiple of 64"),
    (dict(page_size=64, head_dim=128, q_head_tile=3, bits=2), "q_head_tile must be"),
    (dict(page_size=64, head_dim=128, q_head_tile=8, bits=8), "must be 2 or 4"),
])
def test_each_shared_rule_refuses_its_own_violation(kw, needle):
    with pytest.raises(AssertionError) as e:
        _contract().validate_problem(**kw)
    assert needle in str(e.value)


def test_the_gqa_bound_is_not_claimed_as_shared():
    """Both kernels tile high GQA; Hopper requires complete eight-head tiles.

    The shared shape contract has no logical GQA bound. Execute the production
    host admission helper directly, extracted by AST so this contract test keeps
    requiring neither torch nor the DSL.
    """
    import ast
    import inspect
    assert "heads_per_kv" not in inspect.signature(_contract().validate_problem).parameters
    path = _SRC / "foscar" / "wrapper.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "_check_num_qo_heads")
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    admit = scope["_check_num_qo_heads"]
    for arch in ("sm90", "sm100"):
        for ratio in (*range(1, 9), 16, 32, 64, 128):
            admit(arch, 8 * ratio, 8)
        for bad in (0, -8, True, 8.5, 65):
            with pytest.raises(ValueError):
                admit(arch, bad, 8)
    for ratio in (9, 10, 17, 31, 129):
        admit("sm100", 8 * ratio, 8)
        with pytest.raises(NotImplementedError, match="multiples of eight"):
            admit("sm90", 8 * ratio, 8)


@pytest.mark.parametrize("spoof", ["torch.float32", "torch.bfloat16"])
def test_a_string_that_merely_looks_like_a_torch_dtype_is_refused(spoof):
    """The type is checked before `str(x)`, so neither the literal string nor an
    object that spells itself as a torch dtype is accepted."""
    with pytest.raises(AssertionError):
        _contract().normalize_sdtype(spoof)


def test_an_object_impersonating_a_torch_dtype_is_refused():
    class Spoof:
        def __str__(self):
            return "torch.float32"
    with pytest.raises(AssertionError):
        _contract().normalize_sdtype(Spoof())


def test_both_kernels_declare_the_contract_as_their_base():
    """Read from the SOURCE, so this holds without importing cutlass.

    A kernel that stops inheriting it keeps compiling and silently stops sharing the
    vocabulary, which this gate refuses.
    """
    import ast
    for rel, cls in (("_sm100/int_fold_decode.py", "IntDecodeSm100"),
                     ("_sm90/int_decode_pingpong.py", "IntPingPongDecodeSm90")):
        tree = ast.parse((_SRC / "foscar" / rel).read_text())
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.ClassDef) and n.name == cls)
        bases = [ast.unparse(b) for b in node.bases]
        assert "GQAPagedDecode" in bases, (
            f"{cls} must inherit GQAPagedDecode; bases are {bases}")


def _init_of(rel, cls):
    """The `__init__` FunctionDef of a kernel class, parsed from source."""
    import ast
    tree = ast.parse((_SRC / "foscar" / rel).read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == cls)
    return ast.parse((_SRC / "foscar" / rel).read_text()), next(
        m for m in node.body
        if isinstance(m, ast.FunctionDef) and m.name == "__init__")


@pytest.mark.parametrize("rel,cls", [
    ("_sm100/int_fold_decode.py", "IntDecodeSm100"),
    ("_sm90/int_decode_pingpong.py", "IntPingPongDecodeSm90"),
])
def test_each_constructor_actually_calls_the_shared_check(rel, cls):
    """Inheriting the base proves nothing if nobody calls into it.

    This is an AST check rather than a substring one, so reformatting cannot make it
    pass. It still does not prove the call SUCCEEDS -- only constructing the kernel does
    that, which needs cutlass and a GPU, and the hardware gate is where it happens.
    """
    import ast
    _tree, init = _init_of(rel, cls)
    calls = [n for n in ast.walk(init)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_validate_shape"
             and getattr(n.func.value, "id", None) == "self"]
    assert calls, f"{cls}.__init__ must call self._validate_shape(...)"
    # AND IT MUST PASS THE CONSTRUCTOR'S OWN ARGUMENTS. Checking only that the call
    # exists would accept `self._validate_shape(page_size, head_dim, q_head_tile, 4)`,
    # which validates a literal and lets the kernels drift on `bits`.
    want = ["page_size", "head_dim", "q_head_tile", "bits"]
    passed = [ast.unparse(a) for a in calls[0].args]
    assert passed == want, (
        f"{cls}.__init__ must pass its own {want} to _validate_shape; got {passed}")


@pytest.mark.parametrize("rel,cls", [
    ("_sm100/int_fold_decode.py", "IntDecodeSm100"),
    ("_sm90/int_decode_pingpong.py", "IntPingPongDecodeSm90"),
])
def test_every_name_a_constructor_reads_is_bound_before_it(rel, cls):
    """The CRITICAL defect a substring check misses.

    A function-local import makes the name local for the whole function, so a read
    above that import raises UnboundLocalError on every construction. A substring test
    sees the call and the import and is satisfied. This walks the constructor in order
    instead: a name that is imported locally must not be read before that import runs.
    """
    import ast
    _tree, init = _init_of(rel, cls)
    local_imports = {}
    # TOP-LEVEL statements of the function body only. An import nested inside an `if`
    # or a loop may not run at all, so a line-number comparison against it proves
    # nothing -- `if False: from x import y` would satisfy it.
    for st in init.body:
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            for a in st.names:
                local_imports.setdefault(a.asname or a.name.split(".")[0], st.lineno)
    conditional = [n for n in ast.walk(init)
                   if isinstance(n, (ast.Import, ast.ImportFrom))
                   and n not in init.body]
    for n in conditional:
        for a in n.names:
            nm = a.asname or a.name.split(".")[0]
            assert nm not in local_imports, (
                f"{cls}.__init__ imports {nm!r} both at top level and conditionally")
            local_imports[nm] = 10 ** 9   # never satisfiable: a read cannot precede it
    for n in ast.walk(init):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) \
                and n.id in local_imports:
            assert n.lineno >= local_imports[n.id], (
                f"{cls}.__init__ reads {n.id!r} at line {n.lineno}, above its own "
                f"local import at line {local_imports[n.id]}; that is an "
                f"UnboundLocalError on every construction")


def test_the_sdtype_validity_rule_is_not_claimed_as_shared():
    """The contract unifies the SPELLING of sdtype, not which one a configuration may
    use.

    SM100 allows bf16 only for K token-(D/2); SM90 also permits it for V token-64,
    which SM100 does not implement at all. Each kernel therefore keeps its own
    validity rule, and normalisation must stay spelling-only so that it cannot
    become a shared gate by accident.
    """
    src = (_SRC / "foscar" / "kernel_contract.py").read_text()
    assert "DELIBERATELY NOT UNIFIED" in src, (
        "the contract must say that sdtype VALIDITY is per-kernel, not shared")
    for rel, needle in (("_sm100/int_fold_decode.py", 'assert sdtype == "f32"'),
                        ("_sm90/int_decode_pingpong.py", 'assert sdtype_name == "bf16"')):
        got = (_SRC / "foscar" / rel).read_text()
        assert needle in got, f"{rel} must keep its own sdtype rule: {needle}"


def test_neither_kernel_still_asserts_the_shared_rules_itself():
    """The shared rules live in one place, and the kernels must not restate them.

    Each of these belongs in the contract only. A kernel is free to narrow a rule --
    SM90 requires `q_head_tile == 8` and SM100 requires `q_head_tile >= 4` in real
    mode -- but not to restate the shared one.
    """
    banned = {
        "_sm100/int_fold_decode.py": [
            'assert page_size == 64\n',
            'assert bits in (2, 4)\n',
            'assert sdtype in ("f32", "bf16")',
        ],
        "_sm90/int_decode_pingpong.py": [
            'assert page_size == 64,',
            'assert bits in (4, 2),',
            'assert sdtype in (torch.float32, torch.bfloat16)',
        ],
    }
    for rel, needles in banned.items():
        src = (_SRC / "foscar" / rel).read_text()
        for n in needles:
            assert n not in src, f"{rel} still carries the shared rule: {n.strip()}"
