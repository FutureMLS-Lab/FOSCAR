# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""No function uses a name that nothing in scope binds.

WHY THIS EXISTS. A name can be imported in one method and used in another, so the
second raises `NameError` on every call while the first looks correct. A helper
imported into the wrong function takes down the whole column that calls it, and it
costs a GPU run to find because nothing fails until the line executes.

The shape is specific to this repository and that is why a general linter has not
been enough: modules here deliberately import the heavy dependencies
(`_fold_harness`, `repack_sm100`, anything reaching cutlass) INSIDE the functions
that need them, so that a CPU box can still import the file. Function-local
imports are therefore normal here, and a name imported in one function is exactly
the kind of thing that looks bound when it is not.

`python -m py_compile` cannot see this: an unbound load is legal Python until it
runs. Nothing else in the suite covers it, because the failing paths need a GPU.

THE SCOPE MODEL IS LEXICAL, AND IT HAS TO BE. Collecting bindings with `ast.walk`
descends through every nested scope and opens two holes: a load satisfied only by
a CLASS ATTRIBUTE is treated as bound, though a function body cannot see a class
namespace, and so is a load satisfied only by a SIBLING nested function's import.
Both are `NameError` at runtime. Bindings are therefore collected per scope,
stopping at each nested function, lambda and class, and a scope's visible set is
builtins, the module scope, its enclosing FUNCTION scopes, and its own.

WHAT IT DOES NOT CATCH, so nobody reads more into a pass. It resolves names, not
values, and it is conservative on purpose: any name bound anywhere in the same
scope counts, so a use before its assignment passes, and a conditional import
inside `try` counts as bound. A file containing `import *` is skipped entirely,
because what that binds is not knowable from the source. Attribute access is not
checked. It exists to catch "imported over there, used over here".
"""
from __future__ import annotations

import ast
import builtins
import pathlib
import subprocess

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Vendored third-party sources. Not ours to fix, and a finding there would train
#: a reader to ignore this gate. `_vendor/flash_attn/cute/utils.py` legitimately
#: trips it: `ex2_emulation_2` reads `quack` under an import upstream makes
#: conditional.
VENDORED = ("src/cutedsl_decode/_vendor/",)

#: Anything that opens a new namespace. Bindings do not cross these.
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)

#: Scopes whose bindings a NESTED function can read. A class body is not one:
#: a function defined inside a class cannot see the class's own names without
#: `self` or the class name, which is why a `ClassDef` passes its enclosing set
#: through rather than adding its own.
CLOSURE_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)

_MODULE_DUNDERS = {"__file__", "__name__", "__doc__", "__spec__", "__package__",
                   "__loader__", "__builtins__", "__debug__", "__class__"}


def _tracked_python():
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.py"],
                         capture_output=True, text=True, check=True)
    for rel in out.stdout.split("\n"):
        rel = rel.strip()
        if rel.startswith(VENDORED):
            continue
        if rel and (ROOT / rel).is_file():
            yield rel


def _own_nodes(scope: ast.AST):
    """Every node lexically in `scope` but not inside a nested scope's BODY.

    A nested scope's header still belongs here, because it is evaluated where it
    is written: a `def` binds its name, and decorators, argument defaults and base
    classes all run in the enclosing scope. A header part that is ITSELF a scope,
    such as a lambda used as an argument default, is yielded and not entered:
    entering it would charge the enclosing function for the lambda's own
    parameters. The visitor below picks that lambda up as a scope of its own, and
    the "lambda as an argument default" case below holds it there.
    """
    for child in ast.iter_child_nodes(scope):
        yield child
        if not isinstance(child, SCOPES):
            yield from _own_nodes(child)
            continue
        args = getattr(child, "args", None)
        header = list(getattr(child, "decorator_list", []))
        header += list(getattr(child, "bases", []))
        if isinstance(args, ast.arguments):
            header += list(args.defaults)
            header += [d for d in args.kw_defaults if d is not None]
        for part in header:
            yield part
            if not isinstance(part, SCOPES):
                yield from _own_nodes(part)


def _direct_bindings(scope: ast.AST) -> set[str]:
    """Names `scope` binds itself, ignoring anything a nested scope binds."""
    bound: set[str] = set()
    args = getattr(scope, "args", None)
    if isinstance(args, ast.arguments):
        for a in (args.posonlyargs + args.args + args.kwonlyargs
                  + ([args.vararg] if args.vararg else [])
                  + ([args.kwarg] if args.kwarg else [])):
            bound.add(a.arg)
    for node in _own_nodes(scope):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, SCOPES):
            if not isinstance(node, ast.Lambda):
                bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound |= set(node.names)
    # Comprehension targets live in their own scope in Python 3, but treating
    # them as local is the conservative direction: it can only hide a finding,
    # never invent one.
    for node in ast.walk(scope):
        if isinstance(node, ast.comprehension):
            for t in ast.walk(node.target):
                if isinstance(t, ast.Name):
                    bound.add(t.id)
    return bound


def _unbound_in_file(text: str):
    """[(lineno, scope_name, name)] for every load nothing in scope binds."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    if any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)
           for n in ast.walk(tree)):
        # `from x import *` binds an unknowable set, so a finding here would be a
        # guess. Skipping the file is the honest answer.
        return []
    builtin_names = set(dir(builtins)) | _MODULE_DUNDERS
    out = []

    def visit(scope: ast.AST, enclosing: set[str]) -> None:
        own = _direct_bindings(scope)
        visible = enclosing | own
        for node in _own_nodes(scope):
            if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                    and node.id not in visible
                    and node.id not in builtin_names):
                out.append((node.lineno, getattr(scope, "name", "<lambda>"),
                            node.id))
        for child in _own_nodes(scope):
            if not isinstance(child, SCOPES):
                continue
            # A class body sees the enclosing scope, and what it binds is NOT
            # passed down to functions inside it, so the enclosing set travels
            # unchanged through a ClassDef.
            visit(child, visible if isinstance(scope, CLOSURE_SCOPES)
                  or not isinstance(scope, ast.ClassDef) else enclosing)

    module_bound = _direct_bindings(tree)
    for node in _own_nodes(tree):
        if isinstance(node, SCOPES):
            visit(node, module_bound)
    return out


def test_no_function_loads_a_name_nothing_binds():
    findings = []
    for rel in _tracked_python():
        text = (ROOT / rel).read_text(encoding="utf-8")
        for lineno, scope, name in _unbound_in_file(text):
            findings.append(f"{rel}:{lineno}: {scope}() uses unbound {name!r}")
    assert not findings, (
        f"{len(findings)} name(s) are used where nothing in scope binds them. "
        "Each is a NameError the moment that line runs, and in this repository "
        "that usually means a module was imported inside a DIFFERENT function:\n  "
        + "\n  ".join(findings))


@pytest.mark.parametrize("case,snippet,expect_hit", [
    # The shape this file exists for: imported in one function, used in another.
    ("siblings at module level",
     "def a():\n    from m import f\n\ndef b():\n    return f()\n", True),
    # The same one level down: a sibling's import must not count as bound.
    ("siblings nested",
     "def outer():\n    def a():\n        from m import g\n"
     "    def b():\n        return g()\n    return b\n", True),
    # A class attribute is not visible to a function body.
    ("class attribute",
     "class C:\n    hidden = 1\ndef f():\n    return hidden\n", True),
    # Nor to a method of that same class.
    ("method reading a class attribute",
     "class C:\n    hidden = 1\n    def m(self):\n        return hidden\n", True),
    # The correct forms.
    ("local import", "def a():\n    from m import f\n    return f()\n", False),
    ("module import", "from m import f\n\ndef b():\n    return f()\n", False),
    ("closure over a function scope",
     "def a():\n    from m import f\n    def b():\n        return f()\n"
     "    return b\n", False),
    # A star import makes the module's bindings unknowable, so the file is
    # skipped, and `sqrt` must not be reported.
    ("star import", "from math import *\ndef f():\n    return sqrt(4)\n", False),
    ("lambda arg", "def a():\n    return lambda x: x + 1\n", False),
    ("lambda as an argument default",
     "def outer():\n    def a(f=lambda x, y: (x, y)):\n        return f\n"
     "    return a\n", False),
    ("comprehension", "def a(xs):\n    return [y for y in xs]\n", False),
    ("builtin", "def a(xs):\n    return len(xs)\n", False),
    ("decorator", "from m import d\n@d\ndef a():\n    return 1\n", False),
    ("method arg", "class C:\n    def m(self):\n        return self.x\n", False),
    ("class name from a method",
     "class C:\n    def m(self):\n        return C\n", False),
])
def test_the_scan_agrees_with_python_on_scope(case, snippet, expect_hit):
    """A scan nobody has seen fail is a scan nobody should trust.

    Every case is a lexical-scope question Python already answers: each hit is a
    real `NameError` and each miss is legal code.
    """
    assert bool(_unbound_in_file(snippet)) is expect_hit, case
