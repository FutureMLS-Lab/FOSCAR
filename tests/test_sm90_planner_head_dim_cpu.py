# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""The SM90 split planner takes the head dimension, and its D=128 decisions are pinned exactly.

`PINGPONG_MODEL` is keyed by head dimension, and `plan_splits` and `plan_pingpong` require
`head_dim` from every caller, so no caller can plan another head dimension as if it were 128.
Every D=128 decision is held as a literal table over B in {1, 2, 8, 32, 64, 128}, KV in
{4096, 8192, 16384, 32768}, both widths, 8 KV heads and 132 SMs.

`scheduler.py` imports nothing but `math`, so it is loaded from its file and needs no DSL.
`pingpong_runtime.py` imports torch at module level, so `plan_pingpong` is checked on its
source, and its composition with the scheduler is what the table's entries are.
"""
import ast
import importlib.util
import pathlib
import runpy

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "src/cutedsl_decode/_sm90/scheduler.py"
RUNTIME = ROOT / "src/cutedsl_decode/_sm90/pingpong_runtime.py"
BATCHES = (1, 2, 8, 32, 64, 128)
KV_LENS = (4096, 8192, 16384, 32768)
KV_HEADS, SMS = 8, 132

#: Each bit width holds one string per batch size. A string gives the four KV lengths in
#: order, each as outer/min_blocks, and appends /F where the fused combine is selected.
DECISIONS_D128 = {
    2: ("8/2/F 16/2/F 16/2/F 16/2/F", "8/2/F 8/2/F 8/2/F 12/2/F", "4/2 4/2 4/2 4/2",
        "1/2 1/2 1/2 1/2", "1/2 1/2 1/2 1/2", "1/2 1/2 1/2 1/2"),
    4: ("8/2/F 16/2/F 16/2/F 16/2/F", "8/2/F 8/2/F 8/2/F 12/2/F", "4/2 4/2 4/2 4/2",
        "1/2 1/2 1/2 1/2", "1/2 1/2 1/2 1/2", "1/2 1/2 1/2 1/2"),
}

#: The D=128 model entry, one cost per term.
PINGPONG_MODEL_D128 = {"T_us": {2: 0.358, 4: 0.420},
                       "P_us": 5.867, "combine_us": 10.315, "K_us": 0.08}

#: The D=256 model entry, one cost per term.
PINGPONG_MODEL_D256 = {"T_us": {2: 0.85, 4: 0.15},
                       "P_us": 0.0, "combine_us": 0.0, "K_us": 0.05}

#: The D=256 decisions of that entry, in the format of DECISIONS_D128, computed with this
#: file's `_cell` over the same shapes. They differ from the one-wave rule's picks (the
#: absent-entry test below) at B=1 from KV=16384 at both widths and at B=2 at INT4.
DECISIONS_D256 = {
    2: ("8/2/F 16/2/F 16/2/F 32/2/F", "8/2/F 16/2/F 16/2/F 16/2/F", "4/2 4/2 4/2 4/2",
        "1/2 1/2 1/2 1/2", "1/2 1/2 1/2 1/2", "1/2 1/2 1/2 1/2"),
    4: ("8/2/F 16/2/F 16/2/F 16/2/F", "8/2/F 8/2/F 8/2/F 12/2/F", "4/2 4/2 4/2 4/2",
        "1/2 1/2 1/2 1/2", "1/2 1/2 1/2 1/2", "1/2 1/2 1/2 1/2"),
}


@pytest.fixture(scope="module")
def sch():
    spec = importlib.util.spec_from_file_location("_sm90_scheduler_under_test", SCHEDULER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell(sch, bits, batch, kv, head_dim):
    outer, min_blocks = sch.plan_splits(batch, KV_HEADS, kv, bits=bits, head_dim=head_dim,
                                        sm_count=SMS)
    text = f"{outer}/{min_blocks}"
    if sch.plan_fused_combine(batch, KV_HEADS, outer):
        text += "/F"
    return text


@pytest.mark.parametrize("bits", sorted(DECISIONS_D128))
def test_every_d128_decision_is_the_one_main_made(sch, bits):
    got = tuple(" ".join(_cell(sch, bits, b, kv, 128) for kv in KV_LENS) for b in BATCHES)
    assert got == DECISIONS_D128[bits]


def test_the_model_has_the_d128_and_the_d256_entries(sch):
    """Both entries are compared exactly. The decision tables below cannot pin P_us, because
    it adds the same cost to every candidate and so moves no pick."""
    assert sorted(sch.PINGPONG_MODEL) == [128, 256]
    assert sch.PINGPONG_MODEL[128] == PINGPONG_MODEL_D128
    assert sch.PINGPONG_MODEL[256] == PINGPONG_MODEL_D256


@pytest.mark.parametrize("bits", sorted(DECISIONS_D256))
def test_every_d256_decision_is_the_landed_entrys(sch, bits):
    got = tuple(" ".join(_cell(sch, bits, b, kv, 256) for kv in KV_LENS) for b in BATCHES)
    assert got == DECISIONS_D256[bits]


def test_plan_splits_requires_the_head_dimension(sch):
    with pytest.raises(TypeError, match="head_dim"):
        sch.plan_splits(1, KV_HEADS, 8192, bits=2, sm_count=SMS)


def _function(path, name):
    return next(node for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.FunctionDef) and node.name == name)


def test_plan_pingpong_requires_the_head_dimension_and_forwards_it():
    fn = _function(RUNTIME, "plan_pingpong")
    kwonly = [a.arg for a in fn.args.kwonlyargs]
    defaults = dict(zip(kwonly, fn.args.kw_defaults))
    assert "head_dim" in kwonly and defaults["head_dim"] is None, (
        "plan_pingpong must take head_dim as a keyword-only argument without a default")
    (call,) = [node for node in ast.walk(fn) if isinstance(node, ast.Call)
               and ast.unparse(node.func) == "plan_splits"]
    passed = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert passed["head_dim"] == "head_dim"


@pytest.mark.parametrize("bits", [2, 4])
def test_a_head_dimension_without_an_entry_plans_by_the_legacy_rule(sch, bits, monkeypatch):
    """With the 256 entry removed, the planner returns the one-wave rule's pick: on 132 SMs
    at two CTAs per SM that is max(1, min(264 // (8B), tiles, KV // 512)), which is one split
    from B=32 up, and 32 or 33 splits at B=1."""
    monkeypatch.delitem(sch.PINGPONG_MODEL, 256)
    legacy = {1: (8, 16, 32, 33), 2: (8, 16, 16, 16), 8: (4, 4, 4, 4),
              32: (1, 1, 1, 1), 64: (1, 1, 1, 1), 128: (1, 1, 1, 1)}
    for b in BATCHES:
        got = tuple(sch.plan_splits(b, KV_HEADS, kv, bits=bits, head_dim=256, sm_count=SMS)
                    for kv in KV_LENS)
        assert got == tuple((o, 2) for o in legacy[b]), (b, got)


def test_the_scheduler_self_test_passes(capsys):
    runpy.run_path(str(SCHEDULER), run_name="__main__")
    assert "scheduler self-test: 11 cases OK" in capsys.readouterr().out


