# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""The SM90 split planner takes the head dimension, and plans D=128 exactly as before.

`PINGPONG_MODEL` is keyed by head dimension, and `plan_splits` and `plan_pingpong` require
`head_dim` from every caller, so no caller can plan another head dimension as if it were 128.
What must not move is every D=128 decision, so this file holds the decision table over
B in {1, 2, 8, 32, 64, 128}, KV in {4096, 8192, 16384, 32768}, both widths, 8 KV heads and
132 SMs as a literal. The literal was generated from main 0dc8be4, whose `plan_splits` took no
head dimension; the command is in `profile/h100_planb_prb_20261001/PROVENANCE.md`.

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

#: This is the one entry at main 0dc8be4, `src/cutedsl_decode/_sm90/scheduler.py:81-84`.
PINGPONG_MODEL_D128 = {"T_us": {2: 0.358, 4: 0.420},
                       "P_us": 5.867, "combine_us": 10.315, "K_us": 0.08}

#: The landed 256 entry: the constants `fit_sm90_planner.py --head-dim 256 --fit --emit` printed
#: on data/h100_planner_d256_20261003.long.csv
#: (`profile/h100_planb_prd_20261002/planner/fit_A.txt`).
PINGPONG_MODEL_D256 = {"T_us": {2: 0.85, 4: 0.15},
                       "P_us": 0.0, "combine_us": 0.0, "K_us": 0.05}

#: The D=256 decisions of the landed 256 entry, in the format of DECISIONS_D128, generated from
#: that entry with this file's `_cell` over the same shapes. They differ from the legacy one-wave
#: rule's (the absent-entry test below) at B=1 from KV=16384 at both widths and at B=2 at INT4.
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
    from B=32 up. This includes the 32 and 33 splits at B=1 that the D=128 model replaced
    with 16. These are the picks the planner cells of the D=256 fit's ladder time, because that
    ladder runs with no entry."""
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


# ---------------------------------------------------------------------------------------------
# The planner fit, `baselines/fit_sm90_planner.py`
# ---------------------------------------------------------------------------------------------

_LADDER = "splits=1,combine=none=100.00;splits=4,combine=fused=90.00"


def _sweep(tmp_path, rows):
    import csv

    path = tmp_path / "sweep.long.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["column", "B", "KV", "D", "status",
                                                    "variant_kernel_us"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(column, head_dim, kv=8192, ladder=_LADDER):
    return {"column": column, "B": 1, "KV": kv, "D": head_dim, "status": "OK",
            "variant_kernel_us": ladder}


def test_the_fit_reads_only_the_rows_at_its_head_dimension(tmp_path):
    from baselines import fit_sm90_planner

    slow = "splits=1,combine=none=300.00;splits=4,combine=fused=200.00"
    path = _sweep(tmp_path, [_row("ours_i2", 128), _row("ours_i4", 128, kv=16384),
                             _row("ours_i2", 256, ladder=slow),
                             _row("bf16_fi", "", ladder=slow)])
    assert fit_sm90_planner.load(path, 128) == {2: {(1, 8192): {1: 100.0, 4: 90.0}},
                                                4: {(1, 16384): {1: 100.0, 4: 90.0}}}
    assert fit_sm90_planner.load(path, 256) == {2: {(1, 8192): {1: 300.0, 4: 200.0}}}


def test_the_fit_refuses_a_row_without_a_head_dimension(tmp_path):
    from baselines import fit_sm90_planner

    path = _sweep(tmp_path, [_row("ours_i2", 128), _row("ours_i4", "")])
    with pytest.raises(SystemExit, match="has no D value"):
        fit_sm90_planner.load(path, 128)


def test_the_fit_refuses_a_file_with_no_row_at_its_head_dimension(tmp_path):
    from baselines import fit_sm90_planner

    path = _sweep(tmp_path, [_row("ours_i2", 128), _row("ours_i4", 128)])
    with pytest.raises(SystemExit, match="no ours_i2/ours_i4 row at head_dim=256"):
        fit_sm90_planner.load(path, 256)


def _fit_main(monkeypatch, argv, drop=()):
    """`main` reaches the scheduler through the package, which imports the DSL, so the
    scheduler is loaded from its file under that name, with stub parents. The entries of
    `drop` are removed from that copy's PINGPONG_MODEL, which is how a head dimension with no
    entry is reached now that 128 and 256 both have one."""
    import sys
    import types

    from baselines import fit_sm90_planner

    for package in ("cutedsl_decode", "cutedsl_decode._sm90"):
        stub = types.ModuleType(package)
        stub.__path__ = []
        monkeypatch.setitem(sys.modules, package, stub)
    spec = importlib.util.spec_from_file_location("cutedsl_decode._sm90.scheduler", SCHEDULER)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    for head_dim in drop:
        del module.PINGPONG_MODEL[head_dim]
    monkeypatch.setattr(sys, "argv", ["fit", *argv])
    return fit_sm90_planner.main()


#: A synthetic D=256 ladder small enough for the fit's grid to finish quickly: KV=1024 is
#: 16 tiles, and the 512-token floor leaves splits 1 and 2 as the only candidates.
_SHORT_LADDER = "splits=1,combine=none=100.00;splits=2,combine=fused=90.00"


def _d256_sweep(tmp_path):
    return _sweep(tmp_path, [_row("ours_i2", 256, kv=1024, ladder=_SHORT_LADDER),
                             _row("ours_i4", 256, kv=1024, ladder=_SHORT_LADDER)])


def test_a_fresh_fit_runs_and_emits_an_entry_without_one(tmp_path, monkeypatch, capsys):
    """With no D=256 entry, --fit reads no seed and emits the fitted entry under a one-line
    comment naming the ladder, without the D=128 commentary."""
    path = _d256_sweep(tmp_path)
    assert _fit_main(monkeypatch, ["--csv", str(path), "--head-dim", "256", "--fit",
                                   "--emit"], drop=(256,)) == 0
    out = capsys.readouterr().out
    assert "# fitted on sweep.long.csv" in out
    assert "256: {   # the PINGPONG_MODEL entry for head_dim=256" in out
    assert '"T_us": {2: ' in out and '"K_us": ' in out
    assert "READ THIS FIRST" not in out


def test_without_fit_an_absent_entry_is_reported_and_exits_1(tmp_path, monkeypatch, capsys):
    path = _d256_sweep(tmp_path)
    assert _fit_main(monkeypatch, ["--csv", str(path), "--head-dim", "256"],
                     drop=(256,)) == 1
    out = capsys.readouterr().out
    assert "=== SHIPPED plan_splits" in out
    assert "no PINGPONG_MODEL entry at head_dim=256 to score" in out


def test_without_fit_the_landed_entry_is_scored(tmp_path, monkeypatch, capsys):
    """With the 256 entry present, a run without --fit scores that entry's constants and
    emits them under the comment that says they are the shipped entry, not a fit. The
    emitted block is compared whole: its comment, its field order and the four constants as
    the fit prints them, rounded to four places. The exact values are pinned by
    `test_the_model_has_the_d128_and_the_d256_entries`."""
    path = _d256_sweep(tmp_path)
    _fit_main(monkeypatch, ["--csv", str(path), "--head-dim", "256", "--emit"])
    out = capsys.readouterr().out
    assert "=== MODEL form=occ (shipped constants)" in out
    assert ("# the shipped entry, scored on sweep.long.csv\n"
            "256: {   # the PINGPONG_MODEL entry for head_dim=256\n"
            '    "T_us": {2: 0.85, 4: 0.15},\n'
            '    "P_us": 0.0, "combine_us": 0.0, "K_us": 0.05,\n'
            "},\n") in out


@pytest.mark.parametrize("form", ["waves", "width"])
def test_another_form_is_refused_at_256(tmp_path, monkeypatch, capsys, form):
    path = _d256_sweep(tmp_path)
    assert _fit_main(monkeypatch, ["--csv", str(path), "--head-dim", "256", "--fit",
                                   "--form", form]) == 2
    assert f"NO: --form {form} at head_dim=256" in capsys.readouterr().out
