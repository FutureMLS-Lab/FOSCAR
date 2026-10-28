"""CPU gates for `scripts/_drift_check_fold.py`, the differential output checker.

The checker is only worth having if it FAILS what the owner's drift rule rejects, so
most of these feed it a deliberately perturbed output and require a FAIL: an element
two ulp away, one ulp away on a column the rule does not allow, a nondeterministic
candidate, a non-finite value, a missing or an unverified cell. The two passes it must
give are an identical candidate and a one-ulp move on the head%4==0 columns only. It must
also refuse to judge at all when it has fewer than two base captures or when the base
captures disagree with each other, because a base that moves would otherwise let the
candidate's drift be measured against noise.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_test_drift_check_fold",
                                              REPO / "scripts/_drift_check_fold.py")
DC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DC
SPEC.loader.exec_module(DC)

B, HQ, HKV, D = 2, 32, 8, 16


def _base(seed=0, hq=HQ):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, hq, D, generator=g) * 3).to(torch.bfloat16)


def _step(t, idx, ulps):
    """`t` with element `idx` moved `ulps` positions along the ordered bf16 list."""
    out = t.clone()
    bits = out.view(torch.int16)
    u = int(bits[idx]) & 0xFFFF
    key = -(u & 0x7FFF) if u & 0x8000 else (u & 0x7FFF)
    key += ulps
    new = (0x8000 | (-key)) if key < 0 else key
    bits[idx] = new - 0x10000 if new >= 0x8000 else new
    return out


def _judge(base, runs, hq=HQ, hkv=HKV):
    return DC.judge_cell("c", base, runs, hq, hkv)


def test_ordered_bf16_is_monotone_and_merges_signed_zero():
    vals = torch.tensor([-2.0, -1.0, -0.0, 0.0, 1e-40, 1.0, 2.0]).to(torch.bfloat16)
    k = DC.ordered_bf16(vals)
    assert int(k[2]) == int(k[3]) == 0
    assert all(int(a) <= int(b) for a, b in zip(k[:-1], k[1:]))
    one = torch.tensor([1.0], dtype=torch.bfloat16)
    nxt = torch.tensor([1.0 + 2 ** -7], dtype=torch.bfloat16)   # the next bf16 above 1
    assert int(DC.ordered_bf16(nxt)[0] - DC.ordered_bf16(one)[0]) == 1


def test_step_helper_moves_by_exactly_the_requested_ulps():
    b = _base()
    for ulps in (1, -1, 2):
        m = _step(b, (0, 4, 3), ulps)
        d = DC.ordered_bf16(m) - DC.ordered_bf16(b)
        assert int(d[0, 4, 3]) == ulps and int((d != 0).sum()) == 1


def test_identical_candidate_passes():
    b = _base()
    v = _judge(b, [b.clone(), b.clone()])
    assert v.ok and v.differ == 0 and v.nondeterministic == 0


def test_one_ulp_on_head_mod4_columns_passes():
    b = _base()
    c = b
    for idx in ((0, 0, 1), (0, 4, 7), (1, 28, 0), (1, 8, 15)):
        c = _step(c, idx, 1 if idx[2] % 2 else -1)
    v = _judge(b, [c, c.clone()])
    assert v.ok, v.reasons
    assert v.differ == 4 and v.max_ulp == 1 and v.heads == [0, 4, 8, 28]


def test_one_ulp_on_another_column_fails():
    b = _base()
    c = _step(_step(b, (0, 0, 1), 1), (1, 5, 2), 1)
    v = _judge(b, [c, c.clone()])
    assert not v.ok and v.off_column == 1 and v.over_ulp == 0


def test_two_ulp_fails_even_on_an_allowed_column():
    b = _base()
    c = _step(b, (0, 4, 3), 2)
    v = _judge(b, [c, c.clone()])
    assert not v.ok and v.over_ulp == 1 and v.max_ulp == 2 and v.off_column == 0


def test_nondeterministic_candidate_fails():
    b = _base()
    c1 = _step(b, (0, 0, 0), 1)
    c2 = b.clone()
    v = _judge(b, [c1, c1.clone(), c2])
    assert not v.ok and v.nondeterministic == 1
    assert any("nondeterministic" in r for r in v.reasons)


def test_nondeterminism_between_bitwise_identical_to_base_runs_still_fails():
    """A run pair where neither run moved off an allowed column is still a FAIL when the
    two runs disagree: determinism is judged on its own, not through the base."""
    b = _base()
    c1 = _step(b, (0, 8, 0), 1)
    c2 = _step(b, (0, 8, 0), -1)
    v = _judge(b, [c1, c2])
    assert not v.ok and v.nondeterministic == 1


def test_one_run_is_not_enough():
    b = _base()
    assert not _judge(b, [b.clone()]).ok


def test_non_finite_fails():
    b = _base()
    c = b.clone()
    c[0, 0, 0] = float("nan")
    assert not _judge(b, [c, c.clone()]).ok
    c = b.clone()
    c[0, 4, 0] = float("inf")
    assert not _judge(b, [c, c.clone()]).ok


def test_signed_zero_counts_as_a_difference_on_the_column_rule():
    b = _base()
    b[0, 1, 0] = 0.0
    b[0, 4, 0] = 0.0
    c = b.clone()
    c[0, 4, 0] = -0.0
    v = _judge(b, [c, c.clone()])
    assert v.ok and v.differ == 1 and v.max_ulp == 0
    c = b.clone()
    c[0, 1, 0] = -0.0
    v = _judge(b, [c, c.clone()])
    assert not v.ok and v.off_column == 1


def test_dtype_and_shape_mismatches_fail():
    b = _base()
    assert not _judge(b, [b.float(), b.float()]).ok
    assert not _judge(b, [b[:, :16], b[:, :16]]).ok
    assert not DC.judge_cell("c", b.float(), [b.float(), b.float()], HQ, HKV).ok
    assert not _judge(b, [b, b], hq=HQ, hkv=5).ok    # 32 % 5
    assert not _judge(b, [b, b], hq=64, hkv=8).ok    # shape says 32 heads


def test_g8_allows_columns_0_and_4_of_each_group():
    """The owner's rule is literally "head%4==0 columns" (the comment at the folded-score
    store in `_sm100/int_fold_decode.py`; `docs/kernel/int_fold_decode.md` records the
    2026-09-14 drift at "head indices divisible by 4"). At 8 query heads per KV head that
    includes tile column 4, so the checker's (h % g) % 4 == 0 allows columns 0 and 4 of
    every group, which equals h % 4 == 0 here, and it is deliberately not h % g == 0."""
    b = _base(hq=64)
    c = _step(_step(b, (0, 4, 0), 1), (0, 12, 0), 1)       # columns 4 and 4
    assert _judge(b, [c, c.clone()], hq=64, hkv=8).ok
    c = _step(b, (0, 10, 0), 1)                             # column 2
    assert not _judge(b, [c, c.clone()], hq=64, hkv=8).ok


def test_padded_ratio_uses_the_tile_column():
    """At g = 3 query head h sits in tile column h % 3; head 3 is column 0 (allowed) and
    head 4 is column 1 (refused), the reverse of a literal h % 4 reading."""
    b = _base(hq=24)
    c = _step(b, (0, 3, 0), 1)
    assert _judge(b, [c, c.clone()], hq=24, hkv=8).ok
    c = _step(b, (0, 4, 0), 1)
    assert not _judge(b, [c, c.clone()], hq=24, hkv=8).ok


def _specs(names, hq=HQ, hkv=HKV):
    return {n: {"h_q": hq, "h_kv": hkv} for n in names}


def test_judge_fails_a_cell_missing_from_a_run_and_an_unverified_cell():
    base = {"a": _base(0), "b": _base(1)}
    runs = [dict(base), {"a": base["a"]}]
    verdicts, problems = DC.judge(base, runs, _specs(base))
    bad = {v.name for v in verdicts if not v.ok}
    assert bad == {"b"} and not problems
    runs = [dict(base, c=_base(2)), dict(base, c=_base(2))]
    verdicts, problems = DC.judge(base, runs, _specs(base))
    assert all(v.ok for v in verdicts) and problems
    assert not DC.report(verdicts, problems, out=open("/dev/null", "w"))


def test_judge_fails_a_cell_without_a_spec():
    base = {"a": _base(0)}
    verdicts, _ = DC.judge(base, [dict(base), dict(base)], {})
    assert not verdicts[0].ok


def test_cells_filter_scopes_the_judgement():
    base = {"a": _base(0), "b": _base(1)}
    runs = [{"a": base["a"]}, {"a": base["a"]}]
    verdicts, problems = DC.judge(base, runs, _specs(base), cells={"a"})
    assert [v.name for v in verdicts] == ["a"] and verdicts[0].ok and not problems
    verdicts, problems = DC.judge(base, runs, _specs(base), cells={"a", "zz"})
    assert problems


def _write(tmp, name, d):
    p = tmp / name
    torch.save(d, p)
    return str(p)


@pytest.mark.parametrize("case,expect_pass", [
    ("identical", True),
    ("ulp1_head_mod4", True),
    ("ulp1_other_column", False),
    ("ulp2", False),
    ("nondeterministic", False),
])
def test_cli_verdicts(tmp_path, case, expect_pass):
    """The command line, end to end on files in the anchor's saved format."""
    base = {"flex_i4_x": _base(0), "i4_y": _base(1)}
    c = dict(base)
    other = dict(base)
    if case == "ulp1_head_mod4":
        c["flex_i4_x"] = _step(base["flex_i4_x"], (1, 12, 3), 1)
        other = c
    elif case == "ulp1_other_column":
        c["flex_i4_x"] = _step(base["flex_i4_x"], (1, 13, 3), 1)
        other = c
    elif case == "ulp2":
        c["i4_y"] = _step(base["i4_y"], (0, 0, 0), 2)
        other = c
    elif case == "nondeterministic":
        c["i4_y"] = _step(base["i4_y"], (0, 0, 0), 1)
    else:
        other = c
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(_specs(base)))
    bp = _write(tmp_path, "base.pt", base)
    bp1 = _write(tmp_path, "base1.pt", {k: v.clone() for k, v in base.items()})
    r0 = _write(tmp_path, "r0.pt", c)
    r1 = _write(tmp_path, "r1.pt", dict(other))
    rep = tmp_path / "rep.json"
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts/_drift_check_fold.py"), "--base", bp, bp1,
         "--candidate", r0, r1, "--spec-json", str(spec), "--report-json", str(rep)],
        capture_output=True, text=True)
    last = proc.stdout.strip().splitlines()[-1]
    assert last == f"DRIFT CHECK: {'PASS' if expect_pass else 'FAIL'}", proc.stdout
    assert (proc.returncode == 0) == expect_pass
    assert "BASE CHECK: PASS" in proc.stdout
    assert json.loads(rep.read_text())["pass"] == expect_pass


def test_cli_refuses_a_single_candidate_run(tmp_path):
    base = {"a": _base(0)}
    bp = _write(tmp_path, "base.pt", base)
    bp1 = _write(tmp_path, "base1.pt", {k: v.clone() for k, v in base.items()})
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts/_drift_check_fold.py"), "--base", bp, bp1,
         "--candidate", bp, "--spec-json", str(tmp_path / "none.json")],
        capture_output=True, text=True)
    assert proc.returncode != 0 and "--candidate needs at least two" in proc.stderr


# ------------------------------------------------------------------ the base captures

def test_agreeing_base_captures_raise_no_problem():
    base = {"a": _base(0), "b": _base(1)}
    copy = {k: v.clone() for k, v in base.items()}
    assert DC.base_disagreements([base, copy]) == []
    assert DC.base_disagreements([base, copy, dict(copy)]) == []


def test_one_base_capture_is_a_problem():
    assert DC.base_disagreements([{"a": _base(0)}])


def test_base_captures_one_ulp_apart_on_an_allowed_column_are_a_problem():
    """The case the check exists for: a move the drift rule itself would allow, so only
    the base comparison can see it."""
    b0 = {"a": _base(0), "b": _base(1)}
    b1 = dict(b0, b=_step(b0["b"], (1, 12, 3), 1))
    problems = DC.base_disagreements([b0, b1])
    assert len(problems) == 1 and problems[0].startswith("b: base capture 1 differs")
    assert "1 elements" in problems[0]
    # A difference in a third capture only is found as well.
    assert DC.base_disagreements([b0, dict(b0), b1])


def test_base_captures_that_differ_in_dtype_shape_or_cells_are_a_problem():
    b0 = {"a": _base(0), "b": _base(1)}
    assert DC.base_disagreements([b0, dict(b0, a=b0["a"].float())])
    assert DC.base_disagreements([b0, dict(b0, a=b0["a"][:, :16])])
    assert any("other cells" in p for p in DC.base_disagreements([b0, {"a": b0["a"]}]))
    assert any("other cells" in p
               for p in DC.base_disagreements([b0, dict(b0, c=_base(2))]))
    # A non-bf16 tensor is still compared bit for bit.
    f = {"a": torch.zeros(2, 3)}
    g = {"a": torch.zeros(2, 3)}
    g["a"][0, 0] = -0.0
    assert DC.base_disagreements([f, g])


def test_a_scoped_judgement_checks_only_the_selected_cells_of_the_base():
    b0 = {"a": _base(0), "b": _base(1)}
    b1 = dict(b0, b=_step(b0["b"], (0, 0, 0), 1))
    assert DC.base_disagreements([b0, b1], cells={"a"}) == []
    assert DC.base_disagreements([b0, b1], cells={"b"})
    assert DC.base_disagreements([b0, {"b": b0["b"]}], cells={"a"})   # a in capture 0 only


def _cli(tmp_path, bases, candidates, *extra):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(_specs(bases[0])))
    bps = [_write(tmp_path, f"base{i}.pt", b) for i, b in enumerate(bases)]
    rps = [_write(tmp_path, f"r{i}.pt", r) for i, r in enumerate(candidates)]
    rep = tmp_path / "rep.json"
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts/_drift_check_fold.py"), "--base", *bps,
         "--candidate", *rps, "--spec-json", str(spec), "--report-json", str(rep),
         *extra], capture_output=True, text=True)
    return proc, rep


def test_cli_refuses_a_single_base_capture(tmp_path):
    base = {"a": _base(0)}
    bp = _write(tmp_path, "base.pt", base)
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts/_drift_check_fold.py"), "--base", bp,
         "--candidate", bp, bp, "--spec-json", str(tmp_path / "none.json")],
        capture_output=True, text=True)
    assert proc.returncode != 0 and "--base needs at least two" in proc.stderr


@pytest.mark.parametrize("flag", ["--base", "--candidate"])
def test_cli_refuses_one_file_named_twice(tmp_path, flag):
    """Two names for one capture would pass either comparison vacuously, because a file
    always agrees with itself; a second path to the same file is refused as well."""
    base = {"a": _base(0)}
    files = [_write(tmp_path, f"f{i}.pt", {k: v.clone() for k, v in base.items()})
             for i in range(3)]
    alias = tmp_path / "alias.pt"
    alias.symlink_to(files[0])
    twice = [files[0], str(alias)]
    bases, cands = (twice, files[1:]) if flag == "--base" else (files[1:], twice)
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts/_drift_check_fold.py"), "--base", *bases,
         "--candidate", *cands, "--spec-json", str(tmp_path / "none.json")],
        capture_output=True, text=True)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert f"{flag} names the same file more than once" in proc.stderr


def test_cli_fails_before_judging_when_the_base_captures_differ(tmp_path):
    """Without the base check this candidate passes: it is capture 1 exactly, and capture
    1 is one allowed ulp from capture 0. The check must fail the run and judge nothing."""
    b0 = {"flex_i4_x": _base(0), "i4_y": _base(1)}
    b1 = dict(b0, flex_i4_x=_step(b0["flex_i4_x"], (1, 12, 3), 1))
    judged_alone = DC.judge(b0, [b1, dict(b1)], _specs(b0))
    assert DC.report(*judged_alone, out=open("/dev/null", "w"))    # the hole it closes
    proc, rep = _cli(tmp_path, [b0, b1], [b1, dict(b1)])
    lines = proc.stdout.strip().splitlines()
    assert proc.returncode == 1 and lines[-1] == "DRIFT CHECK: FAIL", proc.stdout
    assert any(line.startswith("BASE CHECK: FAIL") for line in lines)
    assert any("flex_i4_x: base capture 1 differs" in line for line in lines)
    assert not any(line.startswith("[flex_i4_x]") or line.startswith("[i4_y]")
                   for line in lines), "a candidate cell was judged"
    report = json.loads(rep.read_text())
    assert report["pass"] is False and report["cells"] == [] and report["base_problems"]


def test_cli_scoped_to_agreeing_cells_still_judges(tmp_path):
    b0 = {"flex_i4_x": _base(0), "i4_y": _base(1)}
    b1 = dict(b0, flex_i4_x=_step(b0["flex_i4_x"], (1, 12, 3), 1))
    proc, _ = _cli(tmp_path, [b0, b1], [dict(b0), dict(b0)], "--cells", "i4_y")
    assert proc.returncode == 0 and "BASE CHECK: PASS" in proc.stdout, proc.stdout
    assert "[i4_y] IDENTICAL" in proc.stdout and "[flex_i4_x]" not in proc.stdout


def test_runs_mode_checks_the_base_before_it_captures(tmp_path):
    """A --runs capture costs minutes of GPU time, so a base that disagrees with itself
    must stop the run before the first capture starts."""
    b0 = {"a": _base(0)}
    b1 = {"a": _step(b0["a"], (0, 0, 0), 1)}
    bps = [_write(tmp_path, f"base{i}.pt", b) for i, b in enumerate((b0, b1))]
    work = tmp_path / "work"
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts/_drift_check_fold.py"), "--base", *bps,
         "--runs", "2", "--gpus", "0", "--workdir", str(work),
         "--anchor", str(tmp_path / "no_such_anchor.py")],
        capture_output=True, text=True)
    assert proc.returncode == 1 and "BASE CHECK: FAIL" in proc.stdout, proc.stdout
    assert not work.exists() and "capture 1/" not in proc.stdout
