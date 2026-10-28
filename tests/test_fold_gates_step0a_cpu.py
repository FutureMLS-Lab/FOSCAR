"""CPU gates for the step-0a additions to the SM100 correctness gates.

`scripts/_anchor_fold_outputs.py`, `scripts/test_fold_quant_flex_sm100.py` and
`scripts/soak_fold_sm100.py` import the kernel stack at module level, so they are loaded
here against STUB modules that record what the scripts hand them. That checks the
manifest (which cells exist, what each claims, which swizzle phase each covers) and that
the head counts a cell or a run names are the ones that reach the pool builder and the
compile. It cannot check an output; the B200 runs recorded in the step-0a evidence do.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def _load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


QE = _load_path("_step0a_quant_envelope", SCRIPTS / "_quant_envelope.py")
QL = _load_path("_step0a_quant_layout", REPO / "src/cutedsl_decode/quant_layout.py")


def _npo2(x):
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


class Recorder:
    """The stub harness: every call is recorded, and `compile_fold` hands back a kernel
    object carrying what a constructor derives from the same arguments."""

    def __init__(self):
        self.calls = []

    def _rec(self, fn, *a, **kw):
        self.calls.append((fn, a, kw))

    def compile_fold(self, **kw):
        self._rec("compile_fold", **kw)
        red = kw.get("reduction")
        kern = types.SimpleNamespace(
            q_head_tile=max(min(32, _npo2(kw["num_qo_heads"] // kw["num_kv_heads"])), 2),
            do_none_red=red == "none", do_kernel_red=red == "kernel",
            k_quant=tuple(kw.get("k_quant", ("token", 128))),
            v_quant=tuple(kw.get("v_quant", ("token", 128))))
        # like a constructor, it stores every other knob under its own name
        for k, v in kw.items():
            if k not in self._SHAPE_ARGS:
                setattr(kern, k, v)
        if "kernel_out" in kw:
            kw["kernel_out"]["kernel"] = kern
        return ("compiled", kw.get("num_qo_heads"), kw.get("num_kv_heads"))

    _SHAPE_ARGS = {"page_size", "num_qo_heads", "num_kv_heads", "head_dim", "reduction",
                   "bits", "kernel_out", "k_quant", "v_quant", "sdtype"}

    def problem(self, fn, bits, seq_lens, h_q, h_kv, **kw):
        self._rec(fn, bits, seq_lens, h_q, h_kv, **kw)
        b = len(seq_lens)
        return dict(b=b, h_q=h_q, h_kv=h_kv, qs_q=8.0, q_fp8_cpu=None,
                    block_table=None, seq_lens_cpu=None,
                    pools_cpu=(None, None, torch.zeros(1)),
                    pools_dev=(None, None, torch.zeros(1)),
                    out=torch.randn(b, h_q, 128).to(torch.bfloat16))

    def run_fold(self, compiled, p, splits, reduction, scale, **kw):
        self._rec("run_fold", compiled, splits, reduction)
        return p["out"]

    def names(self, fn):
        return [c for c in self.calls if c[0] == fn]


def _stub_module(name, attrs):
    """A module holding `attrs`, bound BY NAME. The pool builders and unpackers are names
    `tests/test_pool_call_sites_cpu.py` forbids as attribute values, which is right for
    code that reaches a real builder; a stub that stands in for one is not that, so its
    bindings are written as data."""
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _fake_builder(rec, fn):
    return lambda bits, sl, hq, hkv, **kw: rec.problem(fn, bits, sl, hq, hkv, **kw)


def _harness_stub(rec):
    return _stub_module("_fold_harness", {
        "K_SET": QE.K_SET, "V_SET": QE.V_SET,
        "compile_fold": rec.compile_fold, "run_fold": rec.run_fold,
        "make_pool_problem": _fake_builder(rec, "make_pool_problem"),
        "make_pool_problem_flex": _fake_builder(rec, "make_pool_problem_flex"),
        "v_pool_perm": lambda bits, kkw=None: ("vperm", bits),
        "k_pool_perm": lambda bits, kkw=None: ("kperm", bits),
        "global_pscale_lp": lambda *a, **k: 3.0,
        "global_pscale_lp_flex": lambda *a, **k: 3.0,
        "alloc_fold_workspace": lambda *a, **k: None,
    })


def _flex_bench_stub(rec):
    m = types.ModuleType("bench_fold_quant_flex_sm100")

    def make_problem(B, KV, bits, kq, vq):
        return rec.problem("make_problem", bits, [KV] * B, 32, 8, k_quant=kq, v_quant=vq)
    m.make_problem = make_problem
    return m


@pytest.fixture
def rec(monkeypatch):
    r = Recorder()
    monkeypatch.setitem(sys.modules, "_fold_harness", _harness_stub(r))
    monkeypatch.setitem(sys.modules, "bench_fold_quant_flex_sm100", _flex_bench_stub(r))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    return r


@pytest.fixture
def anchor(rec, monkeypatch):
    A = _load_path("_step0a_anchor", SCRIPTS / "_anchor_fold_outputs.py")
    return A


# ------------------------------------------------------------------- anchor manifest

def _manifest(A):
    cells = A.build_cells()
    return [n for n, _ in cells], dict(cells), A.CELL_SPEC


PAIRS = [(kq, vq) for kq in QE.K_SET for vq in QE.V_SET]


def test_int4_g8_cells_cover_every_pair_at_both_reductions(anchor):
    names, _, spec = _manifest(anchor)
    assert len(PAIRS) == 20
    for red, splits in (("kernel", 4), ("none", 1)):
        got = {(s["k_quant"], s["v_quant"]) for n, s in spec.items()
               if s["bits"] == 4 and (s["h_q"], s["h_kv"]) == (64, 8)
               and s["reduction"] == red and s["splits"] == splits and not s["knobs"]}
        assert got == set(PAIRS), (red, set(PAIRS) - got)


def test_int2_g8_cells_cover_every_pair_at_both_reductions(anchor):
    """The INT2 twin of the INT4 g8 test. The legacy pair at reduction "kernel" is the
    pre-existing `i2_g8`, so the new cells are the other 39."""
    names, _, spec = _manifest(anchor)
    for red, splits in (("kernel", 4), ("none", 1)):
        got = {(s["k_quant"], s["v_quant"]) for n, s in spec.items()
               if s["bits"] == 2 and (s["h_q"], s["h_kv"]) == (64, 8)
               and s["reduction"] == red and s["splits"] == splits and not s["knobs"]}
        assert got == set(PAIRS), (red, set(PAIRS) - got)
    knob, flex, _ = anchor._int2_g8_cells()
    assert [r[0] for r in knob] == ["i2_g8_none"] and len(flex) == 38
    assert {r[0] for r in flex} == {f"flex_i2_{label}_g8{suffix}"
                                    for label, _, _ in anchor.FLEX_CONFIGS
                                    for suffix in ("", "_none")}
    # `i2_g8` is the legacy kernel-reduction member, on the same row a generated cell
    # would have: same builder inputs, same split, same sequence lengths.
    row = next(c for c in anchor.KNOB_CELLS if c[0] == "i2_g8")
    assert row[1:6] == (2, {}, "kernel", 4, [anchor.FLEX_KV] * anchor.FLEX_B)
    assert row[6] == {"num_qo_heads": 64, "num_kv_heads": 8}
    assert anchor.CELL_EXPECTATIONS["i2_g8"] == {"q_head_tile": 8, "do_kernel_red": True}


def test_every_flex_cell_also_runs_at_reduction_none(anchor):
    names, _, spec = _manifest(anchor)
    kernel = {(s["bits"], s["k_quant"], s["v_quant"]) for n, s in spec.items()
              if n.startswith("flex_") and s["reduction"] == "kernel"
              and (s["h_q"], s["h_kv"]) == (32, 8)}
    none = {(s["bits"], s["k_quant"], s["v_quant"]) for n, s in spec.items()
            if n.startswith("flex_") and s["reduction"] == "none" and s["splits"] == 1
            and (s["h_q"], s["h_kv"]) == (32, 8)}
    assert len(kernel) == 38 and none == kernel


def test_extra_head_counts_are_present_and_padded_at_both_tiles(anchor):
    names, _, spec = _manifest(anchor)
    heads = {(s["h_q"], s["h_kv"]) for s in spec.values()}
    assert {(24, 8), (40, 8), (16, 4), (64, 16)} <= heads
    tiles = {(s["h_q"], s["h_kv"]): s["tile"] for s in spec.values()}
    # a padded ratio (g not a power of two) at each tile, and HKV != 8
    assert tiles[(24, 8)] == 4 and tiles[(40, 8)] == 8
    assert any(hkv != 8 for _, hkv in heads)


def test_new_cells_are_appended_and_every_name_is_unique(anchor):
    """Step-0a cells follow every earlier cell, and the INT2 g8 cells follow those, so
    each group keeps the places it had when the anchors were saved."""
    names, _, _ = _manifest(anchor)
    knob, flex, _ = anchor._step0a_cells()
    i2_knob, i2_flex, _ = anchor._int2_g8_cells()
    step0a = [r[0] for r in knob] + [r[0] for r in flex]
    i2g8 = [r[0] for r in i2_knob] + [r[0] for r in i2_flex]
    new = step0a + i2g8
    assert len(names) == len(set(names))
    assert names[-len(new):] == new
    assert names[-len(i2g8):] == i2g8
    assert not set(new) & set(names[:-len(new)])
    assert len(names) - len(i2g8) == 172 and len(i2g8) == 39


def test_every_new_cell_claims_its_tile_and_reduction(anchor):
    _manifest(anchor)
    knob, flex, _ = anchor._step0a_cells()
    i2_knob, i2_flex, _ = anchor._int2_g8_cells()
    for r in knob + i2_knob:
        want = anchor.CELL_EXPECTATIONS[r[0]]
        assert want["q_head_tile"] == anchor.HEADS_TILE[(r[6]["num_qo_heads"],
                                                          r[6]["num_kv_heads"])]
        assert want.get("do_none_red" if r[3] == "none" else "do_kernel_red") is True
    for name, bits, kq, vq, red, splits, hq, hkv in flex + i2_flex:
        want = anchor.CELL_EXPECTATIONS[name]
        assert want["q_head_tile"] == anchor.HEADS_TILE[(hq, hkv)]
        assert want["k_quant"] == kq and want["v_quant"] == vq
        assert want.get("do_none_red" if red == "none" else "do_kernel_red") is True


def test_heads_tile_literals_agree_with_the_packing_rule(anchor):
    """HEADS_TILE is written out as literals so it is a claim; this checks the claim
    against the documented packing rule (`fold_runtime.blk_tile_n_of`)."""
    for (hq, hkv), tile in anchor.HEADS_TILE.items():
        assert tile == max(min(32, _npo2(hq // hkv)), 2)


def test_ring_start_rule_reproduces_the_table(anchor):
    """Stage counts per class read off the 2026-09-29 carve-out table (default knobs)."""
    ors = anchor.operand_ring_start
    # INT4 token-128 / token-128: kv 7, code 16, scale 8, no Q' ring, s 4, p 4, o 3
    assert ors(7, 16, 8, 0, 4, 4, 3) == 0x700
    # INT4 channel-64 / channel-64: kv 5, qp (3+2)*4, o_ring 3*4
    assert ors(5, 16, 8, 20, 4, 4, 12) == 0xa00
    # INT4 channel-128 / channel-128: kv 6, qp (3+2)*2, o_ring 3*2
    assert ors(6, 16, 8, 10, 4, 4, 6) == 0x880
    t = anchor.INT4_OPERAND_RING_START
    assert t[(("token", 128), ("token", 128))] == (0x700, 0x700)
    assert t[(("channel", 64), ("channel", 64))] == (0xa00, 0xa00)
    assert set(t) == set(PAIRS)


def test_every_swizzle_phase_is_covered_at_both_tiles(anchor):
    _, _, spec = _manifest(anchor)
    t = anchor.INT4_OPERAND_RING_START
    for tile, idx in ((4, 0), (8, 1)):
        phases = {t[(s["k_quant"], s["v_quant"])][idx] % 512 for s in spec.values()
                  if s["bits"] == 4 and s["tile"] == tile and not s["knobs"]
                  and (s["h_q"], s["h_kv"]) in ((32, 8), (64, 8))}
        assert phases == {0, 128, 256, 384}, (tile, phases)
    for (tile, phase), names in anchor.SW64_PHASE_CELLS.items():
        for n in names:
            s = spec[n]
            assert s["tile"] == tile
            assert t[(s["k_quant"], s["v_quant"])][tile // 8] % 512 == phase


def test_phase_table_refuses_a_misfiled_cell(anchor, monkeypatch):
    bad = dict(anchor.SW64_PHASE_CELLS)
    bad[(4, 0)] = bad[(4, 0)] + ("flex_i4_Kch128_Vch128",)     # a phase-128 cell
    monkeypatch.setattr(anchor, "SW64_PHASE_CELLS", bad)
    with pytest.raises(SystemExit, match="phase 128"):
        anchor.build_cells()


def test_phase_table_refuses_a_missing_slot_or_cell(anchor, monkeypatch):
    bad = dict(anchor.SW64_PHASE_CELLS)
    del bad[(8, 384)]
    monkeypatch.setattr(anchor, "SW64_PHASE_CELLS", bad)
    with pytest.raises(SystemExit, match="every"):
        anchor.build_cells()
    bad = dict(anchor.SW64_PHASE_CELLS)
    bad[(8, 384)] = ("flex_i4_no_such_cell",)
    monkeypatch.setattr(anchor, "SW64_PHASE_CELLS", bad)
    with pytest.raises(SystemExit, match="not in the manifest"):
        anchor.build_cells()


def _run(anchor, name, rec):
    rec.calls.clear()
    out = dict(anchor.build_cells())[name]()
    return out, rec


def test_g8_flex_cell_reaches_the_builder_and_compile_at_64_8(anchor, rec):
    out, rec = _run(anchor, "flex_i4_Kch64_Vtok128_g8_none", rec)
    (_, a, kw), = rec.names("make_pool_problem_flex")
    assert a[0] == 4 and a[2:] == (64, 8) and a[1] == [anchor.FLEX_KV] * anchor.FLEX_B
    assert kw["k_quant"] == ("channel", 64) and kw["v_quant"] == ("token", 128)
    assert kw["k_store_perm"] == ("kperm", 4) and kw["v_store_perm"] == ("vperm", 4)
    (_, _, ckw), = rec.names("compile_fold")
    assert (ckw["num_qo_heads"], ckw["num_kv_heads"]) == (64, 8)
    assert ckw["reduction"] == "none" and ckw["bits"] == 4
    (_, ra, _), = rec.names("run_fold")
    assert ra[1:] == (1, "none")
    assert tuple(out.shape) == (anchor.FLEX_B, 64, 128)


def test_int2_g8_cells_reach_the_builders_and_compile_at_64_8(anchor, rec):
    out, rec = _run(anchor, "flex_i2_Ktok64_Vch128_g8", rec)
    (_, a, kw), = rec.names("make_pool_problem_flex")
    assert a[0] == 2 and a[2:] == (64, 8) and a[1] == [anchor.FLEX_KV] * anchor.FLEX_B
    assert kw["k_quant"] == ("token", 64) and kw["v_quant"] == ("channel", 128)
    assert kw["k_store_perm"] == ("kperm", 2) and kw["v_store_perm"] == ("vperm", 2)
    assert "rng_device" not in kw            # the anchored CPU fixture, by default
    (_, _, ckw), = rec.names("compile_fold")
    assert (ckw["num_qo_heads"], ckw["num_kv_heads"]) == (64, 8)
    assert ckw["reduction"] == "kernel" and ckw["bits"] == 2 and ckw["sdtype"] == "bf16"
    (_, ra, _), = rec.names("run_fold")
    assert ra[1:] == (4, "kernel")
    assert tuple(out.shape) == (anchor.FLEX_B, 64, 128)
    _, rec = _run(anchor, "i2_g8_none", rec)
    (_, a, kw), = rec.names("make_pool_problem")
    assert a[0] == 2 and a[2:] == (64, 8) and "k_store_perm" in kw
    (_, _, ckw), = rec.names("compile_fold")
    assert ckw["reduction"] == "none" and (ckw["num_qo_heads"], ckw["num_kv_heads"]) == (64, 8)
    (_, ra, _), = rec.names("run_fold")
    assert ra[1:] == (1, "none")


def test_g4_none_cell_uses_the_existing_flex_fixture(anchor, rec):
    _, rec = _run(anchor, "flex_i2_Kch128_Vch128_none", rec)
    assert len(rec.names("make_problem")) == 1 and not rec.names("make_pool_problem_flex")
    (_, _, ckw), = rec.names("compile_fold")
    assert ckw["reduction"] == "none" and (ckw["num_qo_heads"], ckw["num_kv_heads"]) == (32, 8)


def test_legacy_g8_and_extra_head_cells_use_the_knob_builder(anchor, rec):
    for name, heads, red in (("i4_g8", (64, 8), "kernel"),
                             ("i4_hq40_hkv8_none", (40, 8), "none"),
                             ("i4_hq64_hkv16", (64, 16), "kernel")):
        _, rec = _run(anchor, name, rec)
        (_, a, kw), = rec.names("make_pool_problem")
        assert a[0] == 4 and a[2:] == heads and "k_store_perm" in kw
        (_, _, ckw), = rec.names("compile_fold")
        assert (ckw["num_qo_heads"], ckw["num_kv_heads"]) == heads
        assert ckw["reduction"] == red


def test_expectations_fail_when_the_kernel_resolves_another_tile(anchor, rec, monkeypatch):
    """The claim is live: a constructor resolving a different tile fails the cell."""
    real = rec.compile_fold

    def wrong_tile(**kw):
        res = real(**kw)
        kw["kernel_out"]["kernel"].q_head_tile = 4
        return res
    monkeypatch.setattr(anchor, "compile_fold", wrong_tile)
    with pytest.raises(AssertionError, match="q_head_tile"):
        dict(anchor.build_cells())["flex_i4_Kch128_Vch128_g8"]()


def _main(anchor, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["_anchor_fold_outputs.py"] + argv)
    buf = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buf):
        try:
            anchor.main()
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    return code, buf.getvalue()


def test_list_output_is_still_names_then_a_count(anchor, monkeypatch):
    code, out = _main(anchor, ["--list"], monkeypatch)
    lines = out.strip().splitlines()
    assert code == 0 and lines[-1] == f"{len(lines) - 1} cells"
    assert "i4_g8" in lines and "flex_i4_Kch64_Vtok128_g8_none" in lines


def test_spec_json_writes_the_selected_cells(anchor, monkeypatch, tmp_path):
    p = tmp_path / "spec.json"
    code, out = _main(anchor, ["--only", "i4_g8,flex_i2_Kch128_Vch128_hq24_hkv8",
                               "--spec-json", str(p)], monkeypatch)
    assert code == 0 and out == ""
    d = json.loads(p.read_text())
    assert sorted(d) == ["flex_i2_Kch128_Vch128_hq24_hkv8", "i4_g8"]
    assert (d["i4_g8"]["h_q"], d["i4_g8"]["h_kv"], d["i4_g8"]["tile"]) == (64, 8, 8)
    assert d["flex_i2_Kch128_Vch128_hq24_hkv8"]["k_quant"] == ["channel", 128]


def test_phase_report_refuses_to_combine_and_reads_the_trace(anchor, rec, monkeypatch):
    code, _ = _main(anchor, ["--phase-report", "--list"], monkeypatch)
    assert code != 0
    counts = {"kv_stages": 7, "code_stages": 16, "scale_stages": 8, "qp_stages": 0,
              "s_stages": 4, "p_stages": 4, "o_ring": 3}

    def traced(**kw):
        print("[FOLD_TRACE_TILING] sequence_tile=256 mma_tile_mnk=(128, 16, 32) "
              + " ".join(f"{k}={v}" for k, v in counts.items()), flush=True)
        kw["kernel_out"]["kernel"] = types.SimpleNamespace()
        return None
    monkeypatch.setattr(anchor, "compile_fold", traced)
    code, out = _main(anchor, ["--phase-report", "--only", "i4_default_s4,i4_g8"],
                      monkeypatch)
    assert code == 0 and "PHASE REPORT (designated, 2 configurations): PASS" in out
    counts["kv_stages"] = 20                 # a deeper ring: the start moves to 0x800
    code, out = _main(anchor, ["--phase-report", "--only", "i4_default_s4,i4_g8"],
                      monkeypatch)
    assert code == 1 and "MISMATCH" in out


PHASE_TRACE = ("[FOLD_TRACE_TILING] sequence_tile=256 mma_tile_mnk=(128, 16, 32) "
               "kv_stages=7 code_stages=16 scale_stages=8 qp_stages=0 s_stages=4 "
               "p_stages=4 o_ring=3")


@pytest.mark.parametrize("drops", [False, True])
def test_phase_report_holds_kernel_kwargs_to_the_readback(anchor, rec, monkeypatch,
                                                           drops):
    """`--phase-report` compiles with `--kernel-kwargs` like every other mode, so it reads
    each keyword back off the constructed kernel too. A keyword the constructor drops
    leaves the ring where the base puts it, so without the readback the report would PASS
    for an arm that never reached the kernel."""
    def traced(**kw):
        print(PHASE_TRACE, flush=True)
        res = rec.compile_fold(**kw)       # stores every knob under its own name
        if drops:
            del kw["kernel_out"]["kernel"].arm_knob
        return res
    monkeypatch.setattr(anchor, "compile_fold", traced)
    code, out = _main(anchor, ["--phase-report", "--only", "i4_default_s4,i4_g8",
                               "--kernel-kwargs", '{"arm_knob": "x"}'], monkeypatch)
    compiles = rec.names("compile_fold")
    assert len(compiles) == 2 and all(c[2]["arm_knob"] == "x" for c in compiles)
    assert all("kernel_out" in c[2] for c in compiles)
    last = out.strip().splitlines()[-1]
    if drops:
        assert code == 1 and last.endswith("FAIL"), out
        assert out.count("READBACK-FAIL") == 2 and "no attribute 'arm_knob'" in out
    else:
        assert code == 0 and last.endswith("PASS") and "READBACK-FAIL" not in out, out


# ------------------------------------------------------------------- flex oracle

@pytest.fixture
def oracle(rec, monkeypatch):
    rp = _stub_module("repack_sm100", {
        "_k_store_perm_inverse": lambda p: p,
        "dequant_pools_flex": lambda *a, **k: (None, None),
        "ref_paged_attention_fp8q_flex": lambda *a, **k: None,
        "unpack_rows_lut": lambda *a, **k: None,
    })
    monkeypatch.setitem(sys.modules, "repack_sm100", rp)
    pkg = types.ModuleType("cutedsl_decode")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "cutedsl_decode", pkg)
    monkeypatch.setitem(sys.modules, "cutedsl_decode.quant_layout", QL)
    monkeypatch.delenv("FLEX_HEADS", raising=False)
    monkeypatch.delenv("FLEX_ONLY", raising=False)
    return _load_path("_step0a_flex_oracle", SCRIPTS / "test_fold_quant_flex_sm100.py")


def test_oracle_heads_resolution(oracle):
    r = oracle.resolve_heads
    assert r(None, {}) == [(32, 8)]
    assert r(None, {"FLEX_HEADS": "32,4"}) == [(32, 4)]
    assert r(["64,8", "32,4", "64,8"], {}) == [(64, 8), (32, 4)]
    assert r(None, {}, gate=True) == list(oracle.GATE_HEADS) == [(32, 8), (32, 4)]
    for bad in (dict(heads_args=["32,4"], env={"FLEX_HEADS": "32,8"}),
                dict(heads_args=["32,4"], env={}, gate=True),
                dict(heads_args=None, env={"FLEX_HEADS": "32,8"}, gate=True)):
        with pytest.raises(ValueError, match="not"):
            r(**bad)
    for spec in ("32", "32,8,1", "a,b", "32,5", "0,8", "-8,8"):
        with pytest.raises(ValueError):
            oracle.parse_heads(spec)


def test_oracle_t3_runs_at_the_heads_it_is_given(oracle, rec, monkeypatch):
    monkeypatch.setattr(oracle, "ref_channel_k_flex", lambda *a, **k: None)
    monkeypatch.setattr(oracle, "worst_rel", lambda out, ref: 1e-4)
    monkeypatch.setenv("FLEX_ONLY", "Kch128/Vch128")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert oracle.t3_kernel_cells(32, 4)
    builds = rec.names("make_pool_problem_flex")
    assert builds and all(c[1][2:] == (32, 4) for c in builds)
    assert all("k_store_perm" in c[2] for c in builds)
    comps = rec.names("compile_fold")
    assert comps and all((c[2]["num_qo_heads"], c[2]["num_kv_heads"]) == (32, 4)
                         for c in comps)
    assert "h32/4" in buf.getvalue() and "= 6 cells" in buf.getvalue()


def test_oracle_t3_fails_when_the_filter_matches_nothing(oracle, rec, monkeypatch):
    monkeypatch.setenv("FLEX_ONLY", "Knope/Vnope")
    with contextlib.redirect_stdout(io.StringIO()):
        assert not oracle.t3_kernel_cells(32, 8)


def test_oracle_list(oracle, capsys):
    assert oracle.main(["--list", "--gate"]) == 0
    out = capsys.readouterr().out
    assert "heads 32/8 (g=4)" in out and "heads 32/4 (g=8)" in out
    assert "2 head counts x 19 configs" in out


# ------------------------------------------------------------------- soak

@pytest.fixture
def soak(rec, monkeypatch):
    tv = types.ModuleType("tvm_ffi")
    tv.use_torch_stream = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "tvm_ffi", tv)
    return _load_path("_step0a_soak", SCRIPTS / "soak_fold_sm100.py")


def test_soak_default_heads_keep_the_random_code_fixture(soak, rec):
    with contextlib.redirect_stdout(io.StringIO()):
        assert soak.soak_cell("x", ("channel", 64), ("channel", 64), 4, 2, 4096, 1, 2,
                              False)
    assert len(rec.names("make_problem")) == 1
    (_, _, ckw), = rec.names("compile_fold")
    assert (ckw["num_qo_heads"], ckw["num_kv_heads"]) == (32, 8)


@pytest.mark.parametrize("kq,vq,builder", [
    (("token", 128), ("token", 128), "make_pool_problem"),
    (("channel", 128), ("channel", 128), "make_pool_problem_flex"),
])
def test_soak_g8_heads_reach_the_builder_and_compile(soak, rec, kq, vq, builder):
    with contextlib.redirect_stdout(io.StringIO()):
        assert soak.soak_cell("x", kq, vq, 4, 2, 4096, 2, 3, False, 64, 8)
    (_, a, kw), = rec.names(builder)
    assert a[2:] == (64, 8) and "k_store_perm" in kw and a[1] == [4096, 4096]
    (_, _, ckw), = rec.names("compile_fold")
    assert (ckw["num_qo_heads"], ckw["num_kv_heads"]) == (64, 8)
    assert ckw["reduction"] == "kernel"


def test_soak_heads_flag_parses_and_refuses(soak):
    assert soak.parse_heads("64,8") == (64, 8)
    for bad in ("64", "64,7", "x,8"):
        with pytest.raises(ValueError):
            soak.parse_heads(bad)
    with pytest.raises(SystemExit):
        soak.main(["--heads", "64,7"])


# ------------------------------------------------------------------- --kernel-kwargs

def test_anchor_kernel_kwargs_reach_every_compile_and_read_back(anchor, rec, monkeypatch):
    monkeypatch.setattr(anchor, "EXTRA_KERNEL_KWARGS", {"arm_knob": "x"})
    cells = dict(anchor.build_cells())
    for name in ("i4_g8", "flex_i2_Kch128_Vch128_none", "flex_i4_Kch64_Vch64_g8",
                 "i2_default_s4", "flex_i4_Kch64_Vtok128"):
        rec.calls.clear()
        cells[name]()
        (_, _, ckw), = rec.names("compile_fold")
        assert ckw["arm_knob"] == "x", name


def test_anchor_kernel_kwargs_that_do_not_read_back_fail_the_cell(anchor, rec,
                                                                   monkeypatch):
    real = rec.compile_fold

    def drops_it(**kw):
        res = real(**kw)
        del kw["kernel_out"]["kernel"].arm_knob
        return res
    monkeypatch.setattr(anchor, "EXTRA_KERNEL_KWARGS", {"arm_knob": "x"})
    monkeypatch.setattr(anchor, "compile_fold", drops_it)
    with pytest.raises(AssertionError, match="no attribute 'arm_knob'"):
        dict(anchor.build_cells())["i4_g8"]()
    monkeypatch.setattr(anchor, "compile_fold", real)
    monkeypatch.setattr(anchor, "EXTRA_KERNEL_KWARGS", {"o_stages": 2})
    rec.calls.clear()
    with pytest.raises(AssertionError, match="_o_stages"):
        dict(anchor.build_cells())["i4_g8"]()     # stored as o_stages by the stub


def test_anchor_kernel_kwargs_refuse_a_knob_the_cell_sets(anchor, rec, monkeypatch):
    monkeypatch.setattr(anchor, "EXTRA_KERNEL_KWARGS", {"persistent_sm_count": 148})
    with pytest.raises(ValueError, match="collide"):
        dict(anchor.build_cells())["i2_persist_multi"]()
    monkeypatch.setattr(anchor, "EXTRA_KERNEL_KWARGS", {"num_qo_heads": 8})
    with pytest.raises(ValueError, match="collide"):
        dict(anchor.build_cells())["i2_default_s1"]()


def test_parse_kernel_kwargs(anchor):
    p = anchor.parse_kernel_kwargs
    assert p('{"a": 1, "q": ["token", 128]}') == {"a": 1, "q": ("token", 128)}
    for bad in ("[1, 2]", "not json", '"x"'):
        with pytest.raises(ValueError):
            p(bad)


def test_shard_checks_same_output_pairs_on_the_merged_save(tmp_path):
    shard = _load_path("_step0a_shard", SCRIPTS / "_anchor_shard.py")
    with_pairs = tmp_path / "a.py"
    with_pairs.write_text(
        "def check_same_output_cells(outs, whole_manifest):\n"
        "    assert whole_manifest\n"
        "    return [] if outs['x'] == outs['y'] else [('x', 'y')]\n")
    assert shard.same_output_failures(str(with_pairs), {"x": 1, "y": 1}) == []
    with contextlib.redirect_stdout(io.StringIO()):
        assert shard.same_output_failures(str(with_pairs), {"x": 1, "y": 2}) == [("x", "y")]
    without = tmp_path / "b.py"
    without.write_text("X = 1\n")
    assert shard.same_output_failures(str(without), {"x": 1}) == []


def test_shard_passes_kernel_kwargs_to_every_shard():
    src = (SCRIPTS / "_anchor_shard.py").read_text()
    assert 'cmd += ["--kernel-kwargs", args.kernel_kwargs]' in src


def test_oracle_kernel_kwargs(oracle, rec, monkeypatch):
    monkeypatch.setattr(oracle, "ref_channel_k_flex", lambda *a, **k: None)
    monkeypatch.setattr(oracle, "worst_rel", lambda out, ref: 1e-4)
    monkeypatch.setenv("FLEX_ONLY", "Kch128/Vch128")
    with contextlib.redirect_stdout(io.StringIO()):
        assert oracle.t3_kernel_cells(32, 8, {"arm_knob": "x"})
    assert all(c[2]["arm_knob"] == "x" for c in rec.names("compile_fold"))
    with pytest.raises(ValueError, match="collide"):
        oracle.t3_kernel_cells(32, 8, {"sdtype": "bf16"})
    real = rec.compile_fold

    def drops_it(**kw):
        res = real(**kw)
        del kw["kernel_out"]["kernel"].arm_knob
        return res
    monkeypatch.setattr(oracle, "compile_fold", drops_it)
    with pytest.raises(AssertionError, match="arm_knob"):
        oracle.t3_kernel_cells(32, 8, {"arm_knob": "x"})


def test_soak_kernel_kwargs(soak, rec):
    with contextlib.redirect_stdout(io.StringIO()):
        assert soak.soak_cell("x", ("token", 128), ("token", 128), 4, 2, 4096, 1, 2,
                              False, 64, 8, {"arm_knob": "y"})
    (_, a, kw), = rec.names("make_pool_problem")      # still the legacy builder
    (_, _, ckw), = rec.names("compile_fold")
    assert ckw["arm_knob"] == "y"


def test_drift_kernel_kwargs_need_runs(tmp_path):
    import subprocess
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "_drift_check_fold.py"), "--base", "b0.pt",
         "b1.pt", "--candidate", "r0.pt", "r1.pt", "--kernel-kwargs", "{}"],
        capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode != 0 and "--runs" in proc.stderr
