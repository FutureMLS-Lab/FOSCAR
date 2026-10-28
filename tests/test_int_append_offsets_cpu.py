"""CPU gates for IntAppend's 64-bit pool offsets.

IntAppend flattened (page, kv-head, row, word) into a pool word offset in signed 32-bit
arithmetic and widened it only when it formed the pointer. At int4 with HKV = 8, page 262,144
is word 2^31, which wraps to -2^31, so the append wrote 8 GiB before the pool (codex round 5,
2026-09-25; inherited from e4ab066). `serving_cache.code_word_offset` is now the formula the
kernel calls, and it widens the page before multiplying it.

Two things are checked here without a GPU. The formula is evaluated at the boundary with torch
int32 inputs, whose arithmetic wraps the way the device's does, so leaving the page unwidened
fails. The PTX rule of the H100 gate `scripts/test_int_append_wide_offsets.py` is run on PTX
transcribed from real compiles of the old and the new kernel, so the rule is shown to flag the
defect and to pass the fix. The rule is scoped to the form LLVM emits and refuses every other
shape (its module docstring); codex round 6's counterexamples, which the unscoped rule passed,
and the shapes the scope names are each checked to fail it.

Run: pytest tests/test_int_append_offsets_cpu.py
"""
import importlib.util
import pathlib
import sys
import types

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
HKV = 8


def _load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sc(monkeypatch):
    # Importing the package imports the DSL, so serving_cache is loaded on its own.
    for pkg in ("cutedsl_decode", "cutedsl_decode._sm90"):
        stub = types.ModuleType(pkg)
        stub.__path__ = []
        monkeypatch.setitem(sys.modules, pkg, stub)
    _load(monkeypatch, "cutedsl_decode.quant_layout", ROOT / "src/cutedsl_decode/quant_layout.py")
    return _load(monkeypatch, "cutedsl_decode._sm90.serving_cache",
                 ROOT / "src/cutedsl_decode/_sm90/serving_cache.py")


@pytest.fixture
def gate(monkeypatch):
    # The H100 gate imports nothing but the standard library at module level.
    return _load(monkeypatch, "test_int_append_wide_offsets",
                 ROOT / "scripts/test_int_append_wide_offsets.py")


def _i32(x):
    # One-element tensors, not 0-dim ones: torch leaves a 0-dim tensor out of type promotion
    # against a dimensioned tensor, so a 0-dim int64 page plus an int32 [128, 4] grid would
    # come back int32. The device's scalars promote Int64 + Int32 to Int64, and so do these.
    return torch.tensor([x], dtype=torch.int32)


def _wide(x):
    return x.to(torch.int64)


def _boundary(per):
    """The first page whose word offset needs more than 31 bits, at HKV = 8."""
    return (1 << 31) // (HKV * 128 * per // 4)


@pytest.mark.parametrize("order", ["chunk16", "row"])
@pytest.mark.parametrize("per", [32, 16])
def test_the_first_page_past_2_31_words_is_addressed_where_it_is(sc, order, per):
    """Every word of the boundary page's last kv-head lands in that block, once, at 2^31 and up."""
    page = _boundary(per)
    assert page * HKV * 128 * per // 4 == 1 << 31
    t = torch.arange(128, dtype=torch.int32).view(128, 1)
    w = torch.arange(per // 4, dtype=torch.int32).view(1, per // 4)
    for hk in (0, HKV - 1):
        off = sc.code_word_offset(order, per, HKV, _i32(page), _i32(hk), t, w, wide=_wide)
        assert off.dtype == torch.int64
        base = (page * HKV + hk) * 32 * per
        assert sorted(off.flatten().tolist()) == list(range(base, base + 32 * per))
    # word 0 of row 0 of kv-head 0 is exactly word 2^31
    first = sc.code_word_offset(order, per, HKV, _i32(page), _i32(0), _i32(0), _i32(0), wide=_wide)
    assert int(first) == 1 << 31


@pytest.mark.parametrize("per", [32, 16])
def test_the_same_inputs_wrap_negative_without_the_widening(sc, per):
    """The negative control: the inputs really are int32 and really sit on the boundary, so a
    kernel that multiplied the page in 32 bits would address 8 GiB before the pool."""
    page = _i32(_boundary(per))
    off = sc.code_word_offset("chunk16", per, HKV, page, _i32(0), _i32(0), _i32(0),
                              wide=lambda x: x)
    assert off.dtype == torch.int32 and int(off) == -(1 << 31)
    below = sc.code_word_offset("chunk16", per, HKV, page - 1, _i32(HKV - 1), _i32(127),
                                _i32(per // 4 - 1), wide=lambda x: x)
    assert int(below) == (1 << 31) - 1       # the last word below the boundary does not wrap


def test_int2_page_2_20_wraps_to_page_0_without_the_widening(sc):
    """At int2 and HKV = 8 a page is 4096 words, so page 2^20 is word 2^32: in 32 bits it
    lands exactly on page 0, a silent write inside the pool."""
    page = _i32(1 << 20)
    assert int(sc.code_word_offset("chunk16", 16, HKV, page, _i32(0), _i32(0), _i32(0),
                                   wide=lambda x: x)) == 0
    assert int(sc.code_word_offset("chunk16", 16, HKV, page, _i32(0), _i32(0), _i32(0),
                                   wide=_wide)) == 1 << 32


# ---------------------------------------------------------------------------------------------
# The PTX rule. The lines are transcribed from IntAppend compiled at int4, HKV = 8,
# NP = 262,145: codex's e4ab066 compile from fake compact tensors, which reads the slot row as
# one v2 load, and this tree's compile from real tensors, which reads it as two 32-bit loads,
# so both forms appear below (the H100 gate also checks this tree's v2 form). The slot
# row comes from parameter 2, the four owned channels each OR into the K pool (parameter 3)
# and the V pool (parameter 4), and lane 0 stores the scales (parameter 5). Only the integer
# path is kept, and the in-block arithmetic is shortened. `_entry` puts a body into a module
# the way LLVM emits one (an entry with its parameters and its register declarations), because
# the rule reads widths from the declarations and refuses code outside an entry.
# ---------------------------------------------------------------------------------------------

_DECLS = ".reg .pred %p<30>;\n.reg .b32 %r<200>;\n.reg .f32 %f<100>;\n.reg .b64 %rd<200>;\n"


def _module(*entries, decls=_DECLS):
    """A module holding one entry per (body, name), each with its own parameters."""
    out = ".version 8.8\n.target sm_90a\n.address_size 64\n\n"
    for body, name in entries:
        params = ",\n".join(f"\t.param .align 8 .b8 {name}_param_{i}[8]" for i in range(8))
        out += (f".visible .entry {name}(\n{params}\n)\n.reqntid 256, 1, 1\n{{\n"
                + decls + body.replace("kernel_x_param", f"{name}_param") + "}\n")
    return out


def _entry(body, name="kernel_x"):
    return _module((body, name))

_HEAD = """\
\tld.param.u64 \t%rd1, [kernel_x_param_5];
\tld.param.u64 \t%rd2, [kernel_x_param_4];
\tld.param.u64 \t%rd4, [kernel_x_param_3];
\tld.param.u64 \t%rd5, [kernel_x_param_6];
\tld.param.u64 \t%rd8, [kernel_x_param_2];
\tmov.u32 \t%r12, %tid.x;
\tmov.u32 \t%r13, %ctaid.x;
\tand.b32 \t%r14, %r12, 31;
\tshr.u32 \t%r1, %r12, 5;
"""

# e4ab066: the token index and the page are multiplied in 32 bits.
_SLOT_32 = """\
\tadd.s32 \t%r15, %r13, %r13;
\tmul.wide.s32 \t%rd9, %r15, 4;
\tadd.s64 \t%rd10, %rd8, %rd9;
\tld.global.v2.u32 \t{%r2, %r3}, [%rd10];
\tmul.lo.s32 \t%r20, %r3, 384;
\tmul.wide.s32 \t%rd14, %r20, 4;
\tadd.s64 \t%rd15, %rd5, %rd14;
\tshl.b32 \t%r26, %r2, 13;
\tshl.b32 \t%r27, %r1, 10;
\tor.b32 \t%r28, %r26, %r27;
"""


def _atoms_32():
    out = []
    for e in range(4):
        for pool, r in (("%rd4", 100 + 10 * e), ("%rd2", 105 + 10 * e)):
            out.append(f"\tld.global.u32 \t%r{r}, [%rd15+{384 * e}];\n"
                       f"\tadd.s32 \t%r{r + 1}, %r{r}, %r28;\n"
                       f"\tmul.wide.s32 \t%rd{r}, %r{r + 1}, 4;\n"
                       f"\tadd.s64 \t%rd{r + 1}, {pool}, %rd{r};\n"
                       f"\tatom.global.or.b32 \t%r{r + 2}, [%rd{r + 1}], %r{r + 3};\n")
    return "".join(out)


_SCALE_32 = """\
\tsetp.ne.s32 \t%p1, %r14, 0;
\t@%p1 bra \t$L__BB0_2;
\tshl.b32 \t%r90, %r2, 11;
\tshl.b32 \t%r91, %r1, 8;
\tor.b32 \t%r92, %r90, %r91;
\tshl.b32 \t%r93, %r3, 2;
\tadd.s32 \t%r94, %r92, %r93;
\tmul.wide.s32 \t%rd33, %r94, 4;
\tadd.s64 \t%rd34, %rd1, %rd33;
\tst.global.v4.f32 \t[%rd34], {%f78, %f1, %f86, %f2};
$L__BB0_2:
\tret;
"""

# this tree: the token index and the page are widened before they are multiplied, while the
# intra-tile position t (byte 4 of the slot row) is still multiplied in 32 bits, as it may be.
_SLOT_64 = """\
\tmul.wide.u32 \t%rd10, %r13, 8;
\tadd.s64 \t%rd11, %rd8, %rd10;
\tld.global.u32 \t%r2, [%rd11];
\tld.global.u32 \t%r3, [%rd11+4];
\tmul.lo.s32 \t%r20, %r3, 384;
\tmul.wide.s32 \t%rd14, %r20, 4;
\tadd.s64 \t%rd15, %rd5, %rd14;
\tcvt.s64.s32 \t%rd3, %r2;
\tmul.wide.s32 \t%rd22, %r2, 8192;
\tshl.b32 \t%r34, %r1, 10;
\tcvt.u64.u32 \t%rd23, %r34;
\tor.b64 \t%rd24, %rd22, %rd23;
"""


def _atoms_64():
    out = []
    for e in range(4):
        for pool, r in (("%rd4", 100 + 10 * e), ("%rd2", 105 + 10 * e)):
            out.append(f"\tld.global.u32 \t%r{r}, [%rd15+{384 * e}];\n"
                       f"\tshl.b32 \t%r{r + 1}, %r{r}, 2;\n"
                       f"\tcvt.s64.s32 \t%rd{r}, %r{r + 1};\n"
                       f"\tadd.s64 \t%rd{r + 1}, %rd24, %rd{r};\n"
                       f"\tshl.b64 \t%rd{r + 2}, %rd{r + 1}, 2;\n"
                       f"\tadd.s64 \t%rd{r + 3}, {pool}, %rd{r + 2};\n"
                       f"\tatom.global.or.b32 \t%r{r + 2}, [%rd{r + 3}], %r{r + 3};\n")
    return "".join(out)


_SCALE_64 = """\
\tsetp.ne.s32 \t%p25, %r14, 0;
\t@%p25 bra \t$L__BB0_2;
\tshl.b64 \t%rd58, %rd3, 11;
\tshl.b32 \t%r156, %r1, 8;
\tcvt.u64.u32 \t%rd59, %r156;
\tor.b64 \t%rd60, %rd58, %rd59;
\tshl.b32 \t%r157, %r3, 2;
\tcvt.s64.s32 \t%rd61, %r157;
\tadd.s64 \t%rd62, %rd60, %rd61;
\tshl.b64 \t%rd63, %rd62, 2;
\tadd.s64 \t%rd64, %rd1, %rd63;
\tst.global.v4.f32 \t[%rd64], {%f78, %f1, %f86, %f2};
$L__BB0_2:
\tret;
"""


def test_the_rule_passes_the_widened_kernel(gate):
    rep = gate.index_width_report(_entry(_HEAD + _SLOT_64 + _atoms_64() + _SCALE_64))
    assert rep["ok"], rep
    assert rep["violations"] == []
    assert len(rep["atomics"]) == gate.CODE_ATOMICS and all(a for _, a in rep["atomics"])
    assert rep["stores"] and all(s for _, s in rep["stores"])


def test_the_rule_flags_the_e4ab066_kernel(gate):
    rep = gate.index_width_report(_entry(_HEAD + _SLOT_32 + _atoms_32() + _SCALE_32))
    assert not rep["ok"]
    flagged = " | ".join(rep["violations"])
    assert "add.s32 %r15, %r13, %r13" in flagged          # the token index, into the slot row
    assert "shl.b32 %r26, %r2, 13" in flagged             # the page, into the code pools
    assert "shl.b32 %r90, %r2, 11" in flagged             # the page, into the scale pool
    # the sources and sinks were all found, so the failure is the arithmetic alone
    assert rep["page_sources"] and rep["token_sources"]
    assert len(rep["atomics"]) == gate.CODE_ATOMICS and all(a for _, a in rep["atomics"])


def test_a_32_bit_scale_offset_alone_is_flagged(gate):
    rep = gate.index_width_report(_entry(_HEAD + _SLOT_64 + _atoms_64() + _SCALE_32))
    assert not rep["ok"]
    assert [v.split(": ", 1)[1] for v in rep["violations"]] == [
        "shl.b32 %r90, %r2, 11", "or.b32 %r92, %r90, %r91", "add.s32 %r94, %r92, %r93"]


def test_truncating_the_widened_page_is_flagged(gate):
    body = _SLOT_64.replace("\tmul.wide.s32 \t%rd22, %r2, 8192;\n",
                            "\tmul.wide.s32 \t%rd21, %r2, 8192;\n"
                            "\tcvt.u32.u64 \t%r40, %rd21;\n"
                            "\tmul.wide.s32 \t%rd22, %r40, 1;\n")
    rep = gate.index_width_report(_entry(_HEAD + body + _atoms_64() + _SCALE_64))
    assert not rep["ok"] and any("cvt.u32.u64 %r40, %rd21" in v for v in rep["violations"])


def test_the_rule_cannot_pass_by_losing_the_page(gate):
    """With the slot row read from another parameter the page is never found, and a check
    that found no page has checked nothing."""
    ptx = _entry(_HEAD + _SLOT_64 + _atoms_64() + _SCALE_64).replace("_param_2]", "_param_1]")
    rep = gate.index_width_report(ptx)
    assert rep["violations"] == [] and not rep["page_sources"] and not rep["ok"]


def test_the_rule_demands_every_code_atomic(gate):
    atoms = _atoms_64()
    last = atoms.rfind("\tatom.global")
    rep = gate.index_width_report(_entry(_HEAD + _SLOT_64 + atoms[:last] + _SCALE_64))
    assert len(rep["atomics"]) == gate.CODE_ATOMICS - 1 and not rep["ok"]


# ---------------------------------------------------------------------------------------------
# The scope. Codex round 6 (2026-09-25) made the unscoped rule pass four shapes; each is below
# and must now fail. The rule is scoped to the form LLVM emits and refuses the rest. Every shape
# the gate's module docstring names as refused or as a violation is checked below to fail by
# name, each sink the verdict requires is checked to fail it when missing, and the LLVM
# variants the scope accepts are checked to pass.
# ---------------------------------------------------------------------------------------------

_BASE = _HEAD + _SLOT_64 + _atoms_64() + _SCALE_64
_PAGE_MUL = "\tmul.wide.s32 \t%rd22, %r2, 8192;\n"
_PAGE_OR = "\tor.b64 \t%rd24, %rd22, %rd23;\n"
_PAGE_LOAD = "\tld.global.u32 \t%r2, [%rd11];\n"


def _swap(old, new, body=_BASE):
    assert body.count(old) == 1, old
    return body.replace(old, new)


def _names(rep, *words):
    """Whether one refusal names every word."""
    return any(all(w in x for w in words) for x in rep["refusals"])


def test_codex_r6_a_narrowing_cvt_into_its_own_64_bit_register_fails(gate):
    """Codex's first case: the page product truncated to 32 bits in place. At page 2^20 of an
    int2 pool the word offset 2^32 becomes 0, and the old rule passed it because the register
    written is named %rd."""
    rep = gate.index_width_report(_entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tcvt.u32.u64 \t%rd22, %rd22;\n")))
    assert not rep["ok"]
    assert any("cvt.u32.u64 %rd22, %rd22" in v for v in rep["violations"])
    assert _names(rep, "%rd22", "2 definitions")


@pytest.mark.parametrize("cvt", ["cvt.u32.u64", "cvt.s64.s32"])
def test_a_narrowing_cvt_fails_whatever_register_it_writes(gate, cvt):
    """The same truncation into a fresh 64-bit register, which has one definition, so the
    narrowing conversion alone fails it. `cvt.s64.s32` from a 64-bit register is LLVM's
    sign-extend-in-register: it keeps the low 32 bits too."""
    body = _swap(_PAGE_MUL, _PAGE_MUL + f"\t{cvt} \t%rd70, %rd22;\n")
    body = _swap(_PAGE_OR, "\tor.b64 \t%rd24, %rd70, %rd23;\n", body)
    rep = gate.index_width_report(_entry(body))
    assert not rep["ok"] and rep["refusals"] == []
    assert [v.split(": ", 1)[1] for v in rep["violations"]] == [f"{cvt} %rd70, %rd22"]


def test_codex_r6_a_64_bit_slot_load_and_an_unexecuted_assignment_fail(gate):
    """Codex's second case: the page read again by a 64-bit load of the slot row and shifted in
    32 bits, while an always-false predicated copy of the right product also writes %rd22.
    The old rule did not see the 64-bit load and took the unexecuted assignment as the page's
    path to every atomic."""
    wrong = ("\tld.global.u64 \t%rd70, [%rd11];\n\tcvt.u32.u64 \t%r190, %rd70;\n"
             "\tshl.b32 \t%r191, %r190, 13;\n\tcvt.s64.s32 \t%rd22, %r191;\n"
             "\tsetp.ne.u32 \t%p1, 0, 0;\n\t@%p1 mul.wide.s32 \t%rd22, %r2, 8192;\n")
    rep = gate.index_width_report(_entry(_swap(_PAGE_MUL, wrong)))
    assert not rep["ok"]
    assert _names(rep, "ld.global.u64 %rd70, [%rd11]", "not 32 bits wide")
    assert _names(rep, "@%p1 mul.wide.s32 %rd22, %r2, 8192", "predicated")
    assert _names(rep, "%rd22", "2 definitions")
    flagged = " | ".join(rep["violations"])
    assert "cvt.u32.u64 %r190, %rd70" in flagged and "shl.b32 %r191, %r190, 13" in flagged


@pytest.mark.parametrize("assign", [
    "\tmov.u64 \t%rd22, 0;\n\tsetp.ne.u32 \t%p1, 0, 0;\n\t@%p1 mul.wide.s32 \t%rd22, %r2, 8192;\n",
    "\tsetp.ne.u32 \t%p1, 0, 0;\n\t@%p1 mul.wide.s32 \t%rd22, %r2, 8192;\n",
], ids=["with-a-second-definition", "alone"])
def test_codex_r6_a_predicated_page_dependency_fails(gate, assign):
    """The unexecuted assignment by itself: a predicated definition on the dataflow is refused
    whether or not the register has another definition."""
    rep = gate.index_width_report(_entry(_swap(_PAGE_MUL, assign)))
    assert not rep["ok"]
    assert _names(rep, "@%p1 mul.wide.s32 %rd22, %r2, 8192", "a predicated instruction")
    assert _names(rep, "%rd22", "predicated") or _names(rep, "%rd22", "2 definitions")


def test_codex_r6_a_source_and_a_sink_in_two_entries_fail(gate):
    """Codex's third case: the page is loaded in one entry and the atomics are in another. Each
    entry is its own scope, so neither has both."""
    source = _HEAD + _SLOT_64 + "\tret;\n"
    sink = (_swap("\tmov.u32 \t%r13, %ctaid.x;\n", "\tmov.u32 \t%r13, 0;\n", _HEAD)
            + _swap(_PAGE_LOAD, "\tmov.u32 \t%r2, 0;\n", _SLOT_64) + _atoms_64() + _SCALE_64)
    rep = gate.index_width_report(_module((source, "source"), (sink, "sink")))
    assert not rep["ok"] and [e["name"] for e in rep["entries"]] == ["source", "sink"]
    src, snk = rep["entries"]
    assert src["page_sources"] and src["token_sources"] and src["atomics"] == []
    assert not snk["page_sources"] and len(snk["atomics"]) == gate.CODE_ATOMICS
    assert not any(a for _, a in snk["atomics"])
    # codex's own transcription, with no declarations and another entry's parameters
    literal = (".visible .entry source() {\n" + source + "\n}\n"
               ".visible .entry sink() {\n" + sink + "\n}")
    assert not gate.index_width_report(literal)["ok"]


def test_codex_r6_the_controls_that_already_failed_still_fail_by_name(gate):
    """Codex's two controls, which the old rule failed only because it then found no page: a
    64-bit slot load unpacked by a move, and a slot pointer formed by `or`."""
    wide = gate.index_width_report(_entry(_swap(
        _PAGE_LOAD, "\tld.global.u64 \t%rd70, [%rd11];\n\tmov.b64 \t{%r2, %r4}, %rd70;\n")))
    assert not wide["ok"] and _names(wide, "not 32 bits wide")
    assert _names(wide, "mov.b64 {%r2, %r4}, %rd70", "vector operand")
    orptr = gate.index_width_report(_entry(_swap(
        "\tadd.s64 \t%rd11, %rd8, %rd10;\n", "\tor.b64 \t%rd11, %rd8, %rd10;\n")))
    assert not orptr["ok"] and _names(orptr, "or.b64 %rd11, %rd8, %rd10", "slot-row pointer")
    assert _names(orptr, "cannot place")


@pytest.mark.parametrize("load", [
    "\tld.global.u32 \t%r190, [%rd11];\n",
    "\tmov.b64 \t%rd71, %rd11;\n\tld.global.u32 \t%r190, [%rd71];\n",
    "\tld.global.u32 \t%r190, [%rd11+8];\n",
], ids=["same-pointer", "copied-pointer", "next-row"])
def test_a_second_32_bit_read_of_the_page_is_a_source_of_its_own(gate, load):
    """Loads are followed by what they read, not by the register they write: the page read a
    second time and shifted in 32 bits is the defect, found as a violation."""
    body = _swap(_PAGE_MUL, load + "\tshl.b32 \t%r191, %r190, 13;\n\tcvt.s64.s32 \t%rd22, %r191;\n")
    rep = gate.index_width_report(_entry(body))
    assert not rep["ok"] and rep["refusals"] == [] and len(rep["page_sources"]) == 2
    assert [v.split(": ", 1)[1] for v in rep["violations"]] == ["shl.b32 %r191, %r190, 13"]


_OUTSIDE = [
    ("and-b64", lambda: _entry(_swap(_PAGE_MUL, "\tmul.wide.s32 \t%rd21, %r2, 8192;\n"
                                     "\tand.b64 \t%rd22, %rd21, 4294967295;\n")),
     "refusals", "does not classify"),
    ("shr-u64", lambda: _entry(_swap(_PAGE_MUL, "\tmul.wide.s32 \t%rd21, %r2, 8192;\n"
                                     "\tshr.u64 \t%rd22, %rd21, 0;\n")),
     "refusals", "does not classify"),
    ("selp-b64", lambda: _entry(_swap(_PAGE_MUL, "\tmul.wide.s32 \t%rd21, %r2, 8192;\n"
                                      "\tselp.b64 \t%rd22, %rd21, 0, %p25;\n")),
     "refusals", "does not classify"),
    ("setp", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tsetp.gt.s32 \t%p3, %r2, 524287;\n")),
     "refusals", "a comparison or a predicate"),
    ("page-stored", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tst.global.u32 \t[%rd4], %r2;\n")),
     "refusals", "written to memory as data"),
    ("shfl", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tshfl.sync.idx.b32 \t%r41, %r2, 0, 31, -1;\n")),
     "violations", "shfl.sync.idx.b32 %r41, %r2, 0, 31, -1"),
    ("shift-amount", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tshl.b64 \t%rd70, %rd23, %r2;\n")),
     "refusals", "shift amount"),
    ("width-mismatch", lambda: _entry(_swap(_PAGE_OR, "\tor.b64 \t%rd24, %rd22, %r34;\n")),
     "refusals", "declared width"),
    ("nested-block", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\t{\n\t.reg .b64 %rd22;\n"
                                          "\tmov.b64 %rd22, 0;\n\t}\n")),
     "refusals", "nested { } block"),
    ("undeclared", lambda: _entry(_swap(_PAGE_MUL, "\tmul.wide.s32 \t%q22, %r2, 8192;\n"
                                        "\tmov.b64 \t%rd22, %q22;\n")),
     "refusals", "neither a register"),
    ("call", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tcall.uni \tfoo, (%rd22);\n")),
     "refusals", "a call"),
    ("slot-byte-2", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tld.global.u32 \t%r190, [%rd11+2];\n")),
     "refusals", "reads byte 2 of a slot row"),
    ("slot-offset-unknown", lambda: _entry(_swap("\tmul.wide.u32 \t%rd10, %r13, 8;\n",
                                                 "\tmul.wide.u32 \t%rd10, %r13, 4;\n")),
     "refusals", "cannot place"),
    ("slot-param-offset", lambda: _entry(_swap("\tld.param.u64 \t%rd8, [kernel_x_param_2];\n",
                                               "\tld.param.u64 \t%rd8, [kernel_x_param_2+4];\n")),
     "refusals", "parameter 2 (the slot rows) is read other than"),
    ("predicated-atomic", lambda: _entry(_swap("\tatom.global.or.b32 \t%r102, [%rd103]",
                                               "\t@%p25 atom.global.or.b32 \t%r102, [%rd103]")),
     "refusals", "a predicated memory access"),
    ("token-stored", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tst.global.u32 \t[%rd4], %ctaid.x;\n")),
     "refusals", "written to memory as data"),
    ("page-address-in-a-copy", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tadd.s64 \t%rd70, %rd4, %rd22;\n"
                                                    "\tcp.async.bulk.global.shared::cta [%rd70], [%r1], 16;\n")),
     "refusals", "does not classify"),
    ("cluster-index", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tmov.u32 \t%r41, %cluster_ctaid.x;\n")),
     "refusals", "another CTA or cluster index"),
    ("func-body", lambda: _entry(_BASE) + ".func foo()\n{\n\tret;\n}\n", "refusals", "a .func body"),
    ("statement-outside-entry", lambda: _entry(_BASE) + "\tret;\n",
     "refusals", "ret; -- a statement outside an .entry body"),
    ("predicated-branch", lambda: _entry(_swap("\tsetp.ne.s32 \t%p25, %r14, 0;\n",
                                               "\tsetp.ne.s32 \t%p25, %r2, 0;\n")),
     "refusals", "@%p25 bra $L__BB0_2 -- a predicated instruction on the dataflow"),
    ("slot-param-named", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tmov.u64 \t%rd70, kernel_x_param_2;\n")),
     "refusals", "parameter 2 (the slot rows) is named other than by ld.param"),
    ("slot-pointer-stored", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tst.global.u64 \t[%rd4], %rd8;\n")),
     "refusals", "st.global.u64 [%rd4], %rd8 -- the page, the token or the slot-row pointer is "
                 "written to memory as data"),
    ("slot-pointer-compared", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tsetp.eq.s64 \t%p3, %rd8, 0;\n")),
     "refusals", "setp.eq.s64 %p3, %rd8, 0 -- the slot-row pointer is used other than as a load "
                 "address"),
    ("slot-row-written", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tst.global.u32 \t[%rd11+4], %r1;\n")),
     "refusals", "st.global.u32 [%rd11+4], %r1 -- writes the slot rows"),
    ("slot-field-into-64-bit", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tld.global.u32 \t%rd70, [%rd11];\n")),
     "refusals", "loads a slot-row field into %rd70, which is not declared 32 bits wide"),
    ("page-exchanged", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tatom.global.exch.b32 \t%r41, [%rd4], %r2;\n")),
     "refusals", "atom.global.exch.b32 %r41, [%rd4], %r2 -- the page, the token or the slot-row "
                 "pointer is written to memory as data"),
    ("and-b32", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tand.b32 \t%r41, %r2, 255;\n")),
     "violations", "and.b32 %r41, %r2, 255"),
    ("selp-b32", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tselp.b32 \t%r41, %r2, 0, %p25;\n")),
     "violations", "selp.b32 %r41, %r2, 0, %p25"),
    ("mov-other-width", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tmov.b64 \t%rd70, %r2;\n")),
     "refusals", "mov.b64 %rd70, %r2 -- a move that is not between registers"),
    ("pack", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tmov.b64 \t%rd70, {%r2, %r3};\n")),
     "refusals", "mov.b64 %rd70, {%r2, %r3} -- a vector operand (a pack or an unpack)"),
    ("float-cvt", lambda: _entry(_swap(_PAGE_MUL, _PAGE_MUL + "\tcvt.rn.f32.s32 \t%f50, %r2;\n")),
     "refusals", "cvt.rn.f32.s32 %f50, %r2 -- a conversion that is not an integer widening"),
    ("no-entry", lambda: _BASE, "refusals", "no .entry body"),
    ("address-size-32", lambda: _entry(_BASE).replace(".address_size 64", ".address_size 32"),
     "refusals", ".address_size 64"),
]


@pytest.mark.parametrize("ptx,where,word", [c[1:] for c in _OUTSIDE], ids=[c[0] for c in _OUTSIDE])
def test_a_shape_outside_the_scope_fails_by_name(gate, ptx, where, word):
    rep = gate.index_width_report(ptx())
    assert not rep["ok"]
    assert any(word in x for x in rep[where]), rep[where]


@pytest.mark.parametrize("variant", [
    ("\tmul.wide.u32 \t%rd10, %r13, 8;\n\tadd.s64 \t%rd11, %rd8, %rd10;\n",
     "\tmad.wide.u32 \t%rd11, %r13, 8, %rd8;\n"),
    ("\tld.param.u64 \t%rd8, [kernel_x_param_2];\n",
     "\tld.param.u64 \t%rd7, [kernel_x_param_2];\n\tcvta.to.global.u64 \t%rd8, %rd7;\n"),
    ("\tld.global.u32 \t%r2, [%rd11];\n\tld.global.u32 \t%r3, [%rd11+4];\n",
     "\tld.global.v2.u32 \t{%r2, %r3}, [%rd11];\n"),
], ids=["mad-wide-slot-address", "cvta-slot-pointer", "v2-slot-load"])
def test_the_llvm_variants_inside_the_scope_pass(gate, variant):
    rep = gate.index_width_report(_entry(_swap(*variant)))
    assert rep["ok"], rep


@pytest.mark.parametrize("old,new,sink", [
    ("\tshl.b64 \t%rd58, %rd3, 11;\n", "\tshl.b64 \t%rd58, %rd23, 11;\n", "stores"),
    ("\tmul.wide.u32 \t%rd10, %r13, 8;\n", "\tmul.wide.u32 \t%rd10, %r1, 8;\n", "token_addresses"),
], ids=["scale-store-off-the-page", "token-reaching-no-address"])
def test_a_sink_that_loses_its_index_fails(gate, old, new, sink):
    """Both indices are still found and nothing is computed in 32 bits, but the scale store no
    longer depends on the page, or the token reaches no address, so the check has not seen the
    offset it exists to check."""
    rep = gate.index_width_report(_entry(_swap(old, new)))
    assert not rep["ok"] and rep["violations"] == [] and rep["refusals"] == []
    assert rep["page_sources"] and rep["token_sources"]
    assert (not all(s for _, s in rep["stores"])) if sink == "stores" else rep["token_addresses"] == 0


# LLVM declares each register range of an entry once (the saved compiles declare %p, %rs, %r,
# %f and %rd, one range each), so a register declared twice is refused whether the repeat is
# identical, changes the width or the count, names one register of a range, or is a range whose
# prefix extends another's by digits (%r1<5> declares %r10 to %r14, which %r<200> declares).
@pytest.mark.parametrize("decl,item,first", [
    (".reg .b64 %rd<200>;\n", "%rd<200>", "%rd<200>"),
    (".reg .b32 %rd<200>;\n", "%rd<200>", "%rd<200>"),
    (".reg .b64 %rd<10>;\n", "%rd<10>", "%rd<200>"),
    (".reg .b64 %rd22;\n", "%rd22", "%rd<200>"),
    (".reg .b32 %r1<5>;\n", "%r1<5>", "%r<200>"),
    (".reg .b64 %q, %q;\n", "%q", "%q"),
], ids=["identical-range", "other-width", "other-count", "one-register-of-a-range",
        "overlapping-prefix", "one-name-twice"])
def test_a_register_declared_more_than_once_is_refused(gate, decl, item, first):
    rep = gate.index_width_report(_module((_BASE, "kernel_x"), decls=_DECLS + decl))
    assert not rep["ok"] and rep["violations"] == []
    assert _names(rep, f"the .reg item {item} declares a register that the .reg item {first} at",
                  "a register declared more than once is refused")


def test_one_range_per_register_class_passes(gate):
    """The declarations of the saved INT2 compile's classes, %rs included, beside %r: a prefix
    extended by a letter declares no register of the shorter one."""
    decls = (".reg .pred %p<30>;\n.reg .b16 %rs<31>;\n.reg .b32 %r<200>;\n.reg .f32 %f<100>;\n"
             ".reg .b64 %rd<200>;\n")
    rep = gate.index_width_report(_module((_BASE, "kernel_x"), decls=decls))
    assert rep["ok"] and rep["refusals"] == [], rep


def test_the_overlap_of_two_declarations_is_the_overlap_of_the_registers_they_name(gate):
    """`_same_register` decides overlap without listing a range's registers; here it is compared
    with the listed registers for every pair of small items."""
    def regs(item):
        name, count = item
        return {name} if count is None else {f"{name}{i}" for i in range(count)}
    items = ([(p, n) for p in ("%r", "%r1", "%r12", "%r0", "%rd", "%rd1")
              for n in (0, 1, 2, 10, 11, 13, 121, 130)]
             + [(s, None) for s in ("%r", "%r1", "%r01", "%r10", "%r129", "%rd0", "%q")])
    for a in items:
        for b in items:
            assert gate._same_register(a, b) == bool(regs(a) & regs(b)), (a, b)
