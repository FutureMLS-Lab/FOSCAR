#!/usr/bin/env python3
"""CPU gates for what the PTX fingerprint normalizes away.

The fingerprint compares EMITTED CODE. The mangled entry symbol carries the kernel
class's name, so a pure rename changes the PTX text without changing an instruction, and
`normalize(collapse_entry=True)` exists to take that out of the comparison.

It did not, quite. The pattern ended at `_cutedsl`, which appears EARLY inside the
mangling -- `kernel_cutlass___call___cutedsl_decode_sm100int_fold_decodeIntDecodeSm100_
object_at__` -- so it collapsed a prefix and left the class name in the normalized text.
Renaming FoldDecodeSm100 to IntDecodeSm100 on 2026-08-28 therefore moved all 64 SM100
fingerprints and all 49 SM90 ones while every structural count stayed exactly equal on
both sides, which is the signature of a symbol change rather than a code change. The
anchor agreed: 115 of 115 cells bit-identical.

These gates need no GPU, which is the point -- the defect was in a regex, and finding it
cost two hardware runs.

Run: pytest tests/test_ptx_normalization_cpu.py
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def _core():
    spec = importlib.util.spec_from_file_location(
        "ptx_core", _SCRIPTS / "_ptx_fingerprint_core.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ptx(cls: str) -> str:
    return (
        f".visible .entry kernel_cutlass___call___cutedsl_decode_sm100int_fold_decode"
        f"{cls}_object_at___FakeTensorInt32_div1_1(\n"
        "\t.param .u64 p0\n"
        ")\n"
        "{\n"
        "\tmov.u32 %r1, 5;\n"
        "\tadd.s32 %r2, %r1, 1;\n"
        "\tret;\n"
        "}\n"
    )


def test_a_class_rename_is_invisible_after_normalization():
    """The whole reason `collapse_entry` exists."""
    n = _core().normalize
    assert n(_ptx("FoldDecodeSm100")) == n(_ptx("IntDecodeSm100"))
    assert n(_ptx("IntPingPongDecode")) == n(_ptx("IntPingPongDecodeSm90"))


def test_the_class_name_is_actually_gone():
    """Not merely equal to each other -- absent. Two texts can agree while both still
    carry a name, which is what the old pattern did for two spellings of the same
    class."""
    out = _core().normalize(_ptx("IntDecodeSm100"))
    assert "IntDecodeSm100" not in out
    assert "int_fold_decode" not in out
    assert "kernel_cutlass_ENTRY_cutedsl" in out


def test_normalization_still_keeps_the_instructions():
    """A collapse that ate the body would make every kernel look alike and every
    comparison pass."""
    out = _core().normalize(_ptx("IntDecodeSm100"))
    assert "mov.u32" in out and "add.s32" in out and "ret;" in out


def _ptx_with_helper(cls: str, helper: str) -> str:
    return (
        f".visible .entry kernel_cutlass___call___cutedsl_x{cls}_object_at___0(\n"
        "\t.param .u64 p0\n)\n{\n"
        f"\tcall.uni (retval), kernel_cutlass_{helper}, (%r1);\n"
        "\tret;\n}\n"
    )


def test_only_the_entry_symbol_is_collapsed_not_every_similar_name():
    """The second wrong spelling, and the more dangerous one.

    A pattern greedy over `kernel_cutlass_[A-Za-z0-9_]+` erased EVERY identifier with
    that prefix, so a changed `.func` or call target would vanish and two genuinely
    different PTX texts could fingerprint identically. The names to collapse are read
    from the `.entry` lines, so a symbol that merely shares the prefix is compared.
    (codex)
    """
    n = _core().normalize
    assert n(_ptx_with_helper("FoldDecodeSm100", "helper_A")) == \
        n(_ptx_with_helper("IntDecodeSm100", "helper_A")), \
        "a class rename must still be invisible"
    assert n(_ptx_with_helper("IntDecodeSm100", "helper_A")) != \
        n(_ptx_with_helper("IntDecodeSm100", "helper_B")), \
        "a changed call target must NOT be erased"
    assert "helper_A" in n(_ptx_with_helper("IntDecodeSm100", "helper_A"))


def test_a_comment_cannot_nominate_a_name_for_erasure():
    """Harvesting the entry names from the RAW text let a comment do it.

    `// .visible .entry kernel_cutlass_helper_A` made `helper_A` collapsible, so
    renaming both that comment and a real call target produced identical fingerprints.
    Comments are dropped BEFORE the names are collected, and the directive is anchored
    to the start of a line. (codex)
    """
    def ptx(helper):
        return (
            ".visible .entry kernel_cutlass___call___cutedsl_xIntDecodeSm100_object_at___0(\n"
            "\t.param .u64 p0\n)\n{\n"
            f"\t// .visible .entry kernel_cutlass_{helper}\n"
            f"\tcall.uni (retval), kernel_cutlass_{helper}, (%r1);\n"
            "\tret;\n}\n"
        )
    n = _core().normalize
    assert n(ptx("helper_A")) != n(ptx("helper_B"))
    assert "helper_A" in n(ptx("helper_A"))


def test_a_longer_symbol_containing_the_entry_name_is_not_collapsed():
    """`str.replace` was substring-based, so an unrelated `<entry>_helper` became
    `<placeholder>_helper` and two different symbols could normalize alike. PTX
    identifiers are [A-Za-z0-9_$], and the boundaries are stated in those terms.
    (codex)
    """
    entry = "kernel_cutlass_E_cutedsl_0"
    def ptx(suffix):
        return (f".visible .entry {entry}(\n\t.param .u64 p0\n)\n{{\n"
                f"\tcall.uni (retval), {entry}{suffix}, (%r1);\n"
                "\tret;\n}\n")
    n = _core().normalize
    assert n(ptx("_helperA")) != n(ptx("_helperB")), (
        "a longer symbol that merely starts with the entry name must stay compared")
    assert "_helperA" in n(ptx("_helperA"))
    # and the entry symbol ITSELF is still collapsed where it stands alone
    assert entry not in n(ptx("_helperA")).split("call.uni")[0]


def test_a_real_instruction_change_still_moves_the_fingerprint():
    """The check that keeps the one above from being a way to pass everything."""
    core = _core()
    a = core.normalize(_ptx("IntDecodeSm100"))
    b = core.normalize(_ptx("IntDecodeSm100").replace("add.s32 %r2, %r1, 1;",
                                                      "sub.s32 %r2, %r1, 1;"))
    assert a != b, "an instruction change must survive normalization"


# ---------------------------------------------------------------------------
# The parameter names PTX derives from the entry name (Plan A, Step 1, item 1(c)).
#
# Every fixture above names its parameter `p0`, so none of them could see that the DSL's
# real PTX names an entry's parameters `<entry>_param_<N>`. `_` is an identifier
# character, so the whole-symbol rule left each of those carrying the RAW entry name, and
# two PTX texts that differed only in the entry name still normalized differently.
# ---------------------------------------------------------------------------

def _pp_entry(cls: str) -> str:
    return (f"kernel_cutlass___call___cutedsl_decode_sm90int_decode_pingpong"
            f"{cls}_object_at___FakeTensor_0")


def _ptx_with_params(cls: str, extra: str = "") -> str:
    entry = _pp_entry(cls)
    return (
        f".visible .entry {entry}(\n"
        f"\t.param .u64 {entry}_param_0,\n"
        f"\t.param .u32 {entry}_param_1\n"
        ")\n{\n"
        f"\tld.param.u64 %rd1, [{entry}_param_0];\n"
        f"\tld.param.u32 %r1, [{entry}_param_1];\n"
        f"{extra}"
        "\tret;\n}\n"
    )


def test_entry_param_names_collapse_with_the_entry():
    """`<entry>_param_0` is collapsed together with `<entry>`, so a pure entry rename is
    invisible even where the parameters are declared and read."""
    n = _core().normalize
    a, b = n(_ptx_with_params("IntPingPongDecodeSm90")), n(_ptx_with_params("Renamed"))
    assert a == b
    assert "IntPingPongDecodeSm90" not in a
    assert "kernel_cutlass_ENTRY_cutedsl_param_0" in a
    assert "kernel_cutlass_ENTRY_cutedsl_param_1" in a


def test_the_parameter_index_survives_the_collapse():
    """Which parameter an instruction reads is code, so the index is not erased."""
    n = _core().normalize
    entry = _pp_entry("IntPingPongDecodeSm90")
    swapped = _ptx_with_params("IntPingPongDecodeSm90").replace(
        f"[{entry}_param_0]", f"[{entry}_param_1]")
    assert n(swapped) != n(_ptx_with_params("IntPingPongDecodeSm90"))


def test_entry_helper_and_param_lookalikes_are_not_collapsed():
    """Only `_param_<digits>` at a symbol boundary is derived from the entry.
    `<entry>_helper` and `<entry>_param_0_x` are other symbols and are compared. A lookalike
    that itself ends in `_<digits>` has the DSL's entry mangling and is refused instead
    (`test_an_undeclared_name_of_the_dsl_mangling_is_refused`)."""
    n = _core().normalize
    entry = _pp_entry("IntPingPongDecodeSm90")
    for suffix in ("_helper", "_param_0_x", "_param_"):
        call = f"\tcall.uni (retval), {entry}{suffix}, (%r1);\n"
        out = n(_ptx_with_params("IntPingPongDecodeSm90", call))
        assert f"{entry}{suffix}" in out, f"{suffix} must keep its raw name"
        other = n(_ptx_with_params("IntPingPongDecodeSm90",
                                   call.replace(suffix, suffix + "Z")))
        assert out != other, f"a changed {suffix} symbol must stay visible"


def _sass(function: str, body: str = "        /*0000*/                   MOV R1, c[0x0][0x28] ;\n") -> str:
    return (
        '\t.headerflags\t@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM90"\n'
        f"//--------------------- .text.{function} --------------------------\n"
        f'\t.section\t.text.{function},"ax",@progbits\n'
        '\t.sectioninfo\t@"SHI_REGISTERS=104"\n'
        f"        .global         {function}\n"
        f"        .type           {function},@function\n"
        f"        .size           {function},(.L_x_10 - {function})\n"
        f'        .other          {function},@"STO_CUDA_ENTRY STV_DEFAULT"\n'
        f"{function}:\n"
        f".text.{function}:\n"
        f"{body}"
        "        /*0010*/                   EXIT ;\n"
        ".L_x_10:\n"
    )


#: DSL-shaped entry names, `kernel_cutlass_<...>_<N>`, the only shape that is collapsed.
_A = "kernel_cutlass_kernel_xIntPingPongDecodeSm90_object_at__A_0"
_B = "kernel_cutlass_kernel_xIntPingPongDecodeSm90_object_at__B_0"
_X = "kernel_cutlass_kernel_xRenamed_object_at__X_0"
_Y = "kernel_cutlass_kernel_xRenamed_object_at__Y_0"


def test_sass_function_name_collapses_by_the_same_rule():
    core = _core()
    a = core.normalize_sass(_sass(_A))
    b = core.normalize_sass(_sass(_X))
    assert a == b
    assert "IntPingPongDecodeSm90" not in a
    assert core.sass_entry_names(_sass(_A)) == [_A]


def test_sass_normalization_keeps_every_instruction_and_other_symbol():
    core = _core()
    base = core.normalize_sass(_sass(_A))
    changed = core.normalize_sass(_sass(
        _A, "        /*0000*/                   MOV R2, c[0x0][0x28] ;\n"))
    assert base != changed, "a changed instruction must survive SASS normalization"
    helper = core.normalize_sass(_sass(
        _A, f"        /*0000*/                   CALL.REL.NOINC `({_A}_helper) ;\n"))
    assert f"{_A}_helper" in helper


def test_a_sass_comment_cannot_nominate_a_name():
    """Anchored, like the PTX rule: only a real `.other ... STO_CUDA_ENTRY` line counts."""
    core = _core()
    listing = ('// .other kernel_cutlass_helper_A,@"STO_CUDA_ENTRY STV_DEFAULT"\n'
               "        /*0000*/  CALL.REL.NOINC `(kernel_cutlass_helper_A) ;\n")
    assert core.sass_entry_names(listing) == []
    assert "kernel_cutlass_helper_A" in core.normalize_sass(listing)


def test_a_register_named_like_a_parameter_is_refused():
    """`%<entry>_param_0` is a register, not the parameter, and no rule classifies an
    occurrence beside a `%`, in a register declaration or as an operand, so the text is
    refused rather than read either way. Before round 4 it was left raw; a refusal keeps the
    guarantee without a rule for registers named after an entry, which the DSL never
    writes. (codex, 2026-09-25)"""
    core = _core()
    declared = (f".visible .entry {_A}(\n\t.param .u64 {_A}_param_0\n)\n{{\n"
                f"\t.reg .b64 %{_A}_param_0;\n\tret;\n}}\n")
    with pytest.raises(ValueError, match="a register declaration"):
        core.normalize(declared)
    operand = (f".visible .entry {_A}(\n\t.param .u64 {_A}_param_0\n)\n{{\n"
               f"\tmov.b64 %{_A}_param_0, 1;\n\tret;\n}}\n")
    with pytest.raises(ValueError, match="beside '%'"):
        core.normalize(operand)


def test_a_block_comment_cannot_nominate_a_name():
    """Neither in PTX nor in SASS: a declaration inside `/* ... */` is not a declaration."""
    core = _core()
    def ptx(helper):
        return (".visible .entry kernel_cutlass___call___cutedsl_xIntDecodeSm100_object_at___0(\n"
                "\t.param .u64 p0\n)\n{\n/*\n"
                f".visible .entry kernel_cutlass_{helper}\n*/\n"
                f"\tcall.uni (retval), kernel_cutlass_{helper}, (%r1);\n\tret;\n}}\n")
    assert core.normalize(ptx("helper_A")) != core.normalize(ptx("helper_B"))

    def sass(helper):
        return (f'/*\n.other {helper},@"STO_CUDA_ENTRY STV_DEFAULT"\n*/\n'
                f"        /*0000*/  CALL.REL.NOINC `({helper}) ;\n")
    assert core.sass_entry_names(sass("helper_A")) == []
    assert core.normalize_sass(sass("helper_A")) != core.normalize_sass(sass("helper_B"))


def test_a_reference_of_another_normalization_is_refused():
    core = _core()
    current = {"a": {"sha256": "x", "normalization": core.NORMALIZATION_VERSION},
               "failed": {"error": "rc=1"}}
    core.require_current_normalization(current, "ref.json")
    for stale in ({"sha256": "x"}, {"sha256": "x", "normalization": 1}):
        with pytest.raises(SystemExit, match="another version"):
            core.require_current_normalization({**current, "b": stale}, "ref.json")


def test_the_committed_v1_fingerprint_records_are_refused_as_references():
    """The records under data/b200_kmajor_v_20260924/ predate the version field."""
    import json
    core = _core()
    records = sorted((_SCRIPTS.parent / "data" / "b200_kmajor_v_20260924").glob("*_ptx.json"))
    assert len(records) == 3
    for path in records:
        with pytest.raises(SystemExit, match="another version"):
            core.require_current_normalization(json.loads(path.read_text()), str(path))


def test_the_committed_records_entry_names_are_the_shape_that_is_collapsed():
    """The scoped rule collapses only `kernel_cutlass_<...>_<N>`. Every entry the committed
    SM100 records name (the decode kernel and the split reduction) has that shape, so the
    fingerprint tool keeps normalizing the kernels it was written for."""
    import json
    core = _core()
    names = {name for path in (_SCRIPTS.parent / "data" / "b200_kmajor_v_20260924").glob(
                 "*_ptx.json")
             for record in json.loads(path.read_text()).values() if isinstance(record, dict)
             for name in record.get("entries", [])}
    assert len(names) >= 2
    for name in names:
        assert core._DSL_ENTRY.fullmatch(name) and not core._DERIVED_PARAM.fullmatch(name)


@pytest.mark.parametrize(("tool", "reference", "stamp"), [
    ("_ptx_fingerprint.py", "ref", "rec"),
    ("_ptx_fingerprint_sm90.py", "before", "rec"),
    # The census reads its `--check` reference and every `--fingerprint` output through one
    # loader, `load_records`, so that one load is both.
    ("_sass_census_sm90.py", "records", "record"),
])
def test_every_gate_records_the_version_and_refuses_a_stale_reference_on_load(tool,
                                                                             reference,
                                                                             stamp):
    """The refusal itself is tested above; this pins that each gate calls it on the
    reference it just loaded, before any comparison, and stamps its own records."""
    import ast
    source = (_SCRIPTS / tool).read_text()
    assert f'{stamp}["normalization"] = NORMALIZATION_VERSION' in source
    tree = ast.parse(source)
    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and ast.unparse(n.targets[0]) == reference
             and ast.unparse(n.value) == "json.load(fh)"]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "require_current_normalization"]
    assert len(loads) == 1 and len(calls) == 1
    assert ast.unparse(calls[0].args[0]) == reference
    assert calls[0].lineno == loads[0].end_lineno + 1, "the check must follow the load"


def test_a_comment_that_holds_the_other_comment_marker_nominates_nothing():
    """`//` inside a block comment does not end it, `/*` inside a line comment does not
    open one, and neither counts inside a string. (codex, 2026-09-25)"""
    core = _core()
    def ptx(target):
        return (".visible .entry kernel_cutlass_xIntDecodeSm100_object_at___0(\n"
                "\t.param .u64 p0\n)\n/*\n.visible .entry helper_A\n"
                ".visible .entry helper_B // */\n{\n"
                f"\tcall.uni {target};\n\tret;\n}}\n")
    assert core.normalize(ptx("helper_A")) != core.normalize(ptx("helper_B"))

    def sass(function):
        return ("// a line comment that mentions /* opens nothing\n"
                f'        .other          {function},@"STO_CUDA_ENTRY STV_DEFAULT"\n'
                f"{function}:\n        /*0000*/                   EXIT ;\n")
    assert core.sass_entry_names(sass(_A)) == [_A]
    assert core.normalize_sass(sass(_A)) == core.normalize_sass(sass(_X))
    quoted = ('\t.headerflags\t@"EF // not a comment /* nor this"\n'
              f'        .other          {_A},@"STO_CUDA_ENTRY STV_DEFAULT"\n')
    assert core.sass_entry_names(quoted) == [_A]


# ---------------------------------------------------------------------------
# Canonical names that cannot collide (codex rounds 3 and 4, 2026-09-25).
# ---------------------------------------------------------------------------

_P = "kernel_cutlass_ENTRY_cutedsl"
_E = "kernel_cutlass_kernel_x_object_at__E_0"


@pytest.mark.parametrize("store", [False, True])
def test_a_text_that_already_holds_a_canonical_name_is_refused(store):
    """Codex's round-3 counterexample: an entry with the two parameters `<entry>_param_0`
    and `kernel_cutlass_ENTRY_cutedsl_param_0`. `<entry>_param_0` normalized onto the second
    one's unchanged spelling, so a load that selected either parameter gave the same text,
    also in a kernel that stores the selected value to its output. Neither text may
    normalize now, so the two can never compare equal."""
    core = _core()
    def ptx(operand):
        output = "\t.param .u64 output\n" if store else ""
        body = ("\tld.param.u64 %rd2, [output];\n\tst.global.u64 [%rd2], %rd1;\n"
                if store else "")
        return (".version 8.0\n.target sm_90\n.address_size 64\n"
                f".visible .entry {_E}(\n\t.param .u64 {_E}_param_0,\n\t.param .u64 {_P}_param_0"
                f"{',' if store else ''}\n{output})\n{{\n\t.reg .b64 %rd<3>;\n"
                f"\tld.param.u64 %rd1, [{operand}];\n{body}\tret;\n}}\n")
    for operand in (f"{_E}_param_0", f"{_P}_param_0"):
        with pytest.raises(ValueError, match="already contains"):
            core.normalize(ptx(operand))
    # The same text with an unrelated parameter name normalizes, and the two loads differ.
    plain = [core.normalize(ptx(operand).replace(f"{_P}_param_0", "another_param"))
             for operand in (f"{_E}_param_0", f"{_P}_param_0")]
    assert plain[0] != plain[1]


def test_a_sass_listing_that_already_holds_a_canonical_name_is_refused():
    core = _core()
    listing = _sass(_A, f"        /*0000*/                   CALL.REL.NOINC `({_P}) ;\n")
    with pytest.raises(ValueError, match="already contains"):
        core.normalize_sass(listing)


@pytest.mark.parametrize("where", ["ptx_register", "ptx_comment", "sass_register",
                                   "sass_comment", "ptx_second_entry"])
def test_a_canonical_spelling_is_refused_wherever_it_stands(where):
    """Checked on the raw text before any name is resolved, so no position is exempt,
    including those where no replacement could land. (codex round 4, finding 6)"""
    core = _core()
    texts = {
        "ptx_register": (core.normalize, f".visible .entry {_E}(\n)\n{{\n"
                                         f"\t.reg .b64 %{_P}_param_0;\n\tret;\n}}\n"),
        "ptx_comment": (core.normalize, f".visible .entry {_E}(\n)\n{{\n"
                                        f"\tret; // {_P}\n}}\n"),
        "sass_register": (core.normalize_sass,
                          _sass(_A, f"        /*0000*/                   MOV {_P}, R2 ;\n")),
        "sass_comment": (core.normalize_sass, "// kernel_cutlass_ENTRY3_cutedsl\n" + _sass(_A)),
        "ptx_second_entry": (core.normalize,
                             f".visible .entry {_E}(\n)\n{{\n\tret;\n}}\n"
                             f".visible .entry kernel_cutlass_ENTRY1_cutedsl(\n)\n{{\n"
                             "\tret;\n}\n"),
    }
    function, text = texts[where]
    with pytest.raises(ValueError, match="already contains"):
        function(text)


def test_a_declared_canonical_name_is_refused_in_both_formats():
    """Codex round 4, finding 6: `.visible .entry kernel_cutlass_ENTRY_cutedsl()` and
    `.other kernel_cutlass_ENTRY_cutedsl,@"STO_CUDA_ENTRY STV_DEFAULT"` both normalized,
    because `_canonical` resolved the declared name to itself before the refusal ran. The
    canonical check now comes first."""
    core = _core()
    with pytest.raises(ValueError, match="already contains"):
        core.normalize(f".visible .entry {_P}()\n{{\n\tret;\n}}\n")
    with pytest.raises(ValueError, match="already contains"):
        core.normalize_sass(f'        .other          {_P},@"STO_CUDA_ENTRY STV_DEFAULT"\n'
                            f"{_P}:\n        /*0000*/                   EXIT ;\n")


def test_each_entry_gets_its_own_canonical_name():
    """One placeholder for every entry let a load of the first entry's parameter and of the
    second entry's parameter normalize alike. Each entry is numbered in the order the text
    declares it, so a rename is still invisible and the choice of entry is not."""
    n = _core().normalize
    def ptx(first, second, target):
        return (f".visible .entry {first}(\n\t.param .u64 {first}_param_0\n)\n{{\n"
                f"\tld.param.u64 %rd1, [{first}_param_0];\n\tret;\n}}\n"
                f".visible .entry {second}(\n\t.param .u64 {second}_param_0\n)\n{{\n"
                f"\tld.param.u64 %rd1, [{second}_param_0];\n\tret;\n}}\n"
                f"// {target}\n")
    assert n(ptx(_A, _B, _A)) == n(ptx(_X, _Y, _X))
    out = n(ptx(_A, _B, _B))
    assert f".visible .entry {_P}(" in out
    assert ".visible .entry kernel_cutlass_ENTRY1_cutedsl(" in out
    assert "[kernel_cutlass_ENTRY1_cutedsl_param_0]" in out
    assert f"[{_P}_param_0]" in out


def test_one_entry_reading_another_entrys_parameter_is_refused():
    """A parameter belongs to the entry whose list declares it. A load of entry A's
    parameter inside entry B's body, or a parameter named after A in B's list, gives that
    parameter two candidate owners; it used to normalize, and the choice between the two
    entries could then be lost. (codex round 4, finding 3)"""
    core = _core()
    def ptx(declared_in_b, loaded_in_b):
        return (f".visible .entry {_A}(\n\t.param .u64 {_A}_param_0\n)\n{{\n\tret;\n}}\n"
                f".visible .entry {_B}(\n\t.param .u64 {declared_in_b}_param_0\n)\n{{\n"
                f"\tld.param.u64 %rd1, [{loaded_in_b}_param_0];\n\tret;\n}}\n")
    core.normalize(ptx(_B, _B))
    with pytest.raises(ValueError, match="the body of another entry"):
        core.normalize(ptx(_B, _A))
    with pytest.raises(ValueError, match="a parameter declaration of entry"):
        core.normalize(ptx(_A, _A))


# ---------------------------------------------------------------------------
# The scoped rule (codex round 4, owner decision 2026-09-25). The normalizer is written for
# the DSL's PTX and for `nvdisasm -c` listings, and it REFUSES every input it cannot
# classify with certainty rather than trying to be right for every valid PTX or SASS text.
# Each counterexample codex built is either refused here or, where the scoped rule now
# handles it, normalized correctly.
# ---------------------------------------------------------------------------

def test_codex_round4_register_counterexample_is_refused():
    """Finding 1, codex's own template with N=E and N=F: an unprefixed register named like
    the entry was replaced together with it, and renaming only the entry changed the text.
    E and F are not the DSL's mangling, so both are refused; the DSL-shaped equivalent is
    refused because the entry's name stands in a register declaration."""
    core = _core()
    template = ".visible .entry {n}()\n{{\n.reg .b32 {n};\nmov.b32 {n}, 1;\nret;\n}}\n"
    for name in ("E", "F"):
        with pytest.raises(ValueError, match="not the DSL's entry mangling"):
            core.normalize(template.format(n=name))
    with pytest.raises(ValueError, match="a register declaration"):
        core.normalize(template.format(n=_A))
    # The converse: an entry renamed while a register keeps the old name. The register is
    # then a name of the DSL's mangling that the text does not declare, so it is refused too.
    with pytest.raises(ValueError, match="has the DSL's entry mangling"):
        core.normalize(template.format(n=_A).replace(f".entry {_A}(", f".entry {_B}("))
    # A register whose name is not the DSL's mangling is only a register, and keeps it.
    kept = core.normalize(template.format(n=_A).replace(f".entry {_A}(", f".entry {_B}(")
                          .replace(_A, "E"))
    assert ".reg .b32 E;" in kept and f".visible .entry {_P}(" in kept


@pytest.mark.parametrize("statement", [
    "{name}:\n\tret;",
    "\t.reg .pred {name}_param_0;\n\tret;",
    "{name}_param_0:\n\tret;",
    "\tbra.uni {name}_param_0;\n{name}_param_0:\n\tret;",
    "\t.shared .b8 {name}_param_1[4];\n\tret;",
    "\tmov.u64 %rd1, {name};\n\tret;",
    "\t{name}_param_0.u32 %r1, 1;\n\tret;",
    "\t{name}_param_0 %r1, 1;\n\tret;",
    "\tmov.u32 %r1, 1; // {name}_param_0\n\tret;",
])
def test_an_entry_name_where_no_rule_classifies_it_is_refused(statement):
    """A label, a register or other declaration, the entry named as an operand, a token in
    opcode position and a kept comment: none is a position the rule recognizes, so each
    text is refused rather than normalized either way."""
    core = _core()
    text = (f".visible .entry {_A}(\n\t.param .u64 {_A}_param_0\n)\n{{\n"
            + statement.format(name=_A) + "\n}\n")
    with pytest.raises(ValueError, match="cannot classify"):
        core.normalize(text)


def test_codex_round4_split_operand_is_normalized_by_statement():
    """Finding 2: a parameter reference on a line of its own was classified as the opcode
    of that line and kept its raw name, so a pure rename changed the fingerprint. The rule
    now reads statements, so the operand is collapsed and the rename is invisible, while a
    changed parameter index still shows. Codex's own spelling, entry `E`, is refused."""
    core = _core()
    def ptx(entry, index=0):
        return (f".visible .entry {entry}(\n\t.param .u64 {entry}_param_0,\n"
                f"\t.param .u64 {entry}_param_1\n)\n{{\n"
                f"\tld.param.u64 %rd1, [\n\t  {entry}_param_{index}\n\t];\n\tret;\n}}\n")
    assert core.normalize(ptx(_A)) == core.normalize(ptx(_X))
    assert f"  {_P}_param_0" in core.normalize(ptx(_A))
    assert core.normalize(ptx(_A)) != core.normalize(ptx(_A, 1))
    with pytest.raises(ValueError, match="not the DSL's entry mangling"):
        core.normalize(ptx("E"))


def test_codex_round4_quote_inside_a_comment_opens_no_string():
    """Finding 2, second case: `.visible .entry /* " */ E()` left the name raw because the
    quote in the comment opened a supposed string. Comments are blanked before anything is
    classified, so the entry is collapsed and a rename is invisible."""
    core = _core()
    def ptx(entry):
        return f'.visible .entry /* " */ {entry}()\n{{\n\tret;\n}}\n'
    assert core.normalize(ptx(_A)) == core.normalize(ptx(_X))
    assert f'/* " */ {_P}()' in core.normalize(ptx(_A))
    with pytest.raises(ValueError, match="not the DSL's entry mangling"):
        core.normalize(ptx("E"))


def test_codex_round4_parameter_owner_counterexample_is_refused():
    """Finding 3: entries `E` and `E_param_0`, the first loading its parameter `E_param_0`,
    which then took the second entry's canonical name. An entry spelled like a parameter
    name is refused, whatever else the text holds, so no parameter has two owners."""
    core = _core()
    def ptx(first, second):
        return (f".visible .entry {first}(\n\t.param .u64 {first}_param_0\n)\n{{\n"
                f"\tld.param.u64 %rd1, [{first}_param_0];\n\tret;\n}}\n"
                f".visible .entry {second}()\n{{\n\tret;\n}}\n")
    with pytest.raises(ValueError, match="not the DSL's entry mangling"):
        core.normalize(ptx("E", "E_param_0"))
    with pytest.raises(ValueError, match="ends like a parameter name"):
        core.normalize(ptx(_A, f"{_A}_param_0"))
    with pytest.raises(ValueError, match="declared twice"):
        core.normalize(ptx(_A, _A))


@pytest.mark.parametrize("text", [
    ".entry {a}()\n{{\n\tret;\n}}\n",
    ".visible\n.entry {a}()\n{{\n\tret;\n}}\n",
    ".visible .entry {a}\n()\n{{\n\tret;\n}}\n",
    ".visible .entry {a}(\n\t.param .u64 {a}_param_0\n);\n",
    ".visible .entry {a}(\n\t.param .u64 {a}_param_0\n)\n.maxntid 128, 1, 1 junk\n{{\n\tret;\n}}\n",
    ".visible .entry {a}(\n\t.param .u64 {a}_param_0\n)\n{{\n\tret;\n",
    ".visible .entry {a}(\n\tbogus {a}_param_0\n)\n{{\n\tret;\n}}\n",
    ".visible .entry {a}(\n\t.param .u64 {a}_param_0\n)\n{{\n\tret\n}}\n",
    # A second entry whose name is not the DSL's mangling, so that only the `.entry` shape
    # rule, and not the rule on undeclared DSL names, can refuse it.
    ".visible .entry {a}()\n{{\n\tret;\n}}\n.entry helper()\n{{\n\tret;\n}}\n",
    ".visible .entry {a}()\n{{\n\tret;\n}}\n.weak .entry helper()\n{{\n\tret;\n}}\n",
])
def test_a_ptx_entry_of_a_shape_the_rule_does_not_parse_is_refused(text):
    """An `.entry` that is not `.visible .entry <name>(` beginning a line, a declaration
    without a body, anything but performance directives before the body, unbalanced braces,
    an unparsable parameter and a statement without its `;` are all refused."""
    core = _core()
    with pytest.raises(ValueError):
        core.normalize(text.format(a=_A))


@pytest.mark.parametrize("name", ["R1", "RZ", "MOV", "IMAD", "U32", "EXIT", "c", "text",
                                  "global", "type", "size", "other", "section", "function",
                                  "progbits", "L_x_10", "STO_CUDA_ENTRY", "SHI_REGISTERS",
                                  "headerflags", "nv", "constant0", "f_object_at__A",
                                  "kernel_cutlass_helper", "kernel_cutlass_x_0$"])
def test_a_sass_entry_that_is_not_the_dsl_mangling_is_refused(name):
    """Codex round 3's register-shaped entry `.other R1,@"STO_CUDA_ENTRY STV_DEFAULT"`
    rewrote `MOV R1, ...`; round 3 answered with a position rule and round 4 found the same
    class of defect in PTX. The scoped rule refuses every entry that is not
    `kernel_cutlass_<...>_<N>`, so none of these names is collapsed anywhere."""
    core = _core()
    with pytest.raises(ValueError, match="not the DSL's entry mangling"):
        core.normalize_sass(_sass(name))
    assert core.sass_entry_names(_sass(name)) == [name], "the listing still reports it"


def test_a_dsl_sass_entry_is_collapsed_only_where_it_is_a_symbol():
    """Every place `nvdisasm` names the function is collapsed, and nothing else: registers,
    opcodes, modifiers, flags, directive keywords and labels keep their spelling."""
    core = _core()
    body = ("        /*0000*/                   MOV R1, c[0x0][0x28] ;\n"
            "        /*0004*/                   IMAD.MOV.U32 R2, RZ, RZ, 0x1 ;\n"
            "        /*0008*/                   RET.REL.NODEC R6 `({f}) ;\n"
            '\t.section\t.nv.constant0.{f},"a",@progbits\n')
    out = core.normalize_sass(_sass(_A, body.format(f=_A)))
    assert out == core.normalize_sass(_sass(_X, body.format(f=_X)))
    assert _A not in out and "IntPingPongDecodeSm90" not in out
    for keyword in (".global", ".section\t.text.", "@function", "@progbits", ".L_x_10",
                    '"STO_CUDA_ENTRY STV_DEFAULT"', "SHI_REGISTERS=104", "MOV R1, c[0x0]",
                    "IMAD.MOV.U32 R2, RZ, RZ, 0x1", "EXIT ;", ".headerflags",
                    f".section\t.nv.constant0.{_P},", f"`({_P})", f"{_P}:",
                    f".text.{_P}:", f"--- .text.{_P} ---",
                    f".size           {_P},(.L_x_10 - {_P})"):
        assert keyword in out, keyword


@pytest.mark.parametrize("line", [
    "        /*0000*/                   MOV {f}, R2 ;\n",
    "        /*0000*/                   BRA {f} ;\n",
    "        /*0000*/                   CALL.REL.NOINC `({f}.x) ;\n",
    "        /*0000*/                   CALL.REL.NOINC `(@{f}) ;\n",
    '\t.sectionflags\t@"{f}"\n',
    "\t.align\t{f}\n",
    "// see {f}\n",
    "{f}_param_0:\n",
    "        /*0000*/                   CALL.REL.NOINC `({f}_param_0) ;\n",
    ".L_x_{f}_x:\n".replace("{f}", "3") + ".x.{f}.y:\n",
    "        .global         @{f}\n",
    "//--------------------- .text.{f}.x --------------------------\n",
    "//--------------------- .{f} --------------------------\n",
])
def test_a_sass_entry_name_where_no_rule_classifies_it_is_refused(line):
    """An instruction operand outside the address parentheses, a name beside a `.` or `@`,
    a string, a directive that names no symbol, a comment that is not a section banner, a
    label that is not the entry's own, and any parameter name, which `nvdisasm` never
    writes: each is refused."""
    core = _core()
    with pytest.raises(ValueError, match="cannot classify"):
        core.normalize_sass(_sass(_A, line.format(f=_A)))


def test_sass_sto_cuda_entry_outside_an_other_line_is_refused():
    core = _core()
    with pytest.raises(ValueError, match="STO_CUDA_ENTRY flag outside"):
        core.normalize_sass('\t.headerflags\t@"STO_CUDA_ENTRY"\n' + _sass(_A))


def _ptx_statements(entry: str) -> str:
    return (f".visible .entry {entry}(\n\t.param .u64 {entry}_param_0\n)\n{{\n"
            "\t.reg .pred %p<2>;\n\t.reg .b32 %r<3>;\n\t.reg .b64 %rd<2>;\n"
            f"\tld.param.u64 %rd1, [{entry}_param_0];\n"
            "\tmov.u32 %r1, 5;\n"
            "\tsetp.eq.s32 %p1, %r1, 0;\n"
            "\t@%p1 bra $L__BB0_1;\n"
            "\tadd.s32 %r2, %r1, 1;\n"
            "\tst.global.u32 [%rd1], %r2; mov.u32 %r1, 6; ret;\n"
            "$L__BB0_1:\n"
            "\tret;\n}\n")


@pytest.mark.parametrize("name", ["ret", "mov", "ld", "bra", "add", "st", "setp", "eq",
                                  "s32", "param", "u64", "u32", "entry", "visible", "reg",
                                  "pred", "b64", "global", "p", "P1", "LAB_WAIT"])
def test_a_ptx_entry_that_is_not_the_dsl_mangling_is_refused(name):
    """The PTX counterpart: an entry spelled like an opcode, a type, a state space, a
    directive keyword, a register or a label is refused, never collapsed."""
    with pytest.raises(ValueError, match="not the DSL's entry mangling"):
        _core().normalize(_ptx_statements(name))


def test_a_dsl_ptx_entry_is_collapsed_only_where_it_is_a_symbol():
    """Every instruction and directive survives, including a statement that does not begin
    its line, and the parameter is collapsed where it is read."""
    core = _core()
    out = core.normalize(_ptx_statements(_A))
    assert out == core.normalize(_ptx_statements(_X))
    for keyword in (".visible .entry", ".param .u64", "ld.param.u64 %rd1", "mov.u32 %r1",
                    "@%p1 bra $L__BB0_1;", "\tret;", ".reg .b64", "add.s32",
                    "setp.eq.s32", "st.global.u32", "; mov.u32 %r1, 6; ret;"):
        assert keyword in out, keyword
    assert f"[{_P}_param_0]" in out
    assert _A not in out


def test_the_scope_is_stated_where_the_normalizer_is_read():
    core = _core()
    assert "SCOPE OF THE ENTRY-NAME COLLAPSE" in core.__doc__
    assert "REFUSE" in core.__doc__ and "nvdisasm -c" in core.__doc__


@pytest.mark.parametrize("hidden", [
    "/* a comment\n// that a dropped line closes */ ld.param.u64 %rd1, [{a}_param_0];\n",
    ".loc 1 2 3 ; ld.param.u64 %rd1, [{a}_param_0];\n",
])
def test_a_dropped_line_that_holds_code_is_refused(hidden):
    """`//`, `.loc` and `.file` lines are dropped from the normalized text. A `//` line that
    closes a block comment and goes on, or a `.loc` line followed by a statement, holds
    code, and dropping it would let two texts that differ there normalize alike. Found
    while writing the round-5 brief; refused under the same scoped rule."""
    core = _core()
    def ptx(register):
        return (f".visible .entry {_A}(\n\t.param .u64 {_A}_param_0\n)\n{{\n"
                + hidden.format(a=_A).replace("%rd1", register) + "\tret;\n}\n")
    for register in ("%rd1", "%rd2"):
        with pytest.raises(ValueError, match="a line normalization drops also holds code"):
            core.normalize(ptx(register))


@pytest.mark.parametrize(("first", "second"), [
    # `.loc` followed by the first half of a statement that ends on the next, kept, line.
    ("\t.loc 1 2 3 add.u32 %r1,\n\t%r2, %r3;\n", "\t.loc 1 2 3 sub.u32 %r1,\n\t%r2, %r3;\n"),
    ("\t.file 1 \"k.py\" add.u32 %r1,\n\t%r2, %r3;\n",
     "\t.file 1 \"k.py\" sub.u32 %r1,\n\t%r2, %r3;\n"),
    ("\t.loc$x add.u32 %r1, %r2\n\t;\n", "\t.loc$x sub.u32 %r1, %r2\n\t;\n"),
    # A dropped line that opens a block comment comments the next, kept, line out; the same
    # text without the opener keeps it as code, and the two outputs were identical.
    ("\t.loc 1 2 3 /*\n\tmov.u32 %r1, 1;\n// */\n", "\t.loc 1 2 3\n\tmov.u32 %r1, 1;\n// */\n"),
    # A dropped `//` line that closes a block comment, then reopens one.
    ("/* a\n// */ /*\n\tmov.u32 %r1, 1;\n// */\n", "/* a\n// */\n\tmov.u32 %r1, 1;\n//\n"),
])
def test_a_dropped_line_that_changes_the_kept_text_is_refused(first, second):
    """Each pair differs on a line the normalized text keeps, as code in one text and as a
    comment or another opcode in the other, and before this rule both texts normalized to the
    SAME output, because the difference sat on, or was made by, a `.loc`, `.file` or `//`
    line that normalization drops. Found while re-deriving round 4; both texts of every pair
    must now be refused, or at least the one whose dropped line does not stand alone."""
    core = _core()
    def ptx(body):
        return f".visible .entry {_A}()\n{{\n{body}\tret;\n}}\n"
    outcomes = []
    for body in (first, second):
        try:
            outcomes.append(core.normalize(ptx(body)))
        except ValueError as error:
            assert "a line normalization drops also holds code" in str(error)
            outcomes.append(None)
    assert None in outcomes, "at least one text of the pair must be refused"
    assert outcomes[0] != outcomes[1] or outcomes[0] is None


@pytest.mark.parametrize(("line", "variant"), [
    ("\t.loc 1 2 3\n", "\t.loc 1 9 3\n"),
    ("\t.loc\t12 345 6, function_name $L__info_string0, inlined_at 1 20 3\n", "\t.loc 1 2 3\n"),
    ("\t.file 1 \"/tmp/k;{}.py\"\n", "\t.file 1 \"/other.py\"\n"),
    ("\t.file\t2 \"k.py\", 1727000000, 1234\n", ""),
    ("\t.loc 1 2 3 // a trailing comment\n", "\t.loc 1 2 3\n"),
    ("/* a block comment\n// a line inside it\n*/\n", "/* a block comment\n// another\n*/\n"),
])
def test_a_dropped_line_that_stands_alone_is_dropped(line, variant):
    """What LLVM writes, and a comment line that neither opens nor closes a block comment,
    are still dropped, so two texts that differ only there normalize alike."""
    core = _core()
    def ptx(extra):
        return f".visible .entry {_A}()\n{{\n{extra}\tret;\n}}\n"
    assert core.normalize(ptx(line)) == core.normalize(ptx(variant))


@pytest.mark.parametrize("suffix", ["_paramx_0", "_1", "_param_0_1", "_helper_2"])
def test_an_undeclared_name_of_the_dsl_mangling_is_refused(suffix):
    """A name that has the DSL's entry mangling, `kernel_cutlass_<...>_<N>`, and is neither a
    declared entry nor one of its parameters is refused wherever it stands, a comment
    included, because it may be an entry the rule did not recognize, whose name would stay
    raw. No real artifact holds one."""
    core = _core()
    for body in (f"\tcall.uni (retval), {_A}{suffix}, (%r1);\n",
                 f"\t// {_A}{suffix}\n", f"\tmov.u32 %r1, 1; /* {_A}{suffix} */\n"):
        text = (f".visible .entry {_A}(\n\t.param .u64 {_A}_param_0\n)\n{{\n{body}"
                "\tret;\n}\n")
        with pytest.raises(ValueError, match="has the DSL's entry mangling"):
            core.normalize(text)


def test_a_listing_of_another_format_is_refused():
    """`cuobjdump -sass` names a function `Function : <name>` and declares no `.other` line,
    so its entry name would stay raw and two renamed listings would compare different; the
    two such listings of the uncommitted 2026-09-23 census, archived on the volume the H100
    nodes share in `/home/donglin/remote_workspace/.superseded/2026-09-30/sass_census_20260923/`,
    are refused. So is an `nvdisasm` listing whose `.other` line is missing. The committed
    census, whose record is `profile/h100_sass_census_20260930/`, reads `nvdisasm -c` listings."""
    core = _core()
    cuobjdump = (f"\tcode for sm_90a\n\t\tFunction : {_A}\n\t.headerflags\t@\"EF_CUDA_SM90\"\n"
                 "        /*0000*/                   EXIT ;\n")
    with pytest.raises(ValueError, match="has the DSL's entry mangling"):
        core.normalize_sass(cuobjdump)
    without_other = "".join(line for line in _sass(_A).splitlines(keepends=True)
                            if ".other" not in line)
    with pytest.raises(ValueError, match="has the DSL's entry mangling"):
        core.normalize_sass(without_other)


@pytest.mark.parametrize(("text", "message"), [
    (".visible .entry {a}()\n{{\n\tmov.u32 %r1, 1;\u00a0\n\tret;\n}}\n", "alphabet"),
    (".visible .entry {a}()\n{{\n\x1f\n\tret;\n}}\n", "alphabet"),
    (".visible .entry {a}()\n{{\n\tmov.u32 %r1, 1;\x00\n\tret;\n}}\n", "alphabet"),
    (".visible .entry {a}()\n{{\n\tmov.u32 %r1, 1; // \u00e9\n\tret;\n}}\n", "alphabet"),
    ("#define P(x) x\n.visible .entry {a}()\n{{\n\tret;\n}}\n", "preprocessor"),
    (".visible .entry {a}()\n{{\n\tmov.u32 %r1, \\\n1;\n\tret;\n}}\n", "preprocessor"),
])
def test_a_character_the_toolchain_does_not_write_is_refused(text, message):
    """Python strips more than PTX's whitespace from a line's end and blanks more lines, and
    a preprocessor directive or a line continuation can spell a name no rule sees, so each is
    refused. The NUL the DSL writes after the last line of a PTX file is kept."""
    core = _core()
    with pytest.raises(ValueError, match=message):
        core.normalize(text.format(a=_A))
    ended = f".visible .entry {_A}()\n{{\n\tret;\n}}\n\x00"
    assert core.normalize(ended).endswith("}\n\x00\n")


@pytest.mark.parametrize(("text", "message"), [
    (".visible .entry {a}()\r\n{{\r\n\tret;\r\n}}\r\n", "line break other than"),
    (".visible .entry {a}()\n{{\n\t// comment\x0c\tret;\n}}\n", "line break other than"),
    (".visible .entry {a}()\n{{\n\t.pragma \"nounroll\n\";\n\tret;\n}}\n",
     "runs past the end of its line"),
    (".visible .entry {a}()\n{{\n\t.pragma \"nounroll;\n\tret;\n}}\n",
     "runs past the end of its line"),
    (".visible .entry {a}()\n{{\n\tret;\n}}\n/* never closed\n", "never ends"),
])
def test_a_text_the_comment_scanner_and_the_line_split_could_read_differently_is_refused(
        text, message):
    """A line break other than `\\n` splits a line for the output but not for the comment
    scanner; a string that crosses a line or a block comment that never ends would turn the
    rest of the text into one. None of them is written by the toolchain, so each refuses."""
    with pytest.raises(ValueError, match=message):
        _core().normalize(text.format(a=_A))
