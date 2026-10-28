"""CPU gates for `scripts/_sass_census_sm90.py`, the SASS census of the SM90 kernel.

The census fails a configuration on what it reads from three kinds of tool output, so its
parsers are the gate. They are tested here on committed fixtures of REAL tool output, captured
with CUDA 12.9 (`nvdisasm` V12.9.88, `cuobjdump` V12.9.82) and cut, where a fixture is an
excerpt, by keeping chosen line ranges of the tool's file in their original order and changing
nothing else. `tests/data/sass_census_sm90/`:

  decode_i2.excerpt.sass      `nvdisasm -c` of the census's own `i2_default` decode cubin at
                              main 6bf0be2 (research-secure-26, 2026-09-29 UTC; listing sha256
                              206a371e...0d23), lines 1-14, 125-131, 290-296, 636-666,
                              775-790, 846-919, 1185-1192, 1505-1515, 1533-1545 and 2411-2429
  decode_i2.excerpt.ptx.norm  the normalized PTX of the same compile (sha256 4a933a77...9fff,
                              the fingerprint's `i2_default` hash), lines 1-31, 946-1104 and
                              2580-2594 (the file's last line, a NUL, is left out)
  decode_i2.res-usage.txt     `cuobjdump -res-usage` of that cubin
  regions.excerpt.sass        `nvdisasm -c` of an `nvcc -arch=sm_90a -cubin` build of a kernel
                              whose three warpgroups call `setmaxnreg` as the decode kernel's
                              do (two allocate 104 registers, one releases to 32) and hold more
                              live values than those registers (research-secure-26,
                              2026-09-29 UTC; listing sha256 e8b59007...b7af), lines 1-37,
                              175-180, 298-301, 593-601, 613-614, 776-780, 1224-1230,
                              1238-1241, 1300-1306 and 1876-1896: the STL, LDL and LDL.LU of
                              a real spill before the first USETMAXREG and in every region.
                              No configuration of main's decode kernel spills, so a spill
                              listing comes from this build.
  regions.res-usage.txt       `cuobjdump -res-usage` of that cubin (STACK:240, LOCAL:0)
  spill.sass                  `nvdisasm -c` of a two-kernel `nvcc -arch=sm_90a -cubin` build
                              whose first kernel uses local memory (STL.64, STL.128, LDL.64,
                              LDL.128, DADD)
  spill.res-usage.txt         `cuobjdump -res-usage` of that cubin (two functions, STACK:512)
  call.sass                   `nvdisasm -c` of a kernel calling a device function named R12,
                              whose call target therefore names a register
  big.excerpt.sass            `nvdisasm -c` of a kernel whose code passes 64 KiB, lines 1-14,
                              4105-4115 and 6040-6045 (five-digit instruction addresses)
  combine.res-usage.txt       `cuobjdump -res-usage` of `tests/data/sm90_splitk_combine.cubin`
  combine.readelf.txt         `readelf -S -W` (GNU Binutils 2.38) of that cubin
  nvdisasm.version.txt, cuobjdump.version.txt   `--version` of the two tools

The six from `spill.sass` to `combine.readelf.txt` were captured on research-secure-27 on
2026-09-26 with the same toolkit and are carried unchanged, because none of them depends on the
decode kernel (`combine.*` read the committed combine cubin). The others were captured on
research-secure-26 on 2026-09-29 (UTC), and the two `--version` outputs and
`decode_i2.res-usage.txt` are byte-identical to those of 2026-09-26. The CUDA sources of the
four `nvcc` builds and `make_fixtures.py`, which builds them, runs the tools and cuts the
excerpts, are in `tests/data/sass_census_sm90/sources/`, whose README says how every fixture
was made and how to regenerate it. A fail-closed case changes one real line of a fixture in
memory, so every refusal below is a refusal of a listing one edit away from what the toolchain
writes.

One excerpt is inline, `DSL48_EXCERPT` below, because it is the one fixture compiled under
nvidia-cutlass-dsl 4.8.0 instead of 4.5.2, and `make_fixtures.py` does not build it: lines 15-16,
26-28, 95-118, 1040, 1456-1457 and 1481-1486 of `nvdisasm -c` of the census's own
`flex_i2_K-ch128_V-ch64` decode cubin, at main 72a801c with the DSL 4.8.0 port applied to the
working tree (research-secure-20, 2026-10-07 UTC, the same `nvdisasm`; listing sha256
7c0f34fe...9695). They hold the four previously unclassified forms that toolchain adds to the
4.5.2 listings, and the instructions around them that show their operand widths.

Run: pytest tests/test_sass_census_sm90_cpu.py
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
DATA = Path(__file__).resolve().parent / "data" / "sass_census_sm90"
REAL_CUBIN = Path(__file__).resolve().parent / "data" / "sm90_splitk_combine.cubin"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CENSUS = _load("_test_sass_census_sm90", SCRIPTS / "_sass_census_sm90.py")
FINGERPRINT = sys.modules["_ptx_fingerprint_sm90"]
CORE = sys.modules["_ptx_fingerprint_core"]
ENVELOPE = sys.modules["_quant_envelope"]

DECODE_ENTRY = (
    "kernel_cutlass_kernel_cutedsl_decode_sm90int_decode_pingpongIntPingPongDecodeSm90_"
    "object_at__tensorptrbf16_gmem_align16_o_3212840961281_tensor0000_o_2568810111213_"
    "tensor0000_o_25688101112_0")
COMBINE_ENTRY = (
    "kernel_cutlass_kernel_cutedsl_decode_sm90splitk_combineSplitKCombine_object_at__"
    "tensorptrf32_gmem_align16_o_128i64_div128i64_div1281281_tensorptrf32_gmem_align16_o_"
    "i64i641_tensorptrbf16_g_0")
#: Any line a tool wrote with an instruction address, found without the census's parser.
ADDRESS_LINE = re.compile(r"^\s*/\*[0-9a-f]+\*/")


def _text(name: str) -> str:
    return (DATA / name).read_text()


def _replace_line(text: str, old: str, new: str) -> str:
    """Return `text` with the one line that equals `old` after stripping replaced by `new`."""
    lines = text.split("\n")
    hits = [i for i, line in enumerate(lines) if line.strip() == old]
    assert len(hits) == 1, (old, len(hits))
    lines[hits[0]] = new
    return "\n".join(lines)


def _instruction(fixture: str, text: str):
    """Return the parsed instruction of `fixture` whose opcode and operands read `text`."""
    hits = [ins for ins in CENSUS.parse_listing(_text(fixture))
            if f"{ins.opcode} {ins.operands}".strip() == text]
    assert len(hits) >= 1, text
    return hits[0]


# ---------------------------------------------------------------------------
# Opcode extraction.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ["decode_i2.excerpt.sass", "spill.sass",
                                     "big.excerpt.sass", "regions.excerpt.sass", "call.sass"])
def test_every_instruction_line_of_a_real_listing_yields_one_opcode(fixture):
    text = _text(fixture)
    parsed = CENSUS.parse_listing(text)
    lines = [n for n, line in enumerate(text.splitlines(), 1) if ADDRESS_LINE.match(line)]
    assert [ins.line for ins in parsed] == lines
    assert all(ins.opcode and ins.opcode == ins.opcode.strip() for ins in parsed)


def test_predicated_lines_yield_their_predicate_and_their_opcode():
    parsed = CENSUS.parse_listing(_text("spill.sass"))
    predicated = [(ins.predicate, ins.opcode) for ins in parsed if ins.predicate]
    assert ("@!P0", "LDL.64") in predicated and ("@P1", "BRA") in predicated
    excerpt = CENSUS.parse_listing(_text("decode_i2.excerpt.sass"))
    assert ("@!P0", "IMAD.MOV.U32") in {(ins.predicate, ins.opcode) for ins in excerpt}
    # `NOP;` with no space before the semicolon is how nvdisasm writes the padding.
    assert _instruction("spill.sass", "NOP").opcode == "NOP"


ADDRESS_REFUSAL = "carries an instruction address and does not parse"


@pytest.mark.parametrize("old,new,refusal", [
    # the closing semicolon removed
    ("/*3690*/                   WARPGROUP.DEPBAR.LE gsb0, 0x0 ;",
     "        /*3690*/                   WARPGROUP.DEPBAR.LE gsb0, 0x0", ADDRESS_REFUSAL),
    # two guard predicates
    ("/*2700*/              @!P1 IMAD.MOV.U32 R6, RZ, RZ, -0xeb60d36 ;",
     "        /*2700*/              @!P1 @P0 IMAD.MOV.U32 R6, RZ, RZ, -0xeb60d36 ;",
     ADDRESS_REFUSAL),
    # the cuobjdump -sass form, which carries the encoding after the semicolon
    ("/*33a0*/                   FMUL R12, R3, R12 ;",
     "        /*33a0*/                   FMUL R12, R3, R12 ;   /* 0x0000000c030c7220 */",
     ADDRESS_REFUSAL),
    # an opcode the listing grammar does not have
    ("/*3360*/                   FFMA R13, R12, -4, R13 ;",
     "        /*3360*/                   ffma R13, R12, -4, R13 ;", ADDRESS_REFUSAL),
    # an instruction address that is not at the start of its line
    ("/*3320*/                   FMUL R8, R3.reuse, R8 ;",
     "        FMUL R8, R3.reuse, R8 ; /*3320*/", ADDRESS_REFUSAL),
    # an instruction behind a directive, which would otherwise be read as the directive and
    # left out of every count, so only the address rule refuses it
    ("/*3320*/                   FMUL R8, R3.reuse, R8 ;",
     "\t.align\t128 /*3320*/                   FMUL R8, R3.reuse, R8 ;", ADDRESS_REFUSAL),
    # a line of no kind the listing writes
    ("/*3380*/                   USEL UR32, UR13, 0x40, UP0 ;",
     "        USEL UR32, UR13, 0x40, UP0 ;",
     "is not an instruction, a label, a directive or a banner"),
])
def test_a_malformed_line_is_refused(old, new, refusal):
    text = _replace_line(_text("decode_i2.excerpt.sass"), old, new)
    with pytest.raises(CENSUS.CensusError,
                       match=rf"^\[positive-control\] listing line \d+ {re.escape(refusal)}"):
        CENSUS.parse_listing(text)


def test_a_listing_without_instructions_is_refused():
    header = "\n".join(_text("decode_i2.excerpt.sass").splitlines()[:14])
    with pytest.raises(CENSUS.CensusError, match="holds no instruction"):
        CENSUS.parse_listing(header)


def test_a_five_digit_instruction_address_is_read():
    parsed = CENSUS.parse_listing(_text("big.excerpt.sass"))
    addresses = [ins.address for ins in parsed]
    assert 0xfff0 in addresses and 0x10000 in addresses and 0x10010 in addresses
    assert addresses == sorted(addresses)


# ---------------------------------------------------------------------------
# Counts.
# ---------------------------------------------------------------------------

def test_the_counts_of_the_real_decode_excerpt():
    text = _text("decode_i2.excerpt.sass")
    counts = CENSUS.sass_counts(CENSUS.parse_listing(text))
    # The excerpt holds the first consumer's QK chain: eight MMAs, one wait, eight arrives.
    assert (counts["hgmma"], counts["depbar"], counts["arrive"]) == (8, {"0": 1}, 8)
    assert (counts["stl"], counts["ldl"]) == (0, 0)
    assert counts["hgmma"] == text.count(" HGMMA.")
    assert counts["arrive"] == text.count("WARPGROUP.ARRIVE ;")
    assert counts["instructions"] == sum(counts["opcodes"].values())


def test_stl_and_ldl_count_every_width_and_the_predicated_forms():
    text = _text("spill.sass")
    parsed = CENSUS.parse_listing(text)
    counts = CENSUS.sass_counts(parsed)
    roots = [ins for ins in parsed if ins.opcode.split(".")[0] in ("STL", "LDL")]
    assert {ins.opcode for ins in roots} == {"STL.64", "STL.128", "LDL.64", "LDL.128"}
    assert any(ins.predicate for ins in roots if ins.opcode.startswith("LDL"))
    assert counts["stl"] == len(re.findall(r"\bSTL\.", text)) == 37
    assert counts["ldl"] == len(re.findall(r"\bLDL\.", text)) == 43


def test_stl_and_ldl_count_the_unsuffixed_forms_a_register_spill_uses():
    """A register spill goes through plain 32-bit `STL`, `LDL` and `LDL.LU`, which the
    two-kernel spill build does not write, so they are counted on the `setmaxnreg` build."""
    text = _text("regions.excerpt.sass")
    parsed = CENSUS.parse_listing(text)
    counts = CENSUS.sass_counts(parsed)
    assert {ins.opcode for ins in parsed if ins.root in ("STL", "LDL")} == {
        "STL", "STL.64", "LDL", "LDL.LU", "LDL.LU.64"}
    assert counts["stl"] == len(re.findall(r"\*/\s+STL[ .]", text)) == 14
    assert counts["ldl"] == len(re.findall(r"\*/\s+LDL[ .]", text)) == 9


def test_the_depbar_immediate_is_counted_by_value():
    text = _replace_line(_text("decode_i2.excerpt.sass"),
                         "/*3690*/                   WARPGROUP.DEPBAR.LE gsb0, 0x0 ;",
                         "        /*3690*/                   WARPGROUP.DEPBAR.LE gsb0, 0x1 ;")
    assert CENSUS.sass_counts(CENSUS.parse_listing(text))["depbar"] == {"1": 1}


@pytest.mark.parametrize("form", ["WARPGROUP.DEPBAR.LE gsb0, 0x0, 0x1 ;",
                                  "WARPGROUP.DEPBAR.GE gsb0, 0x0 ;",
                                  "WARPGROUP.DEPBAR.LE gsb1, 0x0 ;",
                                  "@P0 WARPGROUP.DEPBAR.LE gsb0, 0x0 ;"])
def test_an_unknown_depbar_form_is_refused(form):
    text = _replace_line(_text("decode_i2.excerpt.sass"),
                         "/*3690*/                   WARPGROUP.DEPBAR.LE gsb0, 0x0 ;",
                         f"        /*3690*/                   {form}")
    with pytest.raises(CENSUS.CensusError, match=r"^\[depbar\]"):
        CENSUS.sass_counts(CENSUS.parse_listing(text))


def test_an_unknown_usetmaxreg_form_is_refused():
    text = _replace_line(_text("decode_i2.excerpt.sass"),
                         "/*0700*/                   USETMAXREG.DEALLOC.CTAPOOL 0x20 ;",
                         "        /*0700*/                   USETMAXREG.RELEASE.CTAPOOL 0x20 ;")
    with pytest.raises(CENSUS.CensusError, match="not a USETMAXREG form"):
        CENSUS.usetmaxreg(CENSUS.parse_listing(text))


# ---------------------------------------------------------------------------
# Register ceilings.
# ---------------------------------------------------------------------------

def test_the_listing_is_split_at_every_usetmaxreg():
    """The expected regions were derived by hand from the excerpt. Region 1 runs from the IO
    warpgroup's release to the line before the first allocation, and its highest register is
    the R10 of `ISETP.GT.AND P0, PT, R10, 0x3, PT`. Region 2 holds the first consumer's lines,
    whose highest register is the 32-bit shared address of `LDS.128 R4, [R99]`; no
    multi-register operand there reaches past it (the HGMMA A fragment R84 covers R84 to R87).
    Region 3 tops out at `VIADD R92, R0.reuse, 0xffffff80`."""
    sass = CENSUS.measure_sass(_text("decode_i2.excerpt.sass"))
    assert [(m["kind"], m["registers"], m["line"]) for m in sass["usetmaxreg"]] == [
        ("release", 32, 18), ("alloc", 104, 39), ("alloc", 104, 172)]
    regions = [(r["first_line"], r["last_line"], r["literal"]["register"],
                r["literal"]["line"], r["widened"]["register"], r["widened"]["line"])
               for r in sass["regions"]]
    assert regions == [(18, 38, 10, 31, 10, 31), (39, 171, 99, 62, 99, 62),
                       (172, 199, 92, 175, 92, 175)]
    assert sass["consumer_ceilings"] == [{"literal": 100, "widened": 100},
                                         {"literal": 93, "widened": 93}]
    assert [(r["stl"], r["ldl"]) for r in sass["regions"]] == [(0, 0)] * 3


def test_a_pair_past_the_highest_register_written_widens_its_region_ceiling():
    """In every real listing the widened ceiling equals the literal one, so this edits one
    real line of region 3: `CS2R R30, SRZ` at line 181 becomes `CS2R R92, SRZ`, whose pair
    also covers R93. The highest register written stays R92, first named by the VIADD at
    line 175, and the widened ceiling rises to R93 at line 181. The other regions keep their
    ceilings."""
    text = _replace_line(_text("decode_i2.excerpt.sass"),
                         "/*5c60*/                   CS2R R30, SRZ ;",
                         "        /*5c60*/                   CS2R R92, SRZ ;")
    sass = CENSUS.measure_sass(text)
    regions = [(r["literal"]["register"], r["literal"]["registers"], r["literal"]["line"],
                r["widened"]["register"], r["widened"]["registers"], r["widened"]["line"])
               for r in sass["regions"]]
    assert regions == [(10, 11, 31, 10, 11, 31), (99, 100, 62, 99, 100, 62),
                       (92, 93, 175, 93, 94, 181)]
    assert sass["consumer_ceilings"] == [{"literal": 100, "widened": 100},
                                         {"literal": 93, "widened": 94}]


def test_each_region_counts_the_spill_instructions_of_its_own_stretch():
    """The excerpt keeps real local-memory instructions on every side of its three
    USETMAXREG lines: one STL.64 before the first allocation of 104 registers, four STL and
    five LDL after it, five STL and two LDL after the second, and four STL and two LDL after
    the release to 32. Each region's counts are checked against a count, taken without the
    census's parser, of the listing lines from its USETMAXREG to the line before the next."""
    text = _text("regions.excerpt.sass")
    lines = text.splitlines()
    sass = CENSUS.measure_sass(text)
    assert [(m["kind"], m["registers"]) for m in sass["usetmaxreg"]] == [
        ("alloc", 104), ("alloc", 104), ("release", 32)]
    marks = [m["line"] for m in sass["usetmaxreg"]]
    stretches = list(zip(marks, marks[1:] + [len(lines) + 1]))

    def count(root, first, stop):
        return sum(1 for line in lines[first - 1:stop - 1]
                   if re.search(rf"\*/\s+{root}[ .]", line))
    independent = [(count("STL", a, z), count("LDL", a, z)) for a, z in stretches]
    assert [(r["stl"], r["ldl"]) for r in sass["regions"]] == independent == [
        (4, 5), (5, 2), (4, 2)]
    # the listing's own totals also hold the store before the first USETMAXREG
    assert (sass["stl"], sass["ldl"]) == (14, 9)


@pytest.mark.parametrize("fixture,text,literal,covered", [
    ("decode_i2.excerpt.sass", "IMAD.WIDE.U32 R2, R7, 0x2, R2", {2, 7}, {2, 3, 7}),
    ("decode_i2.excerpt.sass", "CS2R R38, SRZ", {38}, {38, 39}),
    ("decode_i2.excerpt.sass", "LDS.128 R4, [R99]", {4, 99}, {4, 5, 6, 7, 99}),
    ("decode_i2.excerpt.sass", "STG.E.U16 desc[UR44][R6.64], R5", {5, 6}, {5, 6, 7}),
    ("decode_i2.excerpt.sass", "HGMMA.64x8x16.F32.BF16 R24, R56, gdesc[UR48], RZ, !UPT, gsb0",
     {24, 56}, set(range(24, 28)) | set(range(56, 60))),
    ("decode_i2.excerpt.sass", "HGMMA.64x8x16.F32.BF16 R24, R60, gdesc[UR32], R24, gsb0",
     {24, 60}, set(range(24, 28)) | set(range(60, 64))),
    # the 64-bit return address, whose high half the listing writes just before the RET
    ("decode_i2.excerpt.sass", f"RET.REL.NODEC R6 `({DECODE_ENTRY})", {6}, {6, 7}),
    ("call.sass", "RET.REL.NODEC R4 `(call_kernel)", {4}, {4, 5}),
    ("decode_i2.excerpt.sass", "LOP3.LUT R69, R98.reuse, 0x600060, R69, 0xf8, !PT",
     {69, 98}, {69, 98}),
    # the transaction count is 32-bit; only the destination, RZ here, is a pair
    ("decode_i2.excerpt.sass", "SYNCS.ARRIVE.TRANS64 RZ, [UR25+0x40], R5", {5}, {5}),
    ("spill.sass", "STL.128 [R1+0x10], R4", {1, 4}, {1, 4, 5, 6, 7}),
    ("spill.sass", "LDL.64 R12, [R12]", {12}, {12, 13}),
    ("spill.sass", "STL.64 [R21], R10", {21, 10}, {21, 10, 11}),
    ("spill.sass", "LDC.64 R2, c[0x0][0x210]", {2}, {2, 3}),
    # an addend that is not the destination, so only the addend rule covers R7
    ("regions.excerpt.sass", "IMAD.WIDE R4, R4, 0x4, R6", {4, 6}, {4, 5, 6, 7}),
    ("regions.excerpt.sass", "STL [R1+0x38], R2", {1, 2}, {1, 2}),
    ("regions.excerpt.sass", "LDL R2, [R1+0x4]", {1, 2}, {1, 2}),
    ("regions.excerpt.sass", "LDL.LU R0, [R1+0x78]", {0, 1}, {0, 1}),
    ("regions.excerpt.sass", "LDL.LU.64 R2, [R1+0x80]", {1, 2}, {1, 2, 3}),
])
def test_multi_register_operands_are_widened_by_the_table(fixture, text, literal, covered):
    assert CENSUS.registers_named(_instruction(fixture, text)) == (literal, covered)


@pytest.mark.parametrize("fixture,old,new,literal,covered", [
    # an HGMMA whose C operand is not its accumulator; in the 51 listings C is D or RZ
    ("decode_i2.excerpt.sass",
     "/*34f0*/                   HGMMA.64x8x16.F32.BF16 R24, R60, gdesc[UR32], R24, gsb0 ;",
     "        /*34f0*/                   HGMMA.64x8x16.F32.BF16 R24, R60, gdesc[UR32], R40, gsb0 ;",
     {24, 60, 40}, set(range(24, 28)) | set(range(60, 64)) | set(range(40, 44))),
    # the barrier's 64-bit state returned to a register; the listings discard it to RZ
    ("decode_i2.excerpt.sass",
     "/*10f0*/               @P0 SYNCS.ARRIVE.TRANS64 RZ, [UR25+0x40], R5 ;",
     "        /*10f0*/               @P0 SYNCS.ARRIVE.TRANS64 R8, [UR25+0x40], R5 ;",
     {5, 8}, {5, 8, 9}),
    # the same in the .ART0 form, which the listings also write with an RZ destination
    ("decode_i2.excerpt.sass",
     "/*10f0*/               @P0 SYNCS.ARRIVE.TRANS64 RZ, [UR25+0x40], R5 ;",
     "        /*10f0*/               @P0 SYNCS.ARRIVE.TRANS64.ART0 R8, [UR25+0x40], R5 ;",
     {5, 8}, {5, 8, 9}),
])
def test_a_rule_no_listing_exercises_widens_one_real_line_edited(fixture, old, new, literal,
                                                                 covered):
    (edited,) = [ins for ins in CENSUS.parse_listing(_replace_line(_text(fixture), old, new))
                 if ins.address == int(old[2:6], 16)]
    assert CENSUS.registers_named(edited) == (literal, covered)


def test_an_instruction_naming_no_general_register_is_not_classified():
    assert CENSUS.registers_named(_instruction("decode_i2.excerpt.sass",
                                               "BAR.ARV 0x5, 0x100")) is None
    assert CENSUS.registers_named(
        _instruction("decode_i2.excerpt.sass", "USETMAXREG.DEALLOC.CTAPOOL 0x20")) is None
    assert CENSUS.registers_named(_instruction("decode_i2.excerpt.sass",
                                               "UIADD3 UR28, UR17, 0x980, URZ")) is None
    # A call target is a label, not an operand, even when its name reads as a register.
    call = _instruction("call.sass", "CALL.REL.NOINC `($call_kernel$R12)")
    assert CENSUS._GPR.search(call.operands)[0] == "R12"
    assert CENSUS.registers_named(call) is None


def test_an_unclassified_opcode_is_refused():
    # A real listing: the spill kernel's DADD reads and writes 64-bit register pairs, and
    # the table has no rule for it.
    with pytest.raises(CENSUS.CensusError, match=r"opcode DADD names a general register"):
        CENSUS.measure_sass(_text("spill.sass"))
    # One real line of the decode excerpt with its opcode changed to one the table lacks.
    text = _replace_line(_text("decode_i2.excerpt.sass"),
                         "/*33a0*/                   FMUL R12, R3, R12 ;",
                         "        /*33a0*/                   DMUL R12, R3, R12 ;")
    with pytest.raises(CENSUS.CensusError, match=r"^\[positive-control\].*DMUL"):
        CENSUS.measure_sass(text)


def test_an_unclassified_opcode_before_the_first_usetmaxreg_is_refused():
    text = _replace_line(_text("decode_i2.excerpt.sass"),
                         "/*06d0*/                   R2UR UR5, R2 ;",
                         "        /*06d0*/                   F2F.F64.F32 R2, R3 ;")
    with pytest.raises(CENSUS.CensusError, match="F2F.F64.F32"):
        CENSUS.measure_sass(text)


#: The nvidia-cutlass-dsl 4.8.0 excerpt of the module docstring. Its first two stretches open
#: the kernel, and its third, from 0x04e0 to 0x0650, is the reciprocal refinement and the
#: quotient of an unsigned division.
DSL48_EXCERPT = """\
        /*0000*/                   LDC R1, c[0x0][0x28] ;
        /*0010*/                   S2R R0, SR_TID.X ;
        /*00b0*/                   SHF.R.U32.HI R2, RZ, 0x5, R0 ;
        /*00c0*/                   ULEA UR25, UR8, UR25, 0x18 ;
        /*00d0*/                   ISETP.NE.U32.AND P0, PT, R0, RZ, PT ;
        /*04e0*/                   F2I.FTZ.U32.TRUNC.NTZ R3, R2 ;
        /*04f0*/                   UIADD3 UR11, UR9, 0x3f, URZ ;
        /*0500*/                   MOV R2, RZ ;
        /*0510*/                   USHF.R.S32.HI UR8, URZ, 0x1f, UR11 ;
        /*0520*/                   IMAD.MOV R5, RZ, RZ, -R3 ;
        /*0530*/                   ULEA.HI UR8, UR8, UR11, URZ, 0x6 ;
        /*0540*/                   IMAD R5, R5, R6, RZ ;
        /*0550*/                   ULOP3.LUT UR12, UR8, 0xffffffc0, URZ, 0xc0, !UPT ;
        /*0560*/                   IMAD.HI.U32 R2, R3, R5, R2 ;
        /*0570*/                   UISETP.NE.U32.AND UP0, UPT, UR11, UR12, UPT ;
        /*0580*/                   UISETP.LT.AND UP0, UPT, UR11, URZ, UP0 ;
        /*0590*/                   USHF.R.S32.HI UR11, URZ, 0x6, UR8 ;
        /*05a0*/                   ULEA.HI.SX32 UR8, UR8, 0xffffffff, 0x1a ;
        /*05b0*/                   @!UP0 UIADD3 UR8, UR11, URZ, URZ ;
        /*05c0*/                   UISETP.GE.AND UP0, UPT, UR5, 0x8, UPT ;
        /*05d0*/                   UIADD3 UR12, UR21, -0x1, UR8 ;
        /*05e0*/                   ULOP3.LUT UR11, UR12, UR21, URZ, 0x3c, !UPT ;
        /*05f0*/                   IMAD.U32 R4, RZ, RZ, UR12 ;
        /*0600*/                   ISETP.LE.AND P2, PT, RZ, UR11, PT ;
        /*0610*/                   IABS R4, R4 ;
        /*0620*/                   IMAD.MOV.U32 R3, RZ, RZ, R4 ;
        /*0630*/                   IMAD.MOV.U32 R4, RZ, RZ, R7 ;
        /*0640*/                   IMAD.HI.U32 R2, R2, R3, RZ ;
        /*0650*/                   IMAD R3, R2, R4, R3 ;
        /*3e10*/                   LOP3.LUT R81, R3, 0x7, RZ, 0xc0, !PT ;
        /*57c0*/                   ISETP.GT.U32.AND P0, PT, R0, 0x7, PT ;
        /*57d0*/                   ISETP.NE.U32.OR P1, PT, R81, RZ, P0 ;
        /*5950*/                   ISETP.NE.U32.AND P1, PT, R9, 0x3, PT ;
        /*5960*/                   BAR.SYNC.DEFER_BLOCKING 0x2, 0x80 ;
        /*5970*/                   SEL R40, R40, 0x1, P1 ;
        /*5980*/                   ISETP.NE.U32.AND P2, PT, R40, 0x4, PT ;
        /*5990*/                   ISETP.EQ.U32.XOR P1, PT, R9, 0x3, !P2 ;
        /*59a0*/                   SEL R9, RZ, 0x1, !P1 ;
"""
DSL48_FORMS = ("IMAD.HI.U32", "ISETP.NE.U32.AND", "ISETP.NE.U32.OR", "ISETP.EQ.U32.XOR")

#: Lines 4584-4586, 4600, 4614, 4731, 4770 and 4834 of `nvdisasm -c` of the census's
#: `d256_i4_st4_fused` decode cubin under nvidia-cutlass-dsl 4.8.0, with the NVVM options of
#: `pingpong_runtime.nvvm_options` (research-secure-20, 2026-10-08; listing sha256
#: daa1085ab9ea16cd...): the three P2R that copy predicates into R60, R59 and R58, and the only
#: later reads of those registers, three 32-bit compares.
DSL48_P2R_EXCERPT = """\
        /*11880*/                   UP2UR UR42, UPR, URZ, 0x1 ;
        /*11890*/                   PLOP3.LUT P0, PT, PT, PT, UP0, 0x80, 0x0 ;
        /*118a0*/                   P2R R60, PR, RZ, 0x1 ;
        /*11980*/                   P2R R59, PR, RZ, 0x2 ;
        /*11a60*/                   P2R R58, PR, RZ, 0x4 ;
        /*121b0*/                   ISETP.NE.AND P1, PT, R59, RZ, PT ;
        /*12420*/                   ISETP.NE.AND P2, PT, R58, RZ, PT ;
        /*12820*/                   ISETP.NE.AND P0, PT, R60, RZ, PT ;
"""


def test_p2r_is_classified_by_its_own_table_entry(monkeypatch):
    """The P2R excerpt measures with the table as it is, and is refused once P2R's entry is
    taken out."""
    assert CENSUS.measure_sass(DSL48_P2R_EXCERPT)["opcodes"]["P2R"] == 3
    monkeypatch.setattr(CENSUS, "KNOWN_32", CENSUS.KNOWN_32 - {"P2R"})
    with pytest.raises(CENSUS.CensusError,
                       match=r"^\[positive-control\] listing line \d+: opcode P2R names a "
                             r"general register"):
        CENSUS.measure_sass(DSL48_P2R_EXCERPT)


def _dsl48(text: str):
    """Return the one instruction of the 4.8.0 excerpt whose opcode and operands read `text`."""
    (hit,) = [ins for ins in CENSUS.parse_listing(DSL48_EXCERPT)
              if f"{ins.opcode} {ins.operands}" == text]
    return hit


@pytest.mark.parametrize("form", DSL48_FORMS)
def test_each_form_dsl_4_8_0_adds_is_classified_by_its_own_table_entry(monkeypatch, form):
    """The excerpt measures with the table as it is, and each of the four forms is refused
    once its own entry is taken out, so each entry is what admits a line of the excerpt."""
    assert CENSUS.measure_sass(DSL48_EXCERPT)["opcodes"][form] >= 1
    monkeypatch.setattr(CENSUS, "KNOWN_32", CENSUS.KNOWN_32 - {form})
    monkeypatch.setattr(CENSUS, "PAIR_ADDEND", CENSUS.PAIR_ADDEND - {form})
    with pytest.raises(CENSUS.CensusError,
                       match=rf"^\[positive-control\] listing line \d+: opcode "
                             rf"{re.escape(form)} names a general register"):
        CENSUS.measure_sass(DSL48_EXCERPT)


@pytest.mark.parametrize("text,literal,covered", [
    # S2R R0, SR_TID.X wrote the one register compared, and R1 holds the stack pointer
    ("ISETP.NE.U32.AND P0, PT, R0, RZ, PT", {0}, {0}),
    # LOP3.LUT R81, R3, 0x7, RZ, 0xc0, !PT wrote the one register compared
    ("ISETP.NE.U32.OR P1, PT, R81, RZ, P0", {81}, {81}),
    ("ISETP.EQ.U32.XOR P1, PT, R9, 0x3, !P2", {9}, {9}),
    # the addend pair is R2, which MOV R2, RZ zeroed, and R3, the estimate, which the first
    # source also names
    ("IMAD.HI.U32 R2, R3, R5, R2", {2, 3, 5}, {2, 3, 5}),
    ("IMAD.HI.U32 R2, R2, R3, RZ", {2, 3}, {2, 3}),
])
def test_the_forms_dsl_4_8_0_adds_are_widened_by_the_table(text, literal, covered):
    assert CENSUS.registers_named(_dsl48(text)) == (literal, covered)


def test_imad_hi_widens_its_addend_and_not_its_destination():
    """Both real IMAD.HI.U32 lines name every register they cover, the first because its
    first source is the high half of its addend, so the rule shows in the width function and
    in one real line edited to name neither neighbour."""
    for text in ("IMAD.HI.U32 R2, R3, R5, R2", "IMAD.HI.U32 R2, R2, R3, RZ"):
        width = CENSUS.operand_widths(_dsl48(text))
        assert [width(i) for i in range(4)] == [1, 1, 1, 2]
    edited = _replace_line(DSL48_EXCERPT,
                           "/*0560*/                   IMAD.HI.U32 R2, R3, R5, R2 ;",
                           "        /*0560*/                   IMAD.HI.U32 R6, R3, R5, R8 ;")
    (ins,) = [ins for ins in CENSUS.parse_listing(edited) if ins.address == 0x560]
    assert CENSUS.registers_named(ins) == ({3, 5, 6, 8}, {3, 5, 6, 8, 9})


@pytest.mark.parametrize("old,new,opcode", [
    # the signed high multiply, which no listing writes
    ("/*0640*/                   IMAD.HI.U32 R2, R2, R3, RZ ;",
     "        /*0640*/                   IMAD.HI R2, R2, R3, RZ ;", "IMAD.HI"),
    # the extended compare that reads the high halves of a 64-bit compare, which no listing
    # writes
    ("/*00d0*/                   ISETP.NE.U32.AND P0, PT, R0, RZ, PT ;",
     "        /*00d0*/                   ISETP.NE.U32.AND.EX P0, PT, R0, RZ, PT, P0 ;",
     "ISETP.NE.U32.AND.EX"),
])
def test_a_neighbouring_form_no_listing_writes_is_still_refused(old, new, opcode):
    with pytest.raises(CENSUS.CensusError,
                       match=rf"^\[positive-control\].*opcode {re.escape(opcode)} names"):
        CENSUS.measure_sass(_replace_line(DSL48_EXCERPT, old, new))


# ---------------------------------------------------------------------------
# The PTX.
# ---------------------------------------------------------------------------

def test_ptx_counts_by_full_token():
    text = _text("decode_i2.excerpt.ptx.norm")
    ptx = CENSUS.measure_ptx(text)
    assert ptx["mma_async"] == 8 == text.count("wgmma.mma_async.")
    assert ptx["wait_group"] == {"0": 1}
    assert ptx["tokens"]["wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"] == 8
    assert ptx["tokens"]["wgmma.commit_group.sync.aligned"] == 1
    assert ptx["tokens"]["wgmma.fence.sync.aligned"] == 1
    # predicated statements are counted under their opcode
    assert ptx["tokens"]["bra"] == text.count("\t@%p") and ptx["tokens"]["bra"] > 0


def test_the_wait_group_immediate_is_counted_by_value():
    text = _replace_line(_text("decode_i2.excerpt.ptx.norm"),
                         "wgmma.wait_group.sync.aligned 0;", "\twgmma.wait_group.sync.aligned 1;")
    assert CENSUS.measure_ptx(text)["wait_group"] == {"1": 1}


@pytest.mark.parametrize("new", ["\twgmma.wait_group.sync.aligned %r5;",
                                 "\twgmma.wait_group.sync 0;",
                                 "\twgmma.wait_group.sync.aligned.x 0;"])
def test_a_wait_group_of_another_form_is_refused(new):
    text = _replace_line(_text("decode_i2.excerpt.ptx.norm"),
                         "wgmma.wait_group.sync.aligned 0;", new)
    with pytest.raises(CENSUS.CensusError, match=r"^\[depbar\] PTX statement"):
        CENSUS.measure_ptx(text)


def test_the_positive_control_refuses_a_ptx_without_wgmma():
    header = "\n".join(_text("decode_i2.excerpt.ptx.norm").splitlines()[:31]) + "\n"
    record = _passing_record()
    record["ptx"] = CENSUS.measure_ptx(header)
    record["sass"]["hgmma"], record["sass"]["depbar"] = 0, {}
    failures = CENSUS.judge(record)
    assert failures == ["[positive-control] the PTX has no wgmma.mma_async",
                        "[positive-control] the PTX has no wgmma.wait_group"]


def _structural_counts_before_the_move(norm: str) -> dict:
    """This is `structural_counts` as main 6bf0be2 has it, unchanged since 6d4a06e, before its
    walk became `ptx_statements`."""
    ops: dict[str, int] = {}
    for line in norm.splitlines():
        s = line.split("//")[0].strip()
        if not s or not s.endswith(";"):
            continue
        first = s.split()[0]
        if first.endswith(":"):
            s = s[len(first):].strip()
            if not s:
                continue
        if s.startswith("@"):
            parts = s.split()
            if len(parts) < 2:
                continue
            tok = parts[1]
        elif s.startswith((".", "{", "}", "$")):
            continue
        else:
            tok = s.split()[0]
        tok = tok.rstrip(";")
        key = tok.split(".")[0]
        if key:
            ops[key] = ops.get(key, 0) + 1
    return {
        "lines": len(norm.splitlines()),
        "instructions": sum(ops.values()),
        "distinct_opcodes": len(ops),
        "opcodes": dict(sorted(ops.items())),
    }


@pytest.mark.parametrize("text", [
    _text("decode_i2.excerpt.ptx.norm"),
    # the forms the walk distinguishes, which the real excerpt does not all hold
    "$L__BB0_1: add.s32 %r1, %r2, 1;\n$L__BB0_2:\n@%p1 ret;\n@%p2;\nret;\n"
    "\t@%p3 bra \t$L__BB0_2; // a comment\n.reg .b32 %r<4>;\n{\n}\n",
])
def test_structural_counts_is_unchanged_by_the_move_of_its_walk(text):
    assert CORE.structural_counts(text) == _structural_counts_before_the_move(text)


def test_ptx_statements_yields_the_full_token_and_the_operands():
    text = "$L__BB0_1: add.s32 %r1, %r2, 1;\n@%p1 bra \t$L__BB0_2;\nret;\n" \
           "\twgmma.wait_group.sync.aligned 0;\n"
    assert list(CORE.ptx_statements(text)) == [
        ("add.s32", "%r1, %r2, 1"), ("bra", "$L__BB0_2"), ("ret", ""),
        ("wgmma.wait_group.sync.aligned", "0")]


# ---------------------------------------------------------------------------
# cuobjdump -res-usage and the .text size.
# ---------------------------------------------------------------------------

def test_the_resource_line_of_the_decode_entry():
    usage = CENSUS.parse_res_usage(_text("decode_i2.res-usage.txt"), DECODE_ENTRY)
    assert {k: usage[k] for k in CENSUS.RES_FIELDS} == {"REG": 80, "STACK": 0,
                                                         "SHARED": 1024, "LOCAL": 0}


def test_a_register_spill_is_booked_under_stack_and_not_under_local():
    """The `setmaxnreg` build spills (`regions.excerpt.sass`), and `cuobjdump` books its spill
    slots as the stack frame, STACK:240, with LOCAL:0. So the LOCAL figure alone would pass a
    spilling build, and the STACK figure and the spill instructions are what fail it."""
    usage = CENSUS.parse_res_usage(_text("regions.res-usage.txt"), "regions_kernel")
    assert {k: usage[k] for k in CENSUS.RES_FIELDS} == {"REG": 80, "STACK": 240,
                                                         "SHARED": 0, "LOCAL": 0}
    record = _passing_record()
    record["resource_usage"] = usage
    assert CENSUS.judge(record) == ["[stack-local] STACK is 240 bytes, must be 0"]


def test_a_second_function_is_refused():
    with pytest.raises(CENSUS.CensusError, match="lists 2 functions"):
        CENSUS.parse_res_usage(_text("spill.res-usage.txt"), "_Z12spill_kernelPKdPdii")


def test_a_function_that_is_not_the_decode_entry_is_refused():
    with pytest.raises(CENSUS.CensusError, match="not the decode entry"):
        CENSUS.parse_res_usage(_text("combine.res-usage.txt"), DECODE_ENTRY)


@pytest.mark.parametrize("field", CENSUS.RES_FIELDS)
def test_a_missing_resource_field_is_refused(field):
    text = re.sub(rf"\b{field}:\d+ ?", "", _text("decode_i2.res-usage.txt"))
    with pytest.raises(CENSUS.CensusError, match=rf"omits \['{field}'\]"):
        CENSUS.parse_res_usage(text, DECODE_ENTRY)


def test_the_text_size_of_the_real_cubin_is_the_one_readelf_prints():
    row = next(line for line in _text("combine.readelf.txt").splitlines()
               if f" .text.{COMBINE_ENTRY} " in line)
    tokens = row.split()                 # [Nr] Name Type Address Off Size ES ...
    size = int(tokens[tokens.index(f".text.{COMBINE_ENTRY}") + 4], 16)
    assert size == 0x1800
    assert CENSUS.text_section_size(REAL_CUBIN.read_bytes(), COMBINE_ENTRY) == size


def _shstrndx_past_the_end(image: bytearray) -> None:
    struct.pack_into("<H", image, 62, struct.unpack_from("<H", image, 60)[0])


def _section_table_past_the_end(image: bytearray) -> None:
    struct.pack_into("<Q", image, 40, len(image))


@pytest.mark.parametrize("damage", [
    lambda image: image[:40],
    lambda image: image[:len(image) // 2],
    _shstrndx_past_the_end,
    _section_table_past_the_end,
])
def test_a_damaged_image_is_refused(damage):
    image = bytearray(REAL_CUBIN.read_bytes())
    damaged = damage(image)
    with pytest.raises(CENSUS.CensusError, match=r"^\[artifacts\] the ELF image does not parse"):
        CENSUS.text_section_size(bytes(image if damaged is None else damaged), COMBINE_ENTRY)


def _section_header(image, name: str) -> int:
    """Return the offset of the header of the section called `name`."""
    shoff = struct.unpack_from("<Q", image, 40)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", image, 58)
    names = struct.unpack_from("<Q", image, shoff + shstrndx * shentsize + 24)[0]
    for index in range(shnum):
        header = shoff + index * shentsize
        start = names + struct.unpack_from("<I", image, header)[0]
        if bytes(image[start:image.index(b"\0", start)]) == name.encode():
            return header
    raise AssertionError(name)


def _text_contents_past_the_end(image: bytearray) -> None:
    struct.pack_into("<Q", image, _section_header(image, f".text.{COMBINE_ENTRY}") + 24,
                     len(image))


def _text_retyped(image: bytearray) -> None:
    struct.pack_into("<I", image, _section_header(image, f".text.{COMBINE_ENTRY}") + 4, 8)


def _name_table_past_the_end(image: bytearray) -> None:
    struct.pack_into("<Q", image, _section_header(image, ".shstrtab") + 32, len(image))


def _name_starting_past_the_table(image: bytearray) -> None:
    header = _section_header(image, ".shstrtab")
    size = struct.unpack_from("<Q", image, header + 32)[0]
    struct.pack_into("<I", image, _section_header(image, ".strtab"), size + 1)


def _last_name_unterminated(image: bytearray) -> None:
    header = _section_header(image, ".shstrtab")
    names, size = struct.unpack_from("<QQ", image, header + 24)
    assert image[names + size - 1] == 0
    image[names + size - 1] = ord("x")


def _elf32_class(image: bytearray) -> None:
    assert image[4] == 2
    image[4] = 1


def _big_endian_encoding(image: bytearray) -> None:
    assert image[5] == 1
    image[5] = 2


def _section_headers_of_elf32_size(image: bytearray) -> None:
    assert struct.unpack_from("<H", image, 58)[0] == 64
    struct.pack_into("<H", image, 58, 40)


def _debug_frame_named_as_the_entry(image: bytearray) -> None:
    name = struct.unpack_from("<I", image, _section_header(image, f".text.{COMBINE_ENTRY}"))[0]
    struct.pack_into("<I", image, _section_header(image, ".debug_frame"), name)


@pytest.mark.parametrize("damage,message", [
    (_elf32_class, "ELF class 1 and data encoding 1, not little-endian ELF64"),
    (_big_endian_encoding, "ELF class 2 and data encoding 2, not little-endian ELF64"),
    (_section_headers_of_elf32_size, "section headers of 40 bytes, not 64"),
    (_text_contents_past_the_end, "runs past the end of the image"),
    (_text_retyped, "is of type 8"),
    (_name_table_past_the_end, "the section-name table (11992 bytes at offset 64) runs past"),
    (_name_starting_past_the_table, "does not start and end inside the section-name table"),
    (_last_name_unterminated, "does not start and end inside the section-name table"),
    (_debug_frame_named_as_the_entry, f"2 sections are named .text.{COMBINE_ENTRY}"),
])
def test_a_section_the_image_cannot_hold_is_refused(damage, message):
    image = bytearray(REAL_CUBIN.read_bytes())
    damage(image)
    with pytest.raises(CENSUS.CensusError, match=r"^\[artifacts\] the ELF image does not parse"
                       ) as refused:
        CENSUS.text_section_size(bytes(image), COMBINE_ENTRY)
    assert message in str(refused.value)


def test_an_entry_the_image_does_not_hold_is_refused():
    with pytest.raises(CENSUS.CensusError, match=r"0 sections are named"):
        CENSUS.text_section_size(REAL_CUBIN.read_bytes(), DECODE_ENTRY)


@pytest.mark.parametrize("fixture", ["nvdisasm.version.txt", "cuobjdump.version.txt"])
def test_a_tool_version_names_its_release(fixture):
    lines = CENSUS.release_lines(_text(fixture))
    assert "Cuda compilation tools, release 12.9" in " ".join(lines)
    with pytest.raises(CENSUS.CensusError, match="without naming its release"):
        CENSUS.release_lines("\n".join(line for line in _text(fixture).splitlines()
                                       if "release" not in line))


# ---------------------------------------------------------------------------
# The pass criteria.
# ---------------------------------------------------------------------------

def _passing_record() -> dict:
    """Return a record whose measurements are those of the real fixtures, which pass
    together."""
    return {
        "label": "i2_default", "errors": [], "normalization": CORE.NORMALIZATION_VERSION,
        "resource_usage": CENSUS.parse_res_usage(_text("decode_i2.res-usage.txt"),
                                                 DECODE_ENTRY),
        "sass": CENSUS.measure_sass(_text("decode_i2.excerpt.sass")),
        "ptx": CENSUS.measure_ptx(_text("decode_i2.excerpt.ptx.norm")),
        "checks": {"manifest_verified": True, "ptx_decode_entries": 1, "decode_images": 1},
        "identity_checks": {"package_in_tree": True, "cute_dsl_arch": "sm_90a",
                            "dsl_identity": True, "nvdisasm_version": True,
                            "cuobjdump_version": True},
        "cross_check_failures": [],
    }


def test_the_real_fixtures_pass_every_criterion():
    assert CENSUS.judge(_passing_record()) == []


def _set(path, value):
    def mutate(record):
        target = record
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return mutate


def _hgmma_and_mma_zero(record):
    record["sass"]["hgmma"] = 0
    record["ptx"]["mma_async"] = 0


def _no_allocation(record):
    record["sass"]["usetmaxreg"] = [m for m in record["sass"]["usetmaxreg"]
                                    if m["kind"] != "alloc"]


def _no_release(record):
    record["sass"]["usetmaxreg"] = [m for m in record["sass"]["usetmaxreg"]
                                    if m["kind"] != "release"]


@pytest.mark.parametrize("criterion,mutate", [
    ("stack-local", _set(("resource_usage", "STACK"), 512)),
    ("stack-local", _set(("resource_usage", "LOCAL"), 16)),
    ("spill", _set(("sass", "stl"), 1)),
    ("spill", _set(("sass", "ldl"), 2)),
    ("depbar", _set(("sass", "depbar"), {"0": 2})),
    ("depbar", _set(("sass", "depbar"), {"1": 1})),
    ("hgmma", _set(("sass", "hgmma"), 7)),
    ("positive-control", _hgmma_and_mma_zero),
    ("positive-control", _no_allocation),
    ("positive-control", _no_release),
    ("artifacts", _set(("checks", "manifest_verified"), False)),
    ("artifacts", _set(("checks", "ptx_decode_entries"), 0)),
    ("artifacts", _set(("checks", "decode_images"), 2)),
    ("identity", _set(("identity_checks", "package_in_tree"), False)),
    ("identity", _set(("identity_checks", "cute_dsl_arch"), "sm_90")),
    ("identity", _set(("identity_checks", "dsl_identity"), False)),
    ("identity", _set(("identity_checks", "nvdisasm_version"), False)),
    ("identity", _set(("identity_checks", "cuobjdump_version"), False)),
    ("cross-check", _set(("cross_check_failures",), ["sha256 differs"])),
])
def test_each_criterion_broken_on_its_own_fails_and_names_itself(criterion, mutate):
    record = _passing_record()
    mutate(record)
    failures = CENSUS.judge(record)
    assert len(failures) == 1 and failures[0].startswith(f"[{criterion}] "), failures


def _record_of_edited_listing(*edits) -> dict:
    """Return the passing record with its SASS measured again from the real decode excerpt,
    each (old, new) line of `edits` replaced."""
    text = _text("decode_i2.excerpt.sass")
    for old, new in edits:
        text = _replace_line(text, old, new)
    record = _passing_record()
    record["sass"] = CENSUS.measure_sass(text)
    return record


def test_a_release_to_another_register_count_fails_the_positive_control():
    record = _record_of_edited_listing((
        "/*0700*/                   USETMAXREG.DEALLOC.CTAPOOL 0x20 ;",
        "        /*0700*/                   USETMAXREG.DEALLOC.CTAPOOL 0x18 ;"))
    assert [(m["kind"], m["registers"]) for m in record["sass"]["usetmaxreg"]] == [
        ("release", 24), ("alloc", 104), ("alloc", 104)]
    assert CENSUS.judge(record) == [
        "[positive-control] no USETMAXREG that releases to 32 registers"]


def test_an_allocation_of_another_register_count_fails_and_records_no_consumer_ceiling():
    """Both allocations of the real excerpt are edited from 104 registers to 112, so the
    listing is a build at another split: the positive control fails on the missing allocation
    of 104, and neither region counts as a consumer's."""
    record = _record_of_edited_listing(*[
        (f"/*{address}*/                   USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0x68 ;",
         f"        /*{address}*/                   USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0x70 ;")
        for address in ("25e0", "5bd0")])
    assert [(m["kind"], m["registers"]) for m in record["sass"]["usetmaxreg"]] == [
        ("release", 32), ("alloc", 112), ("alloc", 112)]
    assert record["sass"]["consumer_ceilings"] == []
    assert CENSUS.judge(record) == [
        "[positive-control] no USETMAXREG that allocates 104 registers"]


def test_a_record_with_nothing_measured_fails_every_criterion():
    failures = CENSUS.judge({"label": "x", "errors": []})
    assert {f.split("]")[0].lstrip("[") for f in failures} == set(CENSUS.CRITERIA)
    assert all("not measured" in f for f in failures)


def test_a_recorded_error_is_a_failure_of_its_criterion():
    record = _passing_record()
    record["errors"] = ["[identity] no cuobjdump at /x/bin/cuobjdump; set CUDA_HOME"]
    assert CENSUS.judge(record) == record["errors"]


# ---------------------------------------------------------------------------
# --check.
# ---------------------------------------------------------------------------

def _run_record(cubin="c" * 64, sass="s" * 64, ptx="p" * 64, host="research-secure-27",
                identity=None, passing=True) -> dict:
    return {
        "cubin_sha256": [cubin], "normalization": CORE.NORMALIZATION_VERSION,
        "dsl_identity": identity or {"arch": "sm_90a", "packages": ["nvidia-cutlass-dsl==4.5.2"]},
        "environment": {"host": host, "cuda_home": "/usr/local/cuda-12.9",
                        "nvdisasm_version": ["Cuda compilation tools, release 12.9, V12.9.88"],
                        "cuobjdump_version": ["Cuda compilation tools, release 12.9, V12.9.82"]},
        "artifacts": {"normalization": CORE.NORMALIZATION_VERSION, "ptx_norm_sha256": ptx,
                      "sass_norm_sha256": sass, "cubin_sha256": [cubin],
                      "ptx_entries": ["e"], "sass_entries": ["e"]},
        "verdict": {"pass": passing, "failures": [] if passing else ["[spill] 1 STL"]},
    }


def _compare(run, ref, selected=None, only=False):
    return CENSUS.compare_runs(run, ref, selected=selected or set(run), only=only)


def test_identical_records_pass_the_check():
    lines, ok = _compare({"a": _run_record(), "b": _run_record()},
                         {"a": _run_record(), "b": _run_record()})
    assert ok and lines[-1] == "CUBIN: identical=2 differ=0 absent=0 of 2 reference configs"


@pytest.mark.parametrize("change,sass_equal,ptx_equal", [
    (dict(cubin="d" * 64), True, True),
    (dict(cubin="d" * 64, sass="t" * 64), False, True),
    (dict(cubin="d" * 64, sass="t" * 64, ptx="q" * 64), False, False),
])
def test_a_changed_cubin_fails_and_says_what_else_changed(change, sass_equal, ptx_equal):
    lines, ok = _compare({"a": _run_record(**change)}, {"a": _run_record()})
    assert not ok
    assert f"[a] CUBIN-DIFF sass_norm_equal={sass_equal} ptx_norm_equal={ptx_equal} " \
           f"raw_entries_equal=True" in lines


@pytest.mark.parametrize("side", ["this run", "the reference"])
def test_a_record_without_a_manifest_fails_even_when_its_own_cubin_hash_matches(side):
    """`main` verifies only the manifests a reference carries, so the comparison reads the
    cubin hashes of the verified manifests and never the record's own copy."""
    bare = {k: v for k, v in _run_record().items() if k != "artifacts"}
    run, ref = ({"a": bare}, {"a": _run_record()}) if side == "this run" \
        else ({"a": _run_record()}, {"a": bare})
    lines, ok = _compare(run, ref)
    assert not ok and f"[a] NO CUBIN in a verified manifest on {side}" in lines
    assert lines[-1] == "CUBIN: identical=0 differ=0 absent=0 of 1 reference configs"


def test_the_manifests_cubin_hashes_are_compared_and_not_the_records_copies():
    run = _run_record()
    run["cubin_sha256"] = ["d" * 64]
    lines, ok = _compare({"a": run}, {"a": _run_record()})
    assert ok and lines[-1] == "CUBIN: identical=1 differ=0 absent=0 of 1 reference configs"
    run = _run_record(cubin="d" * 64)
    run["cubin_sha256"] = ["c" * 64]
    lines, ok = _compare({"a": run}, {"a": _run_record()})
    assert not ok and lines[-1] == "CUBIN: identical=0 differ=1 absent=0 of 1 reference configs"


def test_a_reference_label_that_was_not_produced_fails():
    lines, ok = _compare({"a": _run_record()}, {"a": _run_record(), "b": _run_record()})
    assert not ok and "SELECTED BUT NOT PRODUCED: ['b']" in lines


def test_a_produced_label_absent_from_the_reference_fails():
    lines, ok = _compare({"a": _run_record(), "i2_nolock": _run_record()},
                         {"a": _run_record()})
    assert not ok and "[i2_nolock] NOT IN THE REFERENCE -- unverified" in lines


def test_the_only_intersection_is_enforced():
    ref = {"a": _run_record(), "b": _run_record()}
    lines, ok = _compare({"a": _run_record()}, ref, selected={"a"}, only=True)
    assert ok
    lines, ok = _compare({}, ref, selected={"a"}, only=True)
    assert not ok and "SELECTED BUT NOT PRODUCED: ['a']" in lines


@pytest.mark.parametrize("change,field", [
    (dict(host="research-secure-24"), "environment.host"),
    (dict(identity={"arch": "sm_90a", "packages": ["nvidia-cutlass-dsl==4.6.1"]}),
     "dsl_identity"),
])
def test_another_environment_is_not_comparable(change, field):
    lines, ok = _compare({"a": _run_record(**change)}, {"a": _run_record()})
    assert not ok and any(line.startswith(f"[a] NOT COMPARABLE: {field} ") for line in lines)


@pytest.mark.parametrize("path,value", [
    (("environment", "nvdisasm_version"), ["Cuda compilation tools, release 13.0, V13.0.48"]),
    (("environment", "cuobjdump_version"), ["Cuda compilation tools, release 13.0, V13.0.48"]),
    (("environment", "cuda_home"), "/usr/local/cuda-13.0"),
    (("normalization",), 1),
])
def test_another_tool_toolkit_or_normalization_is_not_comparable(path, value):
    run = _run_record()
    target = run
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    lines, ok = _compare({"a": run}, {"a": _run_record()})
    assert not ok and any(f"NOT COMPARABLE: {'.'.join(path)} " in line for line in lines)


def test_a_failing_record_of_this_run_fails_the_check():
    lines, ok = _compare({"a": _run_record(passing=False)}, {"a": _run_record()})
    assert not ok and "FAILING THE CRITERIA IN THIS RUN: ['a']" in lines


def test_a_version_1_reference_is_refused(tmp_path):
    reference = {"a": {k: v for k, v in _run_record().items() if k != "normalization"}}
    reference["a"]["sha256"] = "p" * 64
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(reference))
    with pytest.raises(SystemExit, match="another version"):
        CENSUS.load_records(str(path))


def test_a_reference_that_is_not_records_by_label_is_refused(tmp_path):
    path = tmp_path / "ref.json"
    path.write_text("[1, 2]")
    with pytest.raises(SystemExit, match="as records by label"):
        CENSUS.load_records(str(path))


def _fingerprint_record(digest="1" * 64) -> dict:
    """Return a record with the fields `scripts/_ptx_fingerprint_sm90.py` writes for one
    label."""
    return {"distinct_opcodes": 1, "entries": ["e"], "instructions": 1, "lines": 1,
            "n_modules": 1, "normalization": CORE.NORMALIZATION_VERSION, "opcodes": {"ret": 1},
            "sha256": digest, "sha256_structural": "2" * 64}


def test_a_fingerprint_output_and_a_failed_compile_are_accepted_as_fingerprint_records():
    CENSUS.require_fingerprint_output("fp.json", {"a": _fingerprint_record(),
                                                  "b": {"error": "rc=1: no PTX"}})


@pytest.mark.parametrize("record,message", [
    # a census record carries `sha256` and the stamp under the fingerprint's own names
    ({**_run_record(), "sha256": "1" * 64},
     "census fields ['cubin_sha256', 'verdict', 'artifacts']"),
    ({k: v for k, v in _fingerprint_record().items() if k != "sha256_structural"},
     "fingerprint fields missing ['sha256_structural']"),
    ([1, 2], "not a fingerprint record"),
])
def test_a_file_the_fingerprint_did_not_write_is_refused_as_a_fingerprint_output(record,
                                                                                message):
    with pytest.raises(SystemExit, match=re.escape(message)):
        CENSUS.require_fingerprint_output("fp.json", {"a": record})


def _write(path: Path, records: dict) -> str:
    path.write_text(json.dumps(records))
    return str(path)


def test_a_census_output_given_as_the_fingerprint_is_refused_before_any_compile(tmp_path):
    census = _write(tmp_path / "census.json",
                    {"i2_default": {**_passing_record(), **_run_record(), "sha256": "1" * 64}})
    save = tmp_path / "new.json"
    with pytest.raises(SystemExit, match="is not a record the fingerprint writes"):
        CENSUS.main(["--save", str(save), "--only", "i2_default", "--fingerprint", census])
    assert not Path(str(save) + ".artifacts").exists()


def test_a_reference_whose_artifacts_do_not_verify_is_refused_before_any_compile(tmp_path):
    fingerprint = _write(tmp_path / "fp.json", {"i2_default": _fingerprint_record()})
    reference = _write(tmp_path / "ref.json", {"i2_default": {
        **_run_record(), "artifacts": {"name": "i2_default"}}})
    save = tmp_path / "new.json"
    with pytest.raises(SystemExit, match=r"the reference's artifacts for i2_default do not "
                                         r"verify .*not a manifest this harness writes"):
        CENSUS.main(["--save", str(save), "--check", reference, "--only", "i2_default",
                     "--fingerprint", fingerprint])
    assert not Path(str(save) + ".artifacts").exists()


@pytest.mark.parametrize("existing", ["new.json", "new.json.artifacts"])
def test_a_save_path_an_earlier_run_owns_is_refused(tmp_path, existing):
    (tmp_path / existing).mkdir()
    with pytest.raises(SystemExit, match="already exists; pass a --save path this run can own"):
        CENSUS.main(["--save", str(tmp_path / "new.json"), "--fingerprint",
                     str(tmp_path / "fp.json")])


def test_the_parent_verifies_every_manifest_again_before_it_settles_the_verdict(tmp_path):
    """A child's record is judged only after its manifest is verified again from the files;
    one that no longer verifies fails `artifacts` whatever the child recorded."""
    kept = {**_passing_record(), "sha256": "1" * 64}
    kept.pop("cross_check_failures")
    moved = json.loads(json.dumps(kept))
    moved["artifacts"] = {"name": "moved"}
    records = {"kept": kept, "moved": moved}
    CENSUS.settle(records, tmp_path, {"fp.json": {"kept": _fingerprint_record(),
                                                  "moved": _fingerprint_record()}})
    assert records["kept"]["verdict"] == {"pass": True, "failures": []}
    assert records["moved"]["checks"]["manifest_verified"] is False
    failures = records["moved"]["verdict"]["failures"]
    assert failures[0].startswith("[artifacts] moved: not a manifest this harness writes")
    assert {f.split("]")[0] for f in failures} == {"[artifacts"}
    assert records["moved"]["cross_check_failures"] == []


@pytest.mark.parametrize("argv,message", [
    (["--save", "x.json", "--check", "", "--fingerprint", "f"], "empty path"),
    (["--list", "--check", "r.json"], "compares nothing"),
    (["--list", "--one", "i2_default"], "compares nothing"),
    (["--save", "x.json", "--one", "i2_default", "--check", "r.json"], "does not compare"),
    (["--save", "r.json", "--check", "r.json", "--fingerprint", "f"], "the same file"),
    (["--save", "x.json", "--check", "r.json", "--fingerprint", "r.json"],
     "is also this run's --save or --check"),
    (["--save", "x.json", "--fingerprint", "x.json"], "is also this run's --save or --check"),
    (["--save", "x.json"], "--fingerprint is required"),
    (["--check", "r.json"], "--save is required"),
    (["--list", "--only", "i2_default,nope"], "unknown config labels"),
])
def test_the_fingerprints_refusals_are_mirrored(argv, message):
    with pytest.raises(SystemExit, match=re.escape(message)):
        CENSUS.main(argv)


# ---------------------------------------------------------------------------
# The fingerprint cross-check.
# ---------------------------------------------------------------------------

def _cross(records, *sources):
    CENSUS.cross_check(records, {f"fp{i}.json": s for i, s in enumerate(sources)})
    return {label: record["cross_check_failures"] for label, record in records.items()}


def test_equal_hashes_pass_the_cross_check():
    assert _cross({"a": {"sha256": "1" * 64}, "b": {"sha256": "2" * 64}},
                  {"a": {"sha256": "1" * 64}}, {"b": {"sha256": "2" * 64}}) == {"a": [], "b": []}


def test_a_different_hash_fails_the_cross_check():
    assert _cross({"a": {"sha256": "1" * 64}}, {"a": {"sha256": "3" * 64}})["a"] == [
        f"sha256 {'1' * 16} is not the fingerprint's {'3' * 16}"]


def test_a_label_no_fingerprint_output_records_fails_the_cross_check():
    assert _cross({"a": {"sha256": "1" * 64}}, {"b": {"sha256": "1" * 64}})["a"] == [
        "no fingerprint output given records a"]


def test_a_conflicting_duplicate_fails_the_cross_check():
    failures = _cross({"a": {"sha256": "1" * 64}}, {"a": {"sha256": "1" * 64}},
                      {"a": {"sha256": "4" * 64}})["a"]
    assert failures and "two hashes" in failures[0]
    # the same hash in two outputs is not a conflict
    assert _cross({"a": {"sha256": "1" * 64}}, {"a": {"sha256": "1" * 64}},
                  {"a": {"sha256": "1" * 64}})["a"] == []


def test_a_fingerprint_record_without_a_hash_fails_the_cross_check():
    failures = _cross({"a": {"sha256": "1" * 64}}, {"a": {"error": "rc=1"}})["a"]
    assert failures == ["fp0.json records no hash (its compile failed)"]


# ---------------------------------------------------------------------------
# The child's identity checks and tools, run as the parent runs it.
# ---------------------------------------------------------------------------

def _tool(directory: Path, name: str, script: str) -> Path:
    tool = directory / "bin" / name
    tool.parent.mkdir(parents=True, exist_ok=True)
    tool.write_text("#!/bin/sh\n" + script)
    tool.chmod(0o755)
    return tool


def _cuda_home(tmp_path: Path, nvdisasm: str | None = None, cuobjdump: str | None = None):
    """Return a CUDA_HOME whose two tools print the real `--version` fixtures unless told
    otherwise."""
    home = tmp_path / "cuda"
    _tool(home, "nvdisasm", nvdisasm or f"cat '{DATA / 'nvdisasm.version.txt'}'\n")
    _tool(home, "cuobjdump", cuobjdump or f"cat '{DATA / 'cuobjdump.version.txt'}'\n")
    return home


#: Where the planted package claims to live, outside the tree, which is what an editable
#: install of another checkout makes `import cutedsl_decode` return.
FOREIGN = "/elsewhere/cutedsl-decode/src/cutedsl_decode/__init__.py"


def _child(tmp_path: Path, env: dict, *, foreign: bool = False) -> dict:
    """Return the record the census's `--one i2_default` child writes, in its own process,
    with only `env` beyond PATH, HOME and PYTHONPATH. With `foreign`, `cutedsl_decode` resolves outside
    the tree before the child imports it, so the child stops at its in-tree check and never
    imports the DSL, which this machine does not have."""
    save = tmp_path / "record.json"
    plant = ("import types\nm = types.ModuleType('cutedsl_decode')\n"
             f"m.__file__ = {FOREIGN!r}\nsys.modules['cutedsl_decode'] = m\n" if foreign
             else "")
    launcher = ("import runpy, sys\n" + plant +
                "sys.argv = sys.argv[1:]\nrunpy.run_path(sys.argv[0], run_name='__main__')\n")
    base = {k: os.environ[k] for k in ("PATH", "HOME", "PYTHONPATH") if k in os.environ}
    proc = subprocess.run(
        [sys.executable, "-c", launcher, str(SCRIPTS / "_sass_census_sm90.py"), "--save",
         str(save), "--one", "i2_default", "--artifacts", str(tmp_path / "artifacts")],
        env={**base, **env}, capture_output=True, text=True, timeout=300, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(save.read_text())


@pytest.mark.parametrize("env,error", [
    ({}, "[identity] CUTE_DSL_ARCH is (unset); the census judges the SM90 kernel"),
    ({"CUTE_DSL_ARCH": "sm_90"}, "[identity] CUTE_DSL_ARCH is sm_90; the census judges"),
    ({"CUTE_DSL_ARCH": "sm_90a"}, "[identity] CUDA_HOME is unset"),
])
def test_the_child_stops_on_the_architecture_and_the_toolkit(tmp_path, env, error):
    record = _child(tmp_path, env)
    assert len(record["errors"]) == 1 and record["errors"][0].startswith(error)
    assert "sass" not in record and "cutedsl_decode_file" not in record.get("environment", {})
    assert any(f.startswith("[identity] ") for f in CENSUS.judge(record))


@pytest.mark.parametrize("missing,error", [
    ("nvdisasm", "[identity] no nvdisasm at "), ("cuobjdump", "[identity] no cuobjdump at ")])
def test_the_child_stops_when_a_tool_is_absent(tmp_path, missing, error):
    home = _cuda_home(tmp_path)
    (home / "bin" / missing).unlink()
    record = _child(tmp_path, {"CUTE_DSL_ARCH": "sm_90a", "CUDA_HOME": str(home)})
    assert len(record["errors"]) == 1 and record["errors"][0].startswith(error)


def test_the_child_stops_when_the_package_resolves_outside_the_tree(tmp_path):
    home = _cuda_home(tmp_path)
    record = _child(tmp_path, {"CUTE_DSL_ARCH": "sm_90a", "CUDA_HOME": str(home)},
                    foreign=True)
    assert record["identity_checks"] == {"cute_dsl_arch": "sm_90a", "nvdisasm_version": True,
                                         "cuobjdump_version": True, "package_in_tree": False}
    assert record["environment"]["nvdisasm_version"] == CENSUS.release_lines(
        _text("nvdisasm.version.txt"))
    assert record["environment"]["cutedsl_decode_file"] == FOREIGN
    assert record["errors"] == [(f"[identity] cutedsl_decode resolves to {FOREIGN}, outside "
                                 f"{REPO / 'src'}; the census would judge another tree")]
    assert "config" not in record
    assert any(f.startswith("[identity] cutedsl_decode was imported from")
               for f in CENSUS.judge(record))


@pytest.mark.parametrize("nvdisasm,cuobjdump,field,error", [
    (f"grep -v release '{DATA / 'nvdisasm.version.txt'}'\n", None, "nvdisasm_version",
     "[identity] a --version output without naming its release"),
    (None, "echo broken >&2\nexit 3\n", "cuobjdump_version", "--version exited 3: broken"),
])
def test_a_tool_version_that_cannot_be_read_fails_identity(tmp_path, nvdisasm, cuobjdump,
                                                           field, error):
    home = _cuda_home(tmp_path, nvdisasm, cuobjdump)
    record = _child(tmp_path, {"CUTE_DSL_ARCH": "sm_90a", "CUDA_HOME": str(home)},
                    foreign=True)
    assert record["identity_checks"][field] is False and record["environment"][field] is None
    assert any(error in e for e in record["errors"])
    assert f"[identity] {field} was not read" in CENSUS.judge(record)


@pytest.mark.parametrize("script", ["echo 'cannot read' >&2\nexit 1\n", "exit 0\n"])
def test_a_resource_usage_run_that_fails_or_prints_nothing_is_refused(tmp_path, script):
    tool = _tool(tmp_path, "cuobjdump", script)
    with pytest.raises(CENSUS.CensusError, match=r"^\[stack-local\] cuobjdump -res-usage"):
        CENSUS.res_usage(str(tool), REAL_CUBIN)


def test_a_resource_usage_run_returns_what_the_tool_printed(tmp_path):
    tool = _tool(tmp_path, "cuobjdump", f"cat '{DATA / 'combine.res-usage.txt'}'\n")
    usage = CENSUS.parse_res_usage(CENSUS.res_usage(str(tool), REAL_CUBIN), COMBINE_ENTRY)
    assert usage["STACK"] == 0 and usage["LOCAL"] == 0


# ---------------------------------------------------------------------------
# The configuration list.
# ---------------------------------------------------------------------------

#: The fourteen D=256 rows: each admitted layout (stages 2 and 4) at both widths with the fused
#: combine off and on, at int2 also with the phase lock off, and the two int2 GQA rows.
D256_LABELS = {f"d256_i{bits}_st{stages}{lock}{fused}"
               for stages in (2, 4) for bits in (2, 4)
               for lock in (("", "_nolock") if bits == 2 else ("",))
               for fused in ("", "_fused")} | {"d256_i2_g8", "d256_i2_g2"}


def test_the_census_compiles_exactly_the_fingerprints_configurations():
    rows = CENSUS.select_rows(None)
    assert rows == FINGERPRINT.all_rows(False)
    assert len(rows) == 67
    at128 = [r for r in rows if r[2]["head_dim"] == ENVELOPE.HEAD_DIM]
    assert len(at128) == 53
    assert sum(1 for _, bits, _ in at128 if bits == 2) == 29
    assert sum(1 for _, bits, _ in at128 if bits == 4) == 24
    assert [r[0] for r in rows].count("i2_nolock") == 1
    nolock = next(r for r in rows if r[0] == "i2_nolock")
    assert nolock[1] == 2 and nolock[2]["phase_lock"] is False
    assert len(D256_LABELS) == 14
    assert {r[0] for r in rows if r[2]["head_dim"] == 256} == D256_LABELS


def test_every_row_carries_a_head_dimension_the_envelope_builds():
    rows = FINGERPRINT.all_rows(False) + FINGERPRINT.all_rows(True)
    assert {kw["head_dim"] for _, _, kw in rows} == {ENVELOPE.HEAD_DIM, 256}
    assert ENVELOPE.SM90_WIDE_HEAD_DIMS == (256,)
    # the D=128 labels name no head dimension, so older references compare label for label
    assert all(not label.startswith("d") for label, _, kw in rows
               if kw["head_dim"] == ENVELOPE.HEAD_DIM)
    # all_rows builds new dicts, so the module's CONFIGS stay as written
    assert all("head_dim" not in kw for _, _, kw in FINGERPRINT.CONFIGS)


def test_the_d256_rows_are_the_envelope_layouts_and_nothing_else():
    """Every D=256 row is a layout `sm90_wide_layouts(256)` admits, so the constructor's envelope
    and the census agree on what is built at D=256, and both admitted depths appear."""
    layouts = ENVELOPE.sm90_wide_layouts(256)
    keys = set(layouts[0])
    seen = set()
    for label, bits, kw in FINGERPRINT.all_rows(False):
        if kw["head_dim"] != 256:
            continue
        if label.startswith("d256_i2_g"):
            assert set(kw) == {"head_dim", "num_qo_heads", "num_kv_heads"}, label
            continue
        layout = {k: kw[k] for k in keys}
        assert layout in layouts, label
        seen.add(layout["num_stages"])
        assert kw.get("phase_lock", True) is (not label.count("_nolock")), label
        assert kw["fused_combine"] is label.endswith("_fused"), label
    assert seen == {2, 4}


def test_exactly_the_constructor_only_rows_are_constructed():
    constructed = {label for label, _, kw in FINGERPRINT.all_rows(False)
                   if FINGERPRINT.constructed_only(kw)}
    assert constructed == {"i2_foldoff", "i4_foldoff", "i2_nolock", "i2b128_default",
                           "i2b128_foldoff", "d256_i2_st2_nolock", "d256_i2_st2_nolock_fused",
                           "d256_i2_st4_nolock", "d256_i2_st4_nolock_fused"}


def test_row_config_keeps_the_row_and_refuses_one_without_a_head_dimension():
    torch = pytest.importorskip("torch")
    row = next(r for r in FINGERPRINT.all_rows(False) if r[0] == "flex_i2_K-to64_V-to128")
    cf = FINGERPRINT.row_config(row[1], row[2])
    assert cf["sdtype"] is torch.bfloat16 and cf["head_dim"] == ENVELOPE.HEAD_DIM
    assert (cf["num_qo_heads"], cf["num_kv_heads"], cf["page_size"]) == (32, 8, 64)
    g8 = next(r for r in FINGERPRINT.all_rows(False) if r[0] == "i2_g8")
    assert FINGERPRINT.row_config(g8[1], g8[2])["num_qo_heads"] == 64
    with pytest.raises(ValueError, match="carry no head_dim"):
        FINGERPRINT.row_config(2, {})


# ---------------------------------------------------------------------------
# Checks by syntax tree: what the GPU path runs.
# ---------------------------------------------------------------------------

def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    (node,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    return node


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and ast.unparse(n.func).split(".")[-1] == name]


def test_compile_one_sends_exactly_the_constructor_only_rows_to_compile_constructed():
    compile_one = _function(SCRIPTS / "_ptx_fingerprint_sm90.py", "compile_one")
    branches = [n for n in ast.walk(compile_one) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "constructed_only(cf)"]
    assert len(branches) == 1
    (branch,) = branches
    body, orelse = ast.Module(branch.body, []), ast.Module(branch.orelse, [])
    (constructed,) = _calls(body, "compile_constructed")
    assert ast.unparse(constructed.args[0]) == "construct_row(cf)"
    assert not _calls(body, "compile_pingpong") and not _calls(orelse, "compile_constructed")
    assert len(_calls(orelse, "compile_pingpong")) == 1
    # Every compile call in compile_one is one of those two.
    assert len(_calls(compile_one, "compile_constructed")) == 1
    assert len(_calls(compile_one, "compile_pingpong")) == 1


def test_construct_row_passes_compile_pingpongs_arguments_plus_the_constructor_only_ones():
    construct = _function(SCRIPTS / "_ptx_fingerprint_sm90.py", "construct_row")
    (ours,) = _calls(construct, "IntPingPongDecodeSm90")
    (theirs,) = _calls(_function(REPO / "src" / "cutedsl_decode" / "_sm90" /
                                 "pingpong_runtime.py", "compile_pingpong"),
                       "IntPingPongDecodeSm90")
    assert len(ours.args) == len(theirs.args) == 3
    assert ast.unparse(ours.args[2]) == ast.unparse(theirs.args[2]) == "PG"
    named = {k.arg for k in ours.keywords if k.arg is not None}
    assert named == {k.arg for k in theirs.keywords} | {"phase_lock"}
    assert not {"phase_lock", "int2_fast_mantissa_injection"} & {k.arg for k in theirs.keywords}
    # `int2_fast_mantissa_injection` is unpacked from a mapping that holds it only for a row
    # that sets it, so every other row is built with the constructor's own default.
    (unpacked,) = [k for k in ours.keywords if k.arg is None]
    (encoding,) = [n for n in ast.walk(construct) if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == ast.unparse(unpacked.value)]
    assert ast.unparse(encoding.value) == (
        "{'int2_fast_mantissa_injection': cf['int2_fast_mantissa_injection']} "
        "if 'int2_fast_mantissa_injection' in cf else {}")


def test_construct_row_hands_every_rows_values_to_the_constructor(monkeypatch):
    """`construct_row` is run for real on every row, with the kernel class and `PG` replaced
    by recorders, so a constructor-only keyword or a default that stopped taking the row's value
    shows here rather than as a GPU fingerprint that compiled another configuration."""
    torch = pytest.importorskip("torch")
    calls = []
    kernel = types.ModuleType("cutedsl_decode._sm90.int_decode_pingpong")
    kernel.IntPingPongDecodeSm90 = lambda *args, **kwargs: calls.append((args, kwargs))
    runtime = types.ModuleType("cutedsl_decode._sm90.pingpong_runtime")
    runtime.PG = object()
    monkeypatch.setitem(sys.modules, kernel.__name__, kernel)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    D = ENVELOPE.HEAD_DIM
    built = {}
    for label, bits, kw in FINGERPRINT.all_rows(False):
        calls.clear()
        FINGERPRINT.construct_row(FINGERPRINT.row_config(bits, kw))
        (built[label],) = calls
    # The defaults are compile_pingpong's, and the quantization is at the row's head dimension.
    assert built["i2_default"] == ((D, 64, runtime.PG), dict(
        num_stages=None, bits=2, sdtype=torch.float32, min_blocks_per_sm=2,
        fold_scale_zero=True, fused_combine=False, k_quant=("token", D),
        v_quant=("token", D), heads_per_kv=4, phase_lock=True))
    # `int2_fast_mantissa_injection` reaches the constructor from the two rows that set it and
    # from no other, so every other row keeps the constructor's default.
    b128 = {"i2b128_default", "i2b128_foldoff"}
    for label, bits, kw in FINGERPRINT.all_rows(False):
        args, got = built[label]
        row_d = kw["head_dim"]
        assert args == (row_d, 64, runtime.PG), label
        assert got == dict(
            num_stages=kw.get("num_stages"), bits=bits,
            sdtype=torch.bfloat16 if kw.get("sdtype") == "bf16" else torch.float32,
            min_blocks_per_sm=2, fold_scale_zero=kw.get("fold_scale_zero", True),
            fused_combine=kw.get("fused_combine", False),
            k_quant=kw.get("k_quant", ("token", row_d)),
            v_quant=kw.get("v_quant", ("token", row_d)),
            heads_per_kv=kw.get("num_qo_heads", 32) // kw.get("num_kv_heads", 8),
            phase_lock=kw.get("phase_lock", True),
            **({"int2_fast_mantissa_injection": False} if label in b128 else {})), label
    assert (built["i2_nolock"][1]["phase_lock"], built["i2_nolock"][1]["fold_scale_zero"]) \
        == (False, True)
    assert (built["i2_foldoff"][1]["phase_lock"], built["i4_foldoff"][1]["fold_scale_zero"]) \
        == (True, False)
    assert built["i2b128_default"] == ((D, 64, runtime.PG), {
        **built["i2_default"][1], "int2_fast_mantissa_injection": False})
    assert built["i2b128_foldoff"] == ((D, 64, runtime.PG), {
        **built["i2_foldoff"][1], "int2_fast_mantissa_injection": False})
    assert built["i2_g8"][1]["heads_per_kv"] == 8 and built["i2_g2"][1]["heads_per_kv"] == 2
    # Every row is at the envelope's head dimension, where a literal 128 in place of the
    # row's head_dim builds the same kernel, so one row at another head dimension shows that
    # the positional head dimension and the default quantization both follow the row.
    other = 2 * D
    calls.clear()
    FINGERPRINT.construct_row(FINGERPRINT.row_config(2, {"head_dim": other}))
    ((args, got),) = calls
    assert args == (other, 64, runtime.PG)
    assert got["k_quant"] == got["v_quant"] == ("token", other)


def test_row_keys_are_exactly_the_keys_construct_row_reads():
    construct = _function(SCRIPTS / "_ptx_fingerprint_sm90.py", "construct_row")
    read = {ast.literal_eval(n.slice) for n in ast.walk(construct)
            if isinstance(n, ast.Subscript) and ast.unparse(n.value) == "cf"}
    read |= {ast.literal_eval(n.args[0]) for n in _calls(construct, "get")
             if ast.unparse(n.func) == "cf.get"}
    assert read == FINGERPRINT.ROW_KEYS


@pytest.mark.parametrize("key", ["num_stage", "opt_level"])
def test_construct_row_refuses_a_key_it_does_not_read(key):
    """A misspelt keyword, or a `compile_pingpong` keyword that the constructor does not take,
    is refused, where dropping it would build `i2_default`."""
    cf = {"head_dim": ENVELOPE.HEAD_DIM, "num_qo_heads": 32, "num_kv_heads": 8, "bits": 2,
          "page_size": 64, key: 2}
    with pytest.raises(TypeError, match=rf"does not read \['{key}'\]"):
        FINGERPRINT.construct_row(cf)


def test_the_child_claims_the_dump_switches_before_importing_the_package():
    child = _function(SCRIPTS / "_sass_census_sm90.py", "run_child")
    (claim,) = _calls(child, "claim_dump_env")
    imports = [n for n in ast.walk(child) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imports and all(n.lineno > claim.lineno for n in imports)
    (compile_call,) = _calls(child, "compile_constructed")
    assert [ast.unparse(k) for k in compile_call.keywords] == [
        "head_dim=cf['head_dim']", "num_qo_heads=cf['num_qo_heads']",
        "num_kv_heads=cf['num_kv_heads']", "page_size=cf['page_size']"]
    assert ast.unparse(compile_call.args[0]) == "kern"


def test_the_census_imports_the_machine_code_library_rather_than_copying_it():
    tree = ast.parse((SCRIPTS / "_sass_census_sm90.py").read_text())
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in ("claim_dump_env", "find_nvdisasm", "carve_cubins", "disassemble",
                 "elf_symbol_names", "write_artifacts", "verify_manifest"):
        assert name not in defined
    (library,) = [n for n in tree.body if isinstance(n, ast.ImportFrom)
                  and n.module == "baselines._machine_code"]
    assert {a.name for a in library.names} == {
        "DECODE_TOKEN", "ELF64_HEADER_SIZE", "ELF64_SECTION_HEADER_SIZE", "claim_dump_env",
        "compare_manifests", "elf_symbol_names", "find_nvdisasm", "verify_manifest",
        "write_artifacts"}


def test_every_criterion_is_judged():
    judge = ast.unparse(_function(SCRIPTS / "_sass_census_sm90.py", "judge"))
    for criterion in CENSUS.CRITERIA:
        assert f"'{criterion}'" in judge



def test_the_reference_and_every_fingerprint_output_go_through_the_one_loader():
    main = _function(SCRIPTS / "_sass_census_sm90.py", "main")
    loads = [ast.unparse(call.args[0]) for call in _calls(main, "load_records")]
    assert sorted(loads) == ["args.check", "path"]
    assert not [n for n in ast.walk(main) if isinstance(n, ast.Call)
                and ast.unparse(n.func) in ("json.load", "json.loads")
                and "part" not in ast.unparse(n)]
