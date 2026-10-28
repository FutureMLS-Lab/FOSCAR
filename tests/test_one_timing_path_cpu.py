"""There is one timing path under `baselines/`, and its exception list is exact.

`CLAUDE.md` states the rule (`_bench_core.measure()` is the only way to time
anything that gets published) and then lists the files that deliberately open a
clock or a profiler capture of their own. That list is only worth having if it is
EXHAUSTIVE: an enforcement pass built on an incomplete list flags required work as
a violation, which is exactly what happened to
`_dynamic_integrity_worker.py::_capture_trace`, whose per-decode capture collects
the device-operation trace a dynamic-output column cannot be published without.

So the list is re-derived from the source here rather than maintained by hand.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BASELINES = REPO / "baselines"
GUIDANCE = REPO / "CLAUDE.md"

#: Code that reads a clock or opens a capture. Deliberately matched as CODE and
#: not as prose: several adapters cite `torch.profiler` in a comment to say where
#: their launch count came from, which is documentation rather than a measurement.
TIMING_PATTERNS = (
    "from torch.profiler import",
    "torch.profiler.profile(",
    "Event(enable_timing=True)",
    "perf_counter(",
    "process_time(",
)
#: `_bench_core.py` IS the timing path, so it is not an exception to it.
THE_TIMING_PATH = "_bench_core.py"


def _section(text: str, heading: str) -> str:
    """The body of one `## ...` section of a markdown file."""
    start = text.index(heading)
    rest = text.index("\n## ", start + len(heading))
    return text[start:rest]


def _tracked_baselines() -> set[Path]:
    """The `baselines/*.py` files git actually has.

    THE RULE BINDS THE COMMITTED TREE, and deriving the list from the working
    directory instead made this gate depend on whoever's uncommitted work happened
    to be on disk. Concretely, on 2026-08-06 three untracked `reconstruction_roof_*`
    harnesses made the derived count seven while a clean checkout of the same commit
    would derive four, so the prose could not be true in both places at once. An
    untracked file cannot falsify a documented rule about the repository; it becomes
    the rule's problem when it lands, which is what the note in CLAUDE.md is for.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z", "baselines"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {
        REPO / name
        for name in out.split("\0")
        if name.endswith(".py")
    }


def _timing_files() -> list[Path]:
    found = []
    for path in sorted(_tracked_baselines()):
        if path.name == THE_TIMING_PATH:
            continue
        source = path.read_text(encoding="utf-8")
        if any(pattern in source for pattern in TIMING_PATTERNS):
            found.append(path)
    return found


def _exception_list_body(section: str) -> str:
    """The bullet block under the deliberate-exceptions heading, and only that."""
    start = section.index("deliberate exception")
    tail = section[start:]
    # The list ends at the next bold paragraph, which is the violations note.
    stop = tail.find("\n**")
    return tail if stop == -1 else tail[:stop]


#: A canonical entry, and nothing else counts as one. Anchored so that a bullet
#: whose filename is bolded, quoted, or bare is a GRAMMAR ERROR rather than an
#: invisible entry: codex bypassed an earlier parser with
#: `- **reconstruction_roof_hbm.py** is fine actually.`, which set equality then
#: ignored entirely.
#: CommonMark permits up to THREE leading spaces before a list marker, so an
#: indented bullet is still a bullet to every renderer. Codex hid
#: `   - **hidden.py** is fine actually.` from an earlier version that matched only
#: a line starting exactly `- `.
_BULLET = re.compile(r"^ {0,3}- ")
_ENTRY = re.compile(r"^ {0,3}- `([A-Za-z_][A-Za-z0-9_]*\.py)(?:::\w+)?`")


def _exception_entries(section: str) -> list[str]:
    """Filenames cited by the exceptions bullet list, in order, with duplicates.

    Returns a LIST so a caller can enforce uniqueness; an earlier version returned
    a set, which made the docstring's "unique" claim unenforceable.
    """
    body = _exception_list_body(section)
    entries = []
    for line in body.splitlines():
        if not _BULLET.match(line):
            continue
        match = _ENTRY.match(line)
        if match is None:
            raise AssertionError(
                f"exception-list bullet does not use the canonical "
                f"`- \u0060file.py::symbol\u0060` form, so it would be an "
                f"invisible entry: {line!r}"
            )
        entries.append(match.group(1))
    return entries


def test_every_exception_bullet_uses_the_canonical_form():
    """A non-canonical bullet must be an error, not a silently ignored line."""
    section = _section(
        GUIDANCE.read_text(encoding="utf-8"),
        "## Every kernel number is GPU kernel time",
    )
    entries = _exception_entries(section)
    assert entries, "the exceptions list is empty"
    assert len(entries) == len(set(entries)), (
        f"an exception is listed more than once: {entries}"
    )


def test_the_guidance_names_every_file_that_times_something_itself():
    section = _section(
        GUIDANCE.read_text(encoding="utf-8"),
        "## Every kernel number is GPU kernel time",
    )
    named = {
        path.relative_to(BASELINES).as_posix(): path.name in section
        for path in _timing_files()
    }
    assert named and all(named.values()), (
        f"files under baselines/ that open their own clock or capture, and "
        f"whether CLAUDE.md's exception list names them: {named}"
    )


def test_the_guidance_does_not_name_an_exception_that_no_longer_times_anything():
    """Every EXCEPTION entry must still open a clock.

    Scoped to the exceptions list via `_exception_entries`, not to the whole
    section: the violations note below it also uses bullets, and those name
    untracked files deliberately, so a whole-section scan read them as false
    exception entries.
    """
    section = _section(
        GUIDANCE.read_text(encoding="utf-8"),
        "## Every kernel number is GPU kernel time",
    )
    times_something = {path.name for path in _timing_files()}
    cited = {
        name
        for name in _exception_entries(section)
        if name != THE_TIMING_PATH
    }
    assert cited <= times_something, (
        f"CLAUDE.md lists {sorted(cited - times_something)} as a timing "
        f"exception, but nothing in those tracked files opens a clock or a "
        f"capture any more; drop the entry rather than leaving a false claim"
    )


def test_the_adapter_contract_cites_only_mechanisms_that_exist():
    """`adapters/base.py` describes what the harness enforces, by name.

    It used to promise an `_bench_core.audit_escape` containment audit that had
    been deleted, which is worse than promising nothing: a reader who believes it
    thinks a second stream is policed. Every `_bench_core.<symbol>` the contract
    names must therefore resolve in `_bench_core.py`.
    """
    contract = (BASELINES / "adapters/base.py").read_text(encoding="utf-8")
    core = (BASELINES / THE_TIMING_PATH).read_text(encoding="utf-8")
    cited = set(re.findall(r"_bench_core\.(\w+)", contract))
    assert cited, "the contract no longer says which mechanism enforces what"

    def defined(name: str) -> bool:
        return bool(
            re.search(rf"^(?:def|class) {name}\b", core, re.MULTILINE)
            or re.search(rf"^{name}\s*[:=]", core, re.MULTILINE)
        )

    assert not defined("audit_escape")          # the resolver actually resolves
    missing = sorted(name for name in cited if not defined(name))
    assert missing == [], f"{missing} do not exist in _bench_core.py"
    for gone in ("audit_escape", "audit_containment", "profile_kernel_us",
                 "flush_cost_us", "escape_ratio", "offstream"):
        assert gone not in contract, f"the contract still promises {gone}"
        assert gone not in core, f"_bench_core still defines {gone}"


def test_the_cudagraph_check_reports_each_measurement_once():
    """The one sanctioned event bracket compares replay against kernel time ONCE.

    It used to print `kernel_us` under two headings, one of them labelled `event`,
    after the event bracket and the profiler pass became the same number -- so a
    reader saw one measurement presented as two agreeing ones.
    """
    source = (BASELINES / "check_cudagraph_parity.py").read_text(encoding="utf-8")
    header = re.search(r"print\(f\"\{'column':14s\}.*?\)\n", source, re.S).group(0)
    assert header.count("kernel_us") == 1, header
    assert "graph_us" in header and "replay-gap" in header
    ratio_header = re.search(r"print\(f\"\{'B':>4s\}.*?\)\n", source, re.S).group(0)
    assert "'event'" not in ratio_header, ratio_header
    assert "'kernel'" in ratio_header and "'graph'" in ratio_header


#: Number words the prose is allowed to count in.
_WORDS = {
    "one": 1, "One": 1, "Two": 2, "Three": 3, "Four": 4,
    "Five": 5, "Six": 6, "Seven": 7, "Eight": 8, "Nine": 9,
}


def test_the_exception_list_is_exactly_the_tracked_self_timing_set():
    """Exhaustive, unique, and counted — asserted as set equality.

    The earlier version compared only a COUNT against the derived total, so a
    bullet could disappear while its filename lingered in prose and every test
    still passed. This parses the exception bullet list on its own and requires it
    to equal the tracked self-timing set exactly, then checks the prose count
    agrees with that same set.
    """
    section = _section(
        GUIDANCE.read_text(encoding="utf-8"),
        "## Every kernel number is GPU kernel time",
    )
    derived = {path.name for path in _timing_files()}
    listed = set(_exception_entries(section))
    assert listed == derived, (
        f"the deliberate-exceptions list and the tracked files that open their own "
        f"clock disagree. Only listed: {sorted(listed - derived)}. Only derived: "
        f"{sorted(derived - listed)}"
    )
    match = re.search(r"\*\*(\w+) deliberate exceptions?", section)
    assert match, "the prose no longer says how many deliberate exceptions there are"
    assert _WORDS[match.group(1)] == len(derived), (
        f"prose says {match.group(1)} exceptions but {len(derived)} tracked files "
        f"open their own clock: {sorted(derived)}"
    )


def test_an_uncommitted_self_timing_file_is_recorded_as_a_violation_not_ignored():
    """Untracked work does not falsify the rule, but it must not vanish either.

    An untracked file under `baselines/` that opens its own clock is outside the
    counted list above by construction; the section must still name it, or landing
    it later would look permitted. The untracked `reconstruction_roof_*` harnesses
    that prompted this check on 2026-08-06 have since been removed.
    """
    section = _section(
        GUIDANCE.read_text(encoding="utf-8"),
        "## Every kernel number is GPU kernel time",
    )
    on_disk_untracked = sorted(
        path.name
        for path in BASELINES.rglob("*.py")
        if path not in _tracked_baselines()
        and path.name != THE_TIMING_PATH
        and any(
            pattern in path.read_text(encoding="utf-8")
            for pattern in TIMING_PATTERNS
        )
    )
    for name in on_disk_untracked:
        assert name in section, (
            f"{name} is an uncommitted file under baselines/ that opens its own "
            f"clock, and the timing section does not mention it; name it so that "
            f"committing it later cannot look sanctioned"
        )
    # And it must NOT be in the counted exception list, or it would be laundered.
    listed = _exception_entries(section)
    laundered = sorted(set(on_disk_untracked) & set(listed))
    assert not laundered, (
        f"{laundered} are uncommitted violations listed as sanctioned exceptions"
    )
