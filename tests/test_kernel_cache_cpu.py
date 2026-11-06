# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""The persistent kernel cache: its key, its bypasses and its store, without a GPU.

WHAT A CACHE CAN GET WRONG, and which test pins each. A hit that hands back the wrong
binary is the failure that matters, and it has three causes: two traces that should
differ share a key (the perturbation tests), a trace that should be served from disk is
rebuilt from a stale or corrupt artifact (the store tests), and a caller whose evidence is
the trace itself is served a hit instead (the bypass tests). A fourth failure is the cache
failing the benchmark: an unwritable root or a damaged sidecar must compile uncached, not
raise (the robustness tests). The DSL is stood in for by a fake compile function whose
`export_to_c` writes bytes, and a fake loader that reads them back; what these tests do not
exercise is the real `export_to_c` / `load_module` round trip and the DSL identity, which
require a GPU and the real DSL.

THE REQUIRED NAMES ARE SPELLED OUT HERE, not imported from the module: a test that derives
its cases from the implementation's own constant would pass with a variable removed from
both.
"""
from __future__ import annotations

import fcntl
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="kernel_cache serialises torch dtypes")

# Loaded by file, not through the package: `foscar/__init__.py` imports cutlass,
# which a CPU box does not have, and the cache module itself imports the DSL only inside
# the functions the fakes below stand in for.
_SPEC = importlib.util.spec_from_file_location(
    "kernel_cache_under_test",
    Path(__file__).resolve().parents[1] / "src" / "foscar" / "kernel_cache.py",
)
kc = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = kc          # dataclasses resolves the class's module by name
_SPEC.loader.exec_module(kc)

PKG = Path(kc.__file__).resolve().parent
IDENTITY = {"arch": "sm_100a", "packages": ["nvidia-cutlass-dsl==4.5.2"],
            "cuda_version": "DSLCudaVersion(major=12, minor=9)", "backend": "_cutlass_ir.so"}
OPTIONS = "--enable-tvm-ffi --opt-level 2"
CONFIG = {"kernel": {"__class__": "K", "bits": 2, "code_stages": 16, "sdtype": "f32",
                     "quant": ["token", 128]},
          "trace": {"num_qo_heads": 32, "reduction": "none", "dtype": torch.bfloat16},
          "flags": {"_FLAG": False}}
SOURCES = {"a.py": "0" * 64, "b.py": "1" * 64}

#: The variables whose PRESENCE must make the cache stand aside, spelled out.
REQUIRED_BYPASS = ("CUTE_DSL_KEEP", "CUTE_DSL_KEEP_PTX", "CUTE_DSL_KEEP_CUBIN",
                   "CUTE_DSL_KEEP_IR", "CUTE_DSL_DUMP_DIR", "CUTE_DSL_DRYRUN",
                   "FOLD_TRACE_TILING")
#: The variables whose VALUE must be part of the key, spelled out.
REQUIRED_KEYED = ("CUTE_DSL_LINEINFO", "CUTE_DSL_ENABLE_ASSERTIONS", "CUTE_DSL_LIBS",
                  "CUTE_DSL_COMPILER_OPT", "CUTE_DSL_DEBUG", "CUTE_DSL_ENABLE_PYIR",
                  "CUTE_DSL_AUTO_M2S", "CUTE_DSL_TOLERATE_M2M", "CUTE_DSL_WARNINGS_AS_ERRORS",
                  "CUTE_DSL_COMPILER_BACKEND", "CUTE_DSL_ENABLE_TVM_FFI",
                  "CUTE_DSL_LOC_TRACEBACKS")


class FakeCompiled:
    """What `cute.compile` returns, reduced to the one method the cache uses."""

    def __init__(self, payload: bytes):
        self.payload = payload
        self.exported_as = None

    def export_to_c(self, path, function_name=None):
        assert function_name, "export_to_c needs the symbol name on DSL 4.5.2"
        self.exported_as = function_name
        Path(path).write_bytes(self.payload)


class FakeLoaded:
    def __init__(self, payload: bytes, symbol: str):
        self.payload = payload
        self.symbol = symbol


def fake_loader(path: Path, symbol: str):
    return FakeLoaded(Path(path).read_bytes(), symbol)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """A fresh cache root, the switch on, no bypass variable set, no profiler imported."""
    monkeypatch.setenv(kc.DIR_ENV, str(tmp_path / "root"))
    monkeypatch.delenv(kc.ENABLE_ENV, raising=False)
    for name in REQUIRED_BYPASS + REQUIRED_KEYED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(sys.modules, kc.PROFILING_MODULE, raising=False)
    monkeypatch.setattr(kc, "_RECORDS", [])
    monkeypatch.setattr(kc, "_BY_ID", {})
    return tmp_path / "root"


def _run(compile_payload=b"OBJ", **overrides):
    calls = []

    def compile_fn():
        calls.append(1)
        return FakeCompiled(compile_payload)

    kwargs = dict(kind="sm100_K", compile_fn=compile_fn, sources=SOURCES, config=CONFIG,
                  options=OPTIONS, _identity=lambda: IDENTITY, _loader=fake_loader)
    kwargs.update(overrides)
    return kc.compile_or_load(**kwargs), calls


# --- the key ---------------------------------------------------------------------------


def _key(**overrides):
    parts = dict(kind="sm100_K", sources=SOURCES, config=CONFIG, options=OPTIONS,
                 identity=IDENTITY)
    parts.update(overrides)
    return kc.make_key(**parts)[0]


def test_equivalent_documents_share_a_key():
    """Dict order and tuple-versus-list spelling must not split the cache."""
    reordered = {"trace": dict(reversed(list(CONFIG["trace"].items()))),
                 "flags": CONFIG["flags"], "kernel": dict(CONFIG["kernel"])}
    reordered["kernel"]["quant"] = ("token", 128)
    assert _key() == _key(config=reordered)
    assert _key() == _key(sources=dict(reversed(list(SOURCES.items()))))


@pytest.mark.parametrize("perturb", [
    ("kind", "sm100_K2"),
    ("options", "--enable-tvm-ffi --opt-level 3"),
    ("sources", {"a.py": "0" * 64, "b.py": "2" * 64}),
    ("sources", {"a.py": "0" * 64, "b.py": "1" * 64, "c.py": "3" * 64}),
    ("identity", {**IDENTITY, "arch": "sm_90a"}),
    ("identity", {**IDENTITY, "packages": ["nvidia-cutlass-dsl==4.6.1"]}),
    ("identity", {**IDENTITY, "cuda_version": "DSLCudaVersion(major=13, minor=1)"}),
    ("identity", {**IDENTITY, "backend": "_cutlass_ir.cu13.so"}),
    ("config", {**CONFIG, "kernel": {**CONFIG["kernel"], "code_stages": 8}}),
    ("config", {**CONFIG, "kernel": {**CONFIG["kernel"], "extra": True}}),
    ("config", {**CONFIG, "trace": {**CONFIG["trace"], "reduction": "kernel"}}),
    ("config", {**CONFIG, "trace": {**CONFIG["trace"], "dtype": torch.float16}}),
    ("config", {**CONFIG, "flags": {"_FLAG": True}}),
])
def test_every_ingredient_moves_the_key(perturb):
    field, value = perturb
    assert _key(**{field: value}) != _key()


@pytest.mark.parametrize("name", REQUIRED_KEYED)
def test_each_codegen_variable_moves_the_key(monkeypatch, name):
    for other in REQUIRED_KEYED:
        monkeypatch.delenv(other, raising=False)
    base = _key()
    monkeypatch.setenv(name, "1")
    assert _key() != base
    assert name in kc.KEYED_ENV


def test_unkeyed_environment_does_not_move_the_key(monkeypatch):
    for name in REQUIRED_KEYED:
        monkeypatch.delenv(name, raising=False)
    base = _key()
    monkeypatch.setenv("CUTE_DSL_LOG_TO_CONSOLE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert _key() == base


def test_objects_with_addresses_cannot_split_the_key():
    """An attribute whose repr carries a memory address reduces to its type name."""

    class Layout:
        pass

    class Kern:
        def __init__(self, layout):
            self.bits = 2
            self.layout = layout
            self.dtype = torch.float32
            self.klass = Layout
            self.__private = 1  # noqa: F841  (dunder-mangled, skipped)

    a = kc.simple_attributes(Kern(Layout()))
    b = kc.simple_attributes(Kern(Layout()))
    assert a == b
    assert set(a["layout"]) == {"type"} and a["layout"]["type"].endswith("Layout")
    assert a["dtype"] == "torch.float32"
    assert a["klass"].endswith("Layout")
    assert a["__class__"].endswith("Kern")


# --- the source closure ----------------------------------------------------------------


def test_source_closure_follows_package_imports():
    fold = PKG / "_sm100" / "int_fold_decode.py"
    closure = kc.source_closure(fold, PKG / "_sm100" / "fold_runtime.py")
    for expected in ("_sm100/int_fold_decode.py", "_sm100/int_fold_asm.py",
                     "_sm100/gqa_decode.py", "_sm100/fold_runtime.py",
                     "kernel_contract.py", "quant_layout.py"):
        assert expected in closure, expected
    assert all(len(v) == 64 for v in closure.values())
    assert "wrapper.py" not in closure, "the trace does not import the wrapper"


def test_source_closure_covers_the_sm90_kernel_and_its_vendored_helpers():
    ping = PKG / "_sm90" / "int_decode_pingpong.py"
    closure = kc.source_closure(ping, PKG / "_sm90" / "pingpong_runtime.py")
    for expected in ("_sm90/int_decode_pingpong.py", "_sm90/pingpong_runtime.py",
                     "kernel_contract.py", "_vendor/quack/sm90_utils.py",
                     "_vendor/flash_attn/cute/copy_utils.py"):
        assert expected in closure, expected
    combine = kc.source_closure(PKG / "_sm90" / "splitk_combine.py",
                                PKG / "_sm90" / "pingpong_runtime.py")
    assert "_sm90/splitk_combine.py" in combine


def test_source_closure_changes_when_a_source_byte_changes(monkeypatch):
    """The closure is over file bytes, so an uncommitted edit is a different key."""
    fold = PKG / "_sm100" / "int_fold_decode.py"
    before = kc.source_closure(fold)
    original = kc._sha256_file
    monkeypatch.setattr(kc, "_sha256_file",
                        lambda p: "f" * 64 if p.name == "quant_layout.py" else original(p))
    after = kc.source_closure(fold)
    assert before != after
    assert before.keys() == after.keys()


# --- the store -------------------------------------------------------------------------


def test_miss_then_hit_round_trip(isolated):
    first, calls = _run()
    assert calls == [1]
    rec = kc.record_of(first)
    assert rec.origin == "traced" and rec.reason == ""
    assert isinstance(first, FakeLoaded), "a miss hands back the reloaded object too"
    assert first.symbol == rec.symbol == f"foscar_sm100_K_{rec.key[:16]}"
    obj = Path(rec.path)
    side = obj.with_suffix(".json")
    assert obj.read_bytes() == b"OBJ" and side.is_file()
    meta = json.loads(side.read_text())
    assert meta["key"] == rec.key and meta["object_sha256"] == rec.object_sha256
    assert meta["key_document"]["identity"]["arch"] == "sm_100a"
    assert not list(obj.parent.glob("*.tmp.*")), "temp files are cleaned up"

    second, calls2 = _run()
    assert calls2 == [], "a hit compiles nothing"
    rec2 = kc.record_of(second)
    assert rec2.origin == "loaded"
    assert rec2.object_sha256 == rec.object_sha256 == kc._sha256_file(obj)
    assert second.payload == b"OBJ"
    assert [r["origin"] for r in kc.records()] == ["traced", "loaded"]


def test_lock_is_released_after_a_hit_and_after_a_miss(isolated):
    first, _ = _run()
    lock_path = Path(kc.record_of(first).path).with_suffix(".lock")
    _run()
    fd = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)   # raises if still held
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def test_corrupt_object_is_a_miss(isolated):
    first, _ = _run(compile_payload=b"GOOD")
    Path(kc.record_of(first).path).write_bytes(b"BAD ")
    second, calls = _run(compile_payload=b"GOOD")
    assert calls == [1], "a hash mismatch recompiles rather than loading"
    assert kc.record_of(second).origin == "traced"
    assert second.payload == b"GOOD"


def test_sidecar_with_another_arch_is_refused(isolated):
    first, _ = _run()
    side = Path(kc.record_of(first).path).with_suffix(".json")
    meta = json.loads(side.read_text())
    meta["key_document"]["identity"]["arch"] = "sm_90a"
    side.write_text(json.dumps(meta))
    _, calls = _run()
    assert calls == [1]


@pytest.mark.parametrize("damage", [
    lambda meta: "",                                   # empty file
    lambda meta: "not json",                           # unparsable
    lambda meta: "[]",                                 # valid JSON, wrong container
    lambda meta: '"bad"',                              # valid JSON, a string
    lambda meta: json.dumps({"format": 1}),            # missing everything else
    # The two below keep the VALID key and symbol, so the read reaches the nested checks
    # rather than stopping at the key comparison.
    lambda meta: json.dumps({**meta, "key_document": []}),
    lambda meta: json.dumps({**meta, "key_document": {**meta["key_document"], "identity": []}}),
])
def test_damaged_sidecar_is_a_miss_not_an_error(isolated, damage):
    first, _ = _run()
    side = Path(kc.record_of(first).path).with_suffix(".json")
    side.write_text(damage(json.loads(side.read_text())))
    second, calls = _run()
    assert calls == [1]
    assert kc.record_of(second).origin == "traced"


def test_missing_sidecar_is_a_miss(isolated):
    first, _ = _run()
    Path(kc.record_of(first).path).with_suffix(".json").unlink()
    _, calls = _run()
    assert calls == [1]


def test_unloadable_object_falls_back_to_compiling(isolated):
    first, _ = _run()

    def broken_loader(path, symbol):
        raise OSError("JITLink refused")

    second, calls = _run(_loader=broken_loader)
    assert calls == [1]
    rec = kc.record_of(second)
    # The fresh compile could be exported but not reloaded, so the caller gets the
    # in-process object and the record says the cache did not serve it.
    assert rec.origin == "uncached" and "reload" in rec.reason
    assert isinstance(second, FakeCompiled)


def test_export_failure_returns_the_in_process_object_and_leaves_nothing(isolated):
    class HalfWritten(FakeCompiled):
        def export_to_c(self, path, function_name=None):
            Path(path).write_bytes(b"PART")      # a partial object, then the failure
            raise RuntimeError("no export provider")

    fn, calls = _run(compile_fn=lambda: HalfWritten(b"X"))
    assert calls == [] and isinstance(fn, HalfWritten)
    rec = kc.record_of(fn)
    assert rec.origin == "uncached" and "export" in rec.reason
    leftovers = [p for p in Path(isolated).rglob("*") if p.is_file() and p.suffix != ".lock"]
    assert leftovers == [], f"partial artifacts left behind: {leftovers}"


def test_unwritable_root_compiles_uncached(isolated):
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    blocked = Path(isolated)
    blocked.mkdir(parents=True)
    blocked.chmod(0o500)
    try:
        fn, calls = _run()
    finally:
        blocked.chmod(0o700)
    assert calls == [1] and isinstance(fn, FakeCompiled)
    rec = kc.record_of(fn)
    assert rec.origin == "uncached" and "cache root unavailable" in rec.reason


def test_compile_errors_propagate(isolated):
    def boom():
        raise ValueError("the trace itself failed")

    with pytest.raises(ValueError, match="trace itself"):
        kc.compile_or_load(kind="k", compile_fn=boom, sources=SOURCES, config=CONFIG,
                           options=OPTIONS, _identity=lambda: IDENTITY, _loader=fake_loader)


def test_different_configs_get_different_artifacts(isolated):
    a, _ = _run()
    b, _ = _run(config={**CONFIG, "trace": {**CONFIG["trace"], "reduction": "kernel"}})
    assert kc.record_of(a).path != kc.record_of(b).path
    assert kc.record_of(a).symbol != kc.record_of(b).symbol


# --- the bypasses ----------------------------------------------------------------------


def test_switch_off_compiles_uncached(isolated, monkeypatch):
    monkeypatch.setenv(kc.ENABLE_ENV, "0")
    fn, calls = _run()
    assert calls == [1] and isinstance(fn, FakeCompiled)
    rec = kc.record_of(fn)
    assert rec.origin == "uncached" and kc.ENABLE_ENV in rec.reason
    assert not list(Path(isolated).rglob("*")), "nothing is written while disabled"


@pytest.mark.parametrize("name", REQUIRED_BYPASS)
@pytest.mark.parametrize("value", ["ptx", ""])
def test_each_dump_and_report_variable_bypasses_on_presence(isolated, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    state = kc.status()
    assert not state["enabled"] and name in state["reason"]
    fn, calls = _run()
    assert calls == [1] and kc.record_of(fn).origin == "uncached"
    assert name in kc.BYPASS_ENV


def test_profiler_import_bypasses(isolated, monkeypatch):
    monkeypatch.setitem(sys.modules, kc.PROFILING_MODULE, object())
    state = kc.status()
    assert not state["enabled"] and kc.PROFILING_MODULE in state["reason"]
    fn, calls = _run()
    assert calls == [1] and kc.record_of(fn).origin == "uncached"


def test_options_without_tvm_ffi_are_not_cached(isolated):
    fn, calls = _run(options="--opt-level 3")
    assert calls == [1]
    assert "TVM-FFI" in kc.record_of(fn).reason


def test_identity_failure_compiles_uncached(isolated):
    def no_identity():
        raise RuntimeError("no DSL here")

    fn, calls = _run(_identity=no_identity)
    assert calls == [1]
    rec = kc.record_of(fn)
    assert rec.origin == "uncached" and "identity" in rec.reason


# --- provenance ------------------------------------------------------------------------


def test_inventory_reports_switch_and_kernels(isolated):
    _run()
    inv = kc.inventory()
    assert inv["enabled"] is True and inv["reason"] == ""
    assert inv["root"] == str(isolated)
    assert [k["origin"] for k in inv["kernels"]] == ["traced"]
    assert kc.record_of(object()) is None
    assert kc.provenance(object()) is None


def test_dropped_callables_do_not_pin_memory_but_records_survive(isolated):
    fn, _ = _run()
    assert kc.provenance(fn)["origin"] == "traced"
    del fn
    gc.collect()
    assert kc._BY_ID == {}, "a weak-referenceable callable is released with its holder"
    assert [r["origin"] for r in kc.records()] == ["traced"], "the history stays"
