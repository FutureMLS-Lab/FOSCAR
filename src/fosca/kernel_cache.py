# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: Apache-2.0
"""A persistent on-disk cache of compiled CuTe DSL kernels.

``compile_or_load`` is the entry point: it returns a kernel from disk when an
identical trace has already been exported, and compiles and exports it otherwise.
``init_cuda_driver`` initializes the driver API before a trace. ``status``,
``record_of``, ``provenance``, ``records`` and ``inventory`` report what happened
to each compile request.

WHY IT EXISTS. ``cute.compile`` forces ``compile_only=True`` and
``no_cache=True`` (``base_dsl/compiler.py``, ``CompileCallable._compile``), so
neither the DSL's in-memory JIT cache nor its file cache ever holds a
``cute.compile`` result, and the only memoisation in this repository is per
process. Tracing a kernel costs seconds, most of it Python-side IR generation,
which no lowered-module cache can skip, so a benchmark that forks one process per
measured configuration traces its kernels again in every process.

WHAT IS CACHED. The object file ``export_to_c`` writes for a kernel compiled with
``--enable-tvm-ffi``. Reloading it with ``cute.runtime.load_module(...,
enable_tvm_ffi=True)`` takes a few milliseconds and hands back a
``tvm_ffi.Function`` with the same positional calling convention as the freshly
compiled object. That is not an approximation: with ``--enable-tvm-ffi`` the DSL
already builds the in-process function by exporting the module to an object and
JIT-linking it, so the reloaded kernel runs the same machine code, with
bitwise-identical outputs on every buffer and kernel time within noise on both
architectures. flashinfer uses the same pair of calls and is the model for this
module.

THE KEY. A cache hit must be the object the trace would have produced, so the key
carries everything that shapes the trace: the DSL's own compile target
(``envar.arch``, read from the DSL singleton rather than re-derived from the torch
device), every installed ``nvidia-cutlass-dsl*`` distribution plus the vendored
CUDA toolkit version and backend library the DSL loaded, the sha256 of every
package source file the kernel module imports (computed from the import graph, not
maintained by hand), the options string exactly as passed to ``cute.compile``, the
constructed kernel's resolved attributes together with the trace arguments that are
not attributes, and the environment flags that change codegen without changing
the Python inputs. Equivalent defaults must produce equal keys, which is why
callers pass resolved attributes rather than raw keyword arguments.

WHEN IT STANDS ASIDE. Whenever the DSL is asked to dump PTX, cubin or IR
(``CUTE_DSL_KEEP*``, ``CUTE_DSL_DUMP_DIR``), to dry-run, or to print the trace-time
tiling report (``FOLD_TRACE_TILING``): those gates' evidence is the trace itself,
and the variable's presence is what counts, not its value. Whenever
``cutedsl_profiling`` has been imported, because an instrumented kernel bakes
session state into the object. Whenever ``FOSCA_KERNEL_CACHE=0``, which a
caller sets when its claim is that a configuration traces or that processes
compile independently. And whenever the cache root cannot be created or locked,
because a benchmark must not fail on a full or read-only disk. In every one of
those cases ``compile_or_load`` simply calls the compile function and records why
it did not cache.

WHAT A CALLER GETS BACK. On a hit, the reloaded function. On a miss, the freshly
compiled object is exported and then reloaded from disk, so hit and miss hand back
the same kind of object and an export fault shows up in the process that compiled.
Both happen under the per-key lock, so the hash a record carries is the hash of
the bytes that were loaded. If exporting or reloading fails the in-process object
is returned and the failure is recorded. ``record_of``, ``provenance``, ``records``
and ``inventory`` expose, per kernel, whether it was traced, loaded or left
uncached, with the object's sha256, so a benchmark artifact can say which.

WHAT THIS DOES NOT COVER. Callers that invoke ``cute.compile`` directly are not
cached; they compile concrete tensor shapes per configuration and would need to
compile through ``compile_constructed`` to benefit.
"""
from __future__ import annotations

import ast
import dataclasses
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Optional

#: ``0`` (or ``false`` / ``off`` / ``no``) turns the on-disk layer off for the process.
ENABLE_ENV = "FOSCA_KERNEL_CACHE"
#: Overrides the cache root. The default is ``$XDG_CACHE_HOME/fosca/kernel-cache``
#: or ``~/.cache/fosca/kernel-cache``. Put it on local disk on a box whose home
#: is NFS.
DIR_ENV = "FOSCA_KERNEL_CACHE_DIR"
#: Environment variables whose PRESENCE means the caller needs a real trace: the DSL
#: dump and dry-run knobs, which a check on the emitted PTX depends on, and the SM100
#: tiling report, which is printed from inside the trace.
BYPASS_ENV = (
    "CUTE_DSL_KEEP",
    "CUTE_DSL_KEEP_PTX",
    "CUTE_DSL_KEEP_CUBIN",
    "CUTE_DSL_KEEP_IR",
    "CUTE_DSL_DUMP_DIR",
    "CUTE_DSL_DRYRUN",
    "FOLD_TRACE_TILING",
)
#: Environment variables that change the emitted object without changing the Python
#: inputs of the trace. Their values are part of the key. The list is every variable that
#: the DSL's env manager marks `affects_compile=True`, except `CUTE_DSL_ARCH`, keyed
#: through `dsl_identity()`, and `CUTE_DSL_DRYRUN`, which is in BYPASS_ENV.
#: `CUTE_DSL_DEBUG` turns LINEINFO on by default; the others select lowering passes,
#: staging, diagnostics or emitted locations.
KEYED_ENV = (
    "CUTE_DSL_LINEINFO",
    "CUTE_DSL_ENABLE_ASSERTIONS",
    "CUTE_DSL_LIBS",
    "CUTE_DSL_COMPILER_OPT",
    "CUTE_DSL_DEBUG",
    "CUTE_DSL_ENABLE_PYIR",
    "CUTE_DSL_AUTO_M2S",
    "CUTE_DSL_TOLERATE_M2M",
    "CUTE_DSL_WARNINGS_AS_ERRORS",
    "CUTE_DSL_COMPILER_BACKEND",
    "CUTE_DSL_ENABLE_TVM_FFI",
    "CUTE_DSL_LOC_TRACEBACKS",
)
def init_cuda_driver() -> None:
    """Initialize the CUDA driver API before a kernel trace.

    `compile_fold`, `compile_constructed` and `compile_splitk_combine` call it first in
    their `_trace`. It lives here rather than in `_utils` because `_utils` imports the
    wrapper, and every module a trace imports joins the kernel's source closure.
    `IntAppend.__call__` and the standalone `splitk_combine.main` trace on their own and
    do not call it, so a caller that traces either before touching CUDA initializes CUDA
    itself.

    The DSL reads a device attribute through the driver API
    (`cuDeviceGet`) while it traces a kernel launch, and raises CUDA_ERROR_NOT_INITIALIZED
    in a process that has not touched CUDA yet, such as a warm-up compile issued before
    any tensor exists. `cuInit` is idempotent.
    """
    import cuda.bindings.driver as cuda_driver

    (err,) = cuda_driver.cuInit(0)
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit(0) failed with {err}")


#: The profiling package whose recording session bakes device pointers into the
#: kernels it instruments. Its presence in ``sys.modules`` is the bypass condition.
PROFILING_MODULE = "cutedsl_profiling"
#: Bumped whenever the sidecar layout or the key composition changes, so an old
#: artifact reads as a miss rather than as a hit with a different meaning.
FORMAT_VERSION = 2
#: The package this module belongs to; the source closure never leaves it.
PACKAGE = "fosca"
PACKAGE_DIR = Path(__file__).resolve().parent


@dataclasses.dataclass(frozen=True)
class KernelRecord:
    """What happened to one compile request, kept for provenance."""

    kind: str
    origin: str          # "traced" (compiled here, then reloaded), "loaded", or "uncached"
    reason: str          # why nothing was cached; empty for traced and loaded
    key: str             # sha256 hex of the key document; empty when uncached
    object_sha256: str   # sha256 of the object file; empty when uncached
    path: str            # the object file; empty when uncached
    symbol: str          # the exported TVM-FFI symbol; empty when uncached

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


#: Every compile request this process made, in order. Bounded by the number of
#: requests, which a benchmark process counts in dozens.
_RECORDS: list[KernelRecord] = []
#: The returned callables by identity. A callable that supports weak references is held
#: weakly and its entry disappears when the caller drops it; one that does not (some
#: extension types) is held strongly for the life of the process, so that ``record_of``
#: can never answer for a recycled ``id()``. Either way the records themselves stay in
#: ``_RECORDS``.
_BY_ID: dict[int, tuple[Any, KernelRecord]] = {}


# ---------------------------------------------------------------------------
# Switches
# ---------------------------------------------------------------------------


def default_root() -> Path:
    override = os.environ.get(DIR_ENV)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join("~", ".cache")
    return Path(base).expanduser() / "fosca" / "kernel-cache"


def status() -> dict:
    """Whether this process may use the disk cache, and why not if not.

    Read at every call rather than once at import, so a gate that sets
    ``FOSCA_KERNEL_CACHE=0`` after importing the package is still honoured.
    A bypass variable counts as set when it is present, whatever its value: an empty
    ``CUTE_DSL_KEEP`` still tells the DSL to keep something.
    """
    switch = os.environ.get(ENABLE_ENV, "1").strip().lower()
    root = str(default_root())
    if switch in ("0", "false", "off", "no"):
        return {"enabled": False, "reason": f"{ENABLE_ENV}={switch}", "root": root}
    for name in BYPASS_ENV:
        if name in os.environ:
            return {"enabled": False, "reason": f"{name} is set", "root": root}
    if PROFILING_MODULE in sys.modules:
        return {
            "enabled": False,
            "reason": f"{PROFILING_MODULE} is imported; instrumented kernels carry session state",
            "root": root,
        }
    return {"enabled": True, "reason": "", "root": root}


# ---------------------------------------------------------------------------
# Key ingredients
# ---------------------------------------------------------------------------


def dsl_identity() -> dict:
    """The compiler that will run: target arch, package stack, vendored toolkit, backend.

    ``envar.arch`` is what ``cute.compile`` compiles for. It is ``CUTE_DSL_ARCH`` when
    that was set before the DSL initialised, else the arch of device ordinal 0 at
    import; re-deriving it from the current torch device would key an artifact under
    an arch the DSL did not compile for. The package list alone does not identify the
    compiler either: the DSL selects a backend library at import and vendors its own
    ptxas, so the toolkit version and the backend basename are recorded too.
    Raises when any of these cannot be read, and the caller then compiles uncached
    rather than guessing.
    """
    from cutlass.cutlass_dsl import CuTeDSL  # lazy: CPU tests import this module

    dsl = CuTeDSL._get_dsl()
    packages = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions()
        if (dist.metadata["Name"] or "").lower().startswith("nvidia-cutlass-dsl")
    )
    if not packages:
        raise RuntimeError("no nvidia-cutlass-dsl distribution is installed")
    try:
        from cutlass.base_dsl import version_info

        cuda_version = str(getattr(version_info, "CUDA_VERSION"))
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"cannot read the DSL's vendored CUDA version: {error}") from error
    backend = importlib.import_module("cutlass._mlir._mlir_libs._cutlass_ir")
    return {
        "arch": str(dsl.envar.arch),
        "packages": packages,
        "cuda_version": cuda_version,
        "backend": os.path.basename(getattr(backend, "__file__", "") or ""),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_file(name: str) -> Optional[Path]:
    """The source file of a package-internal module name, or None when it is not one."""
    if not name.startswith(PACKAGE):
        return None
    parts = name.split(".")[1:]
    candidate = PACKAGE_DIR.joinpath(*parts) if parts else PACKAGE_DIR
    if candidate.is_dir():
        candidate = candidate / "__init__.py"
    else:
        candidate = candidate.with_suffix(".py")
    return candidate if candidate.is_file() else None


def _imports_of(path: Path) -> set[str]:
    """Absolute module names of every import statement in ``path``, package-internal or not."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.resolve().relative_to(PACKAGE_DIR)
    here = PACKAGE + "." + ".".join(rel.with_suffix("").parts)
    if path.name == "__init__.py":
        here = here[: -len(".__init__")]
        package_of_file = here
    else:
        package_of_file = here.rsplit(".", 1)[0]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = (
                    package_of_file.rsplit(".", node.level - 1)[0]
                    if node.level > 1
                    else package_of_file
                )
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            # `from pkg import name` names a SUBMODULE when `pkg/name.py` exists, and
            # then only that submodule is a dependency; `pkg/__init__.py` is one only
            # when an attribute is taken from it. Resolving every `from .. import x` to
            # the package init would drag `wrapper.py` into every kernel's key.
            for alias in node.names:
                if _module_file(f"{module}.{alias.name}") is not None:
                    names.add(f"{module}.{alias.name}")
                else:
                    names.add(module)
    return names


def source_closure(*starts) -> dict[str, str]:
    """sha256 of every package source file reachable by import from ``starts``.

    Follows ``import`` and ``from ... import`` statements (absolute and relative)
    through the package, so a helper the kernel body reaches through a vendored module
    is in the key without anybody maintaining a list. Files outside the package are
    the DSL's and torch's, which the identity and the environment cover. Keys are paths
    relative to the package directory.
    """
    seen: dict[str, str] = {}
    pending = [Path(p).resolve() for p in starts]
    while pending:
        path = pending.pop()
        rel = path.relative_to(PACKAGE_DIR).as_posix()
        if rel in seen:
            continue
        seen[rel] = _sha256_file(path)
        for name in _imports_of(path):
            target = _module_file(name)
            if target is not None:
                pending.append(target.resolve())
    return dict(sorted(seen.items()))


def _jsonable(value: Any) -> Any:
    """A deterministic JSON form for the values a key may carry.

    Simple scalars pass through. Tuples become lists, dicts are re-keyed as strings,
    classes become their qualified name, torch dtypes their string. Anything else is
    reduced to its type name only, so an object whose repr carries an address can
    never make two processes disagree about one configuration.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.dtype):
        return str(value)
    return {"type": type(value).__qualname__}


def simple_attributes(obj: Any) -> dict:
    """The constructed kernel's resolved attributes, as JSON.

    Every scalar, string, tuple, list and dict attribute is included by value; an
    attribute holding anything else is included by its type name only, so the key
    still records that it exists. Constructor arguments become attributes, so this
    covers the codegen inputs after the constructor has resolved its defaults, which is
    what makes ``None`` and the corresponding explicit default share a key.
    """
    out = {"__class__": f"{type(obj).__module__}.{type(obj).__qualname__}"}
    for name, value in sorted(vars(obj).items()):
        if name.startswith("__"):
            continue
        out[name] = _jsonable(value)
    return out


def make_key(*, kind: str, sources: dict[str, str], config: dict, options: str,
             identity: dict) -> tuple[str, dict]:
    """The key document and its sha256."""
    document = {
        "format": FORMAT_VERSION,
        "kind": kind,
        "identity": _jsonable(identity),
        "sources": dict(sorted(sources.items())),
        "options": options,
        "config": _jsonable(config),
        "env": {name: os.environ.get(name) for name in KEYED_ENV},
    }
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), document


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def _log(message: str) -> None:
    print(f"[kernel_cache] {message}", file=sys.stderr, flush=True)


def _load(path: Path, symbol: str) -> Any:
    import cutlass.cute as cute  # lazy: CPU tests import this module

    module = cute.runtime.load_module(str(path), enable_tvm_ffi=True)
    return getattr(module, symbol)


def _read_sidecar(path: Path) -> Optional[dict]:
    """The sidecar as a dict, or None when it is missing, unreadable or not a dict."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def _try_load(object_path: Path, sidecar_path: Path, key: str, arch: str, symbol: str,
              loader: Callable[[Path, str], Any]) -> Optional[tuple[Any, str]]:
    """The cached function and the sha256 of the bytes it was loaded from, or None.

    None whenever the artifact is incomplete, keyed differently, recorded for another
    arch, damaged, or unloadable. Every one of those reads as a miss; none raises.
    Callers hold the per-key lock, so the file cannot change between the hash and the
    load.
    """
    if not object_path.is_file():
        return None
    meta = _read_sidecar(sidecar_path)
    if meta is None or meta.get("format") != FORMAT_VERSION or meta.get("key") != key:
        return None
    if meta.get("symbol") != symbol:
        return None
    document = meta.get("key_document")
    identity = document.get("identity") if isinstance(document, dict) else None
    recorded_arch = identity.get("arch") if isinstance(identity, dict) else None
    if recorded_arch != arch:
        _log(f"refusing {object_path.name}: recorded arch {recorded_arch!r}, DSL arch {arch!r}")
        return None
    try:
        actual = _sha256_file(object_path)
    except OSError:
        return None
    if actual != meta.get("object_sha256"):
        _log(f"refusing {object_path.name}: object sha256 differs from its sidecar")
        return None
    try:
        return loader(object_path, symbol), actual
    except Exception as error:  # noqa: BLE001
        _log(f"could not load {object_path.name} ({type(error).__name__}: {error}); recompiling")
        return None


def _note(fn: Any, record: KernelRecord) -> Any:
    _RECORDS.append(record)
    key = id(fn)
    try:
        holder: Any = weakref.ref(fn, lambda _ref, key=key: _BY_ID.pop(key, None))
    except TypeError:
        holder = fn
    _BY_ID[key] = (holder, record)
    return fn


def _held(holder: Any) -> Any:
    return holder() if isinstance(holder, weakref.ref) else holder


class _FileLock:
    """An exclusive advisory lock on one path, released on exit however it exits."""

    def __init__(self, path: Path):
        self.path = path
        self._fd: Optional[int] = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def __exit__(self, *exc):
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
        return False


def _uncached(kind: str, compile_fn: Callable[[], Any], reason: str) -> Any:
    return _note(compile_fn(), KernelRecord(kind, "uncached", reason, "", "", "", ""))


def compile_or_load(*, kind: str, compile_fn: Callable[[], Any], sources: dict[str, str],
                    config: dict, options: str,
                    _identity: Optional[Callable[[], dict]] = None,
                    _loader: Optional[Callable[[Path, str], Any]] = None) -> Any:
    """Return the compiled kernel, from disk when an identical trace has been exported.

    ``kind`` names the kernel family and the on-disk subdirectory. ``sources`` is
    ``source_closure(...)`` of the kernel module. ``config`` carries the resolved
    kernel attributes and the trace arguments. ``options`` is the string that will be
    passed to ``cute.compile``; it must enable TVM-FFI, because that is the only compiled
    form ``export_to_c`` writes as a single loadable object. The two underscore
    parameters exist so the CPU tests can stand in for the DSL.

    A compile error propagates: it is the caller's error, not the cache's. A cache
    error (unwritable root, unlockable lock file, failed export or reload) never does;
    the kernel is compiled or returned uncached and the reason is recorded.
    """
    kind = _sanitize(kind)
    state = status()
    if not state["enabled"]:
        return _uncached(kind, compile_fn, state["reason"])
    if "--enable-tvm-ffi" not in options.split():
        return _uncached(kind, compile_fn,
                         "options do not enable TVM-FFI, so the object cannot be exported")
    try:
        identity = (_identity or dsl_identity)()
    except Exception as error:  # noqa: BLE001
        return _uncached(kind, compile_fn,
                         f"DSL identity unavailable ({type(error).__name__}: {error})")

    key, document = make_key(kind=kind, sources=sources, config=config, options=options,
                             identity=identity)
    root = Path(state["root"]) / kind
    object_path = root / f"{key}.o"
    sidecar_path = root / f"{key}.json"
    symbol = f"fosca_{kind}_{key[:16]}"
    loader = _loader or _load

    try:
        lock = _FileLock(root / f"{key}.lock").__enter__()
    except OSError as error:
        return _uncached(kind, compile_fn,
                         f"cache root unavailable ({type(error).__name__}: {error})")
    try:
        hit = _try_load(object_path, sidecar_path, key, identity["arch"], symbol, loader)
        if hit is not None:
            fn, object_sha256 = hit
            _log(f"loaded {kind} {key[:16]} from {object_path}")
            return _note(fn, KernelRecord(kind, "loaded", "", key, object_sha256,
                                          str(object_path), symbol))
        started = time.perf_counter()
        compiled = compile_fn()
        traced_s = time.perf_counter() - started
        tmp_object = root / f"{key}.o.tmp.{os.getpid()}"
        tmp_sidecar = root / f"{key}.json.tmp.{os.getpid()}"
        try:
            compiled.export_to_c(str(tmp_object), function_name=symbol)
            object_sha256 = _sha256_file(tmp_object)
            os.replace(tmp_object, object_path)
            meta = {
                "format": FORMAT_VERSION,
                "kind": kind,
                "key": key,
                "key_document": document,
                "symbol": symbol,
                "object_sha256": object_sha256,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "trace_seconds": round(traced_s, 3),
                "exporter": {"python": sys.version.split()[0], "executable": sys.executable},
            }
            with open(tmp_sidecar, "w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=1, sort_keys=True)
            os.replace(tmp_sidecar, sidecar_path)
            fn = loader(object_path, symbol)
        except Exception as error:  # noqa: BLE001
            for tmp in (tmp_object, tmp_sidecar):
                try:
                    tmp.unlink()
                except OSError:
                    pass
            _log(f"traced {kind} {key[:16]} in {traced_s:.1f} s but could not persist or reload it "
                 f"({type(error).__name__}: {error}); using the in-process object")
            return _note(compiled, KernelRecord(
                kind, "uncached", f"export or reload failed: {type(error).__name__}: {error}",
                key, "", "", symbol))
        _log(f"traced {kind} {key[:16]} in {traced_s:.1f} s and stored it at {object_path}")
        return _note(fn, KernelRecord(kind, "traced", "", key, object_sha256, str(object_path),
                                      symbol))
    finally:
        lock.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def record_of(fn: Any) -> Optional[KernelRecord]:
    """The record of a callable ``compile_or_load`` returned, or None for anything else."""
    entry = _BY_ID.get(id(fn))
    if entry is None or _held(entry[0]) is not fn:
        return None
    return entry[1]


def provenance(fn: Any) -> Optional[dict]:
    """``record_of(fn)`` as a JSON-ready dict, or None for a callable this module never saw."""
    record = record_of(fn)
    return None if record is None else record.as_dict()


def records() -> list[dict]:
    """Every compile request this process made, in order."""
    return [record.as_dict() for record in _RECORDS]


def inventory() -> dict:
    """What a benchmark artifact records: the switch state and every kernel's origin.

    This is the worker's compilation and loading activity. It does not say which row
    used which kernel; a cell's variants share kernels through the in-process memo, and
    the rows carry only whether the cache was on.
    """
    state = status()
    return {
        "enabled": state["enabled"],
        "reason": state["reason"],
        "root": state["root"],
        "kernels": records(),
    }
