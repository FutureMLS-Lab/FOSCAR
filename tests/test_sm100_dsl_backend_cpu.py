# Copyright (c) 2026 Donglin Zhuang
# SPDX-License-Identifier: MIT
"""The SM100 compile path refuses nvidia-cutlass-dsl's CUDA 12 backend.

That backend vendors ptxas 12.9, which left output elements of the SM100 kernel unwritten
on B200 under DSL 4.8.0 (2026-10-07); Blackwell runs the CUDA 13 backend (README,
"Install"). The guard is `fold_runtime._refuse_cuda12_dsl_backend`, called in
`compile_fold`'s trace closure before `cute.compile`, and in the Blackwell cache writer's
compile (`_append_core.compile_writer`), which also follows the decode's other DSL 4.8 rules.
"""
from __future__ import annotations

import importlib.util
import inspect

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None,
    reason="the guard reads the CuTe DSL's vendored CUDA version")


class _CudaVersion:
    def __init__(self, major, minor):
        self.major, self.minor = major, minor

    def __str__(self):
        return f"DSLCudaVersion(major={self.major}, minor={self.minor})"


@pytest.mark.parametrize("major, refused", [(12, True), (13, False)])
def test_the_sm100_compile_refuses_only_the_cuda12_backend(monkeypatch, major, refused):
    from cutlass.base_dsl import version_info

    from foscar._sm100 import fold_runtime

    monkeypatch.setattr(version_info, "CUDA_VERSION", _CudaVersion(major, 9))
    if refused:
        with pytest.raises(RuntimeError, match=r"^compile_fold: the SM100 kernel needs "
                                               r"nvidia-cutlass-dsl's CUDA 13 backend"):
            fold_runtime._refuse_cuda12_dsl_backend()
    else:
        fold_runtime._refuse_cuda12_dsl_backend()


def test_the_guard_runs_in_the_trace_before_the_compile():
    """A guard nothing calls protects nothing."""
    from foscar._sm100 import fold_runtime

    src = inspect.getsource(fold_runtime.compile_fold)
    start = src.index("def _trace():")
    assert "_refuse_cuda12_dsl_backend()" in src[start:src.index("cute.compile(", start)]


@pytest.mark.parametrize("prequantized", [False, True])
@pytest.mark.parametrize("arch", ["sm90", "sm100"])
@pytest.mark.parametrize("major", [12, 13])
def test_the_cache_writer_compiles_under_the_decode_rules(monkeypatch, arch, major, prequantized):
    """The cache writer's trace initializes the CUDA driver before it compiles; a Hopper
    writer compiles through `sm90_compiler`, whose options key the cached object; a Blackwell
    writer is refused under the CUDA 12 backend, naming its own compile, before anything is
    traced. The quantizing writer is traced through the writer object itself (`__call__`) and
    the prequantized one through its `call_prequantized`."""
    from cutlass.base_dsl import version_info

    from foscar import _append_core, kernel_cache
    from foscar._sm90 import pingpong_runtime

    calls = []

    def compile_or_load(*, compile_fn, options, **_kw):
        calls.append(("key", options))
        return compile_fn()

    def sm90_compiler(opt_level):
        return (lambda *args: calls.append(("sm90_compiler", args[0])) or "compiled",
                "sm90 options")

    def cute_compile(*args, **kwargs):
        calls.append(("cute.compile", args[0], kwargs))
        return "compiled"

    monkeypatch.setattr(kernel_cache, "init_cuda_driver", lambda: calls.append("init"))
    monkeypatch.setattr(kernel_cache, "compile_or_load", compile_or_load)
    monkeypatch.setattr(kernel_cache, "simple_attributes", lambda kern: {})
    monkeypatch.setattr(_append_core, "layout_key", lambda layout: {})
    monkeypatch.setattr(pingpong_runtime, "sm90_compiler", sm90_compiler)
    monkeypatch.setattr(_append_core.cute, "compile", cute_compile)
    monkeypatch.setattr(version_info, "CUDA_VERSION", _CudaVersion(major, 0))
    monkeypatch.setenv("CUTE_DSL_OPT_LEVEL", "3")
    kern = type("QuantAppendStub", (), {"ARCH": arch, "prequantized": prequantized,
                                        "call_prequantized": "the prequantized entry"})()
    entry = "the prequantized entry" if prequantized else kern

    if arch == "sm100" and major == 12:
        with pytest.raises(RuntimeError, match=r"^QuantAppendStub compile: the SM100 kernel "
                                               r"needs nvidia-cutlass-dsl's CUDA 13 backend"):
            _append_core.compile_writer(kern, _append_core.__file__, None, ())
        assert calls == [("key", "--enable-tvm-ffi --opt-level 3"), "init"]
        return
    assert _append_core.compile_writer(kern, _append_core.__file__, None, ()) == "compiled"
    if arch == "sm90":
        assert calls == [("key", "sm90 options"), "init", ("sm90_compiler", entry)]
    else:
        assert calls == [("key", "--enable-tvm-ffi --opt-level 3"), "init",
                         ("cute.compile", entry, {"options": "--enable-tvm-ffi --opt-level 3"})]
