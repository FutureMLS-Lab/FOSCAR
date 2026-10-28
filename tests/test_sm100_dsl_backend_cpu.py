"""The SM100 compile path refuses nvidia-cutlass-dsl's CUDA 12 backend.

That backend vendors ptxas 12.9, which left output elements of the SM100 kernel unwritten
on B200 under DSL 4.8.0 (2026-10-07); Blackwell runs the CUDA 13 backend (README,
"Install"). The guard is `fold_runtime._refuse_cuda12_dsl_backend`, called in
`compile_fold`'s trace closure before `cute.compile`.
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

    from cutedsl_decode._sm100 import fold_runtime

    monkeypatch.setattr(version_info, "CUDA_VERSION", _CudaVersion(major, 9))
    if refused:
        with pytest.raises(RuntimeError, match=r"^compile_fold: the SM100 kernel needs "
                                               r"nvidia-cutlass-dsl's CUDA 13 backend"):
            fold_runtime._refuse_cuda12_dsl_backend()
    else:
        fold_runtime._refuse_cuda12_dsl_backend()


def test_the_guard_runs_in_the_trace_before_the_compile():
    """A guard nothing calls protects nothing."""
    from cutedsl_decode._sm100 import fold_runtime

    src = inspect.getsource(fold_runtime.compile_fold)
    start = src.index("def _trace():")
    assert "_refuse_cuda12_dsl_backend()" in src[start:src.index("cute.compile(", start)]
