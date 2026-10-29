# Copyright 2026 Donglin Zhuang
# SPDX-License-Identifier: Apache-2.0
"""The capability descriptor and the import path.

The package must import on a machine with no CUDA toolkit, because the CuTe DSL
is imported only when a kernel is planned or compiled.
"""
import pytest
import torch

from fosca import _utils


def test_the_package_imports_without_the_dsl():
    import fosca

    assert isinstance(fosca.__version__, str)


def test_the_api_decorator_is_transparent():
    @_utils.api
    def f(x, y=2):
        return x + y

    assert f(3) == 5
    assert f.__name__ == "f"


def test_capabilities_always_return_the_full_key_set():
    caps = _utils.get_capabilities()
    assert set([
        "arch", "paged", "dtypes", "head_dims", "page_sizes", "reductions",
        "speculative", "atomic_reduction", "skip_softmax", "kv_layouts",
        "quantized_bits", "quantized_validated", "quantized_page_sizes",
    ]).issubset(caps)
    assert caps["arch"] in ("sm90", "sm100", "unsupported")


def test_an_unsupported_device_advertises_nothing():
    if _utils.is_supported():
        pytest.skip("this machine has a supported GPU")
    caps = _utils.get_capabilities()
    assert caps["arch"] == "unsupported"
    assert caps["paged"] is False
    assert caps["dtypes"] == []
    assert caps["head_dims"] == ""
    assert caps["quantized_bits"] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_supported_device_advertises_its_architecture():
    caps = _utils.get_capabilities()
    if caps["arch"] == "unsupported":
        pytest.skip("CUDA device is not SM90 or SM100")
    assert caps["head_dims"] == {"sm90": "128,256", "sm100": "128"}[caps["arch"]]
    assert caps["quantized_bits"] == [2, 4]
    assert caps["page_sizes"] == [64]
    assert _utils.get_num_sm(torch.device("cuda")) > 0
