# Copyright (c) 2026 Donglin Zhuang
# SPDX-License-Identifier: MIT
# tests/test_utils.py
import pytest
import torch
from foscar import _utils


def test_torch_to_cutlass_known_dtypes():
    cutlass = pytest.importorskip("cutlass", reason="requires nvidia-cutlass-dsl")
    assert _utils.torch_to_cutlass(torch.bfloat16) is cutlass.BFloat16
    assert _utils.torch_to_cutlass(torch.float16) is cutlass.Float16


def test_torch_to_cutlass_rejects_unsupported():
    pytest.importorskip("cutlass", reason="requires nvidia-cutlass-dsl")
    with pytest.raises(TypeError):
        _utils.torch_to_cutlass(torch.float64)


def test_api_decorator_is_transparent():
    @_utils.api
    def f(x, y=2):
        return x + y
    assert f(3) == 5
    assert f.__name__ == "f"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_capabilities_shape():
    caps = _utils.get_capabilities()
    assert set(["arch", "paged", "dtypes", "page_sizes", "reductions"]).issubset(caps)
    assert caps["arch"] in ("sm90", "sm100", "unsupported")
    assert _utils.get_num_sm(torch.device("cuda")) > 0
    # The advertised head dimensions are the running architecture's accepted set.
    if caps["arch"] != "unsupported":
        assert caps["head_dims"] == {"sm90": "64,128,256", "sm100": "64,128,256"}[caps["arch"]]
    else:
        assert caps["head_dims"] == ""
