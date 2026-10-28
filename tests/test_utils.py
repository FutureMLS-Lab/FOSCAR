# tests/test_utils.py
import pytest
import torch
from cutedsl_decode import _utils


def test_torch_to_cutlass_known_dtypes():
    import cutlass
    assert _utils.torch_to_cutlass(torch.bfloat16) is cutlass.BFloat16
    assert _utils.torch_to_cutlass(torch.float16) is cutlass.Float16


def test_torch_to_cutlass_rejects_unsupported():
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
        assert caps["head_dims"] == {"sm90": "128,256", "sm100": "128"}[caps["arch"]]
    else:
        assert caps["head_dims"] == ""
