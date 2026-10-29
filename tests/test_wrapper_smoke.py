# tests/test_wrapper_smoke.py
import math
import pytest
import torch

from fosca.wrapper import PagedDecodeWrapper
from fosca._utils import is_supported

pytestmark = pytest.mark.skipif(not is_supported(), reason="needs SM90 or SM100 GPU")
DEV = torch.device("cuda")

# ---------------------------------------------------------------------------
# ARGUMENT-CONTRACT tests only.
#
# These deliberately stop before any cute.compile, because an IntDecodeSm100
# trace is ~40 s, which does not belong in a unit-test suite. This file pins the
# errors a caller can hit without a GPU-minute; the numerical bitwise gate is a
# separate run.
# ---------------------------------------------------------------------------


def _ws():
    return PagedDecodeWrapper(torch.empty(1 << 20, dtype=torch.float32, device=DEV))


def _dummy_plan_args(page_size=64):
    b, kv = 2, 512
    npages = kv // page_size
    indptr = torch.arange(b + 1, dtype=torch.int32, device=DEV) * npages
    indices = torch.arange(b * npages, dtype=torch.int32, device=DEV)
    seq_lens = torch.full((b,), kv, dtype=torch.int32, device=DEV)
    return indptr, indices, seq_lens


@pytest.mark.parametrize("bits", [1, 3, 8])
def test_quant_rejects_unsupported_bits(bits):
    indptr, indices, seq_lens = _dummy_plan_args()
    with pytest.raises(ValueError, match="bits must be one of"):
        _ws().plan(indptr, indices, seq_lens, 32, 8, 128, 64,
                   q_data_type=torch.float8_e4m3fn, bits=bits)


def test_quant_rejects_non_64_page_size():
    indptr, indices, seq_lens = _dummy_plan_args(page_size=32)
    with pytest.raises(ValueError, match="page_size == 64"):
        _ws().plan(indptr, indices, seq_lens, 32, 8, 128, 32,
                   q_data_type=torch.float8_e4m3fn, bits=2)


def test_plan_requires_bits():
    # Omitting `bits` is an error rather than a silent fall-through to a bf16
    # kernel.
    indptr, indices, seq_lens = _dummy_plan_args()
    with pytest.raises(ValueError, match="bits=2 or bits=4"):
        _ws().plan(indptr, indices, seq_lens, 32, 8, 128, 64,
                   q_data_type=torch.float8_e4m3fn)


def test_quant_flex_args_rejected_without_bits():
    # ORDERING TEST, not a duplicate of the one above: the k_quant/v_quant guard
    # must keep firing BEFORE the missing-bits error, so a caller who passes
    # flex args but forgets `bits` is told the specific thing that is wrong.
    indptr, indices, seq_lens = _dummy_plan_args(page_size=16)
    with pytest.raises(ValueError, match="quantized-mode"):
        _ws().plan(indptr, indices, seq_lens, 32, 8, 128, 16,
                   q_data_type=torch.bfloat16, k_quant=("channel", 128))


def test_capabilities_advertise_quantized_bits():
    from fosca import get_capabilities

    caps = get_capabilities()
    assert caps["quantized_bits"] == [2, 4]
    assert caps["quantized_page_sizes"] == [64]
