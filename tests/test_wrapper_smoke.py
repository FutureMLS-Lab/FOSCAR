import pytest
import torch

from fosca import QuantKVLayout, get_capabilities
from fosca.wrapper import PagedDecodeWrapper
from fosca._utils import is_supported

pytestmark = pytest.mark.skipif(not is_supported(), reason="needs SM90 or SM100 GPU")
DEV = torch.device("cuda")

# ---------------------------------------------------------------------------
# ARGUMENT-CONTRACT tests only, on the real device.
#
# These deliberately stop before any cute.compile: an IntDecodeSm100 trace is
# ~40 s, which does not belong in a unit-test suite. The numerical gate is a
# separate on-device run that compares the quantized path bitwise against the
# dequantized reference; `tests/test_public_api.py` holds the host-side plan()
# and run() contract. This file pins what needs the device itself: the
# architecture that a layout and a plan resolve from it.
# ---------------------------------------------------------------------------


def _ws():
    return PagedDecodeWrapper(torch.empty(1 << 20, dtype=torch.float32, device=DEV))


def _dummy_plan_args():
    b, kv = 2, 512
    npages = kv // 64
    indptr = torch.arange(b + 1, dtype=torch.int32, device=DEV) * npages
    indices = torch.arange(b * npages, dtype=torch.int32, device=DEV)
    seq_lens = torch.full((b,), kv, dtype=torch.int32, device=DEV)
    return indptr, indices, seq_lens


def test_the_layout_resolves_the_devices_reader():
    layout = QuantKVLayout.create(bits=2, head_dim=128, num_kv_heads=8)
    assert layout.arch == get_capabilities()["arch"]
    assert layout.reader == {"sm90": "IntPingPongDecodeSm90",
                             "sm100": "IntDecodeSm100"}[layout.arch]


def test_plan_refuses_the_retired_per_argument_form():
    with pytest.raises(TypeError, match="retired call form"):
        _ws().plan(*_dummy_plan_args(), 32, 8, 128, 64,
                   q_data_type=torch.float8_e4m3fn, bits=2)


def test_plan_refuses_a_layout_of_the_other_architecture():
    other = {"sm90": "sm100", "sm100": "sm90"}[get_capabilities()["arch"]]
    layout = QuantKVLayout._create(other, bits=2, head_dim=128, num_kv_heads=8)
    with pytest.raises(ValueError, match=f"resolved for {other}"):
        _ws().plan(*_dummy_plan_args(), 32, layout)


def test_capabilities_advertise_quantized_bits():
    caps = get_capabilities()
    assert caps["quantized_bits"] == [2, 4]
    assert caps["quantized_page_sizes"] == [64]
    assert caps["kv_layouts"] == ["QuantKVLayout"]
