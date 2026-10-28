"""The contention probe must reach the GPU torch is actually using.

Torch numbers devices WITHIN `CUDA_VISIBLE_DEVICES`, so under
`CUDA_VISIBLE_DEVICES=7` the device torch calls 0 is physical GPU 7. Passing the
torch ordinal to `nvidia-smi -i` interrogates the wrong card: that bug refused a
75-cell B200 sweep because a stray process sat on physical GPU 0 while the GPU
being measured was idle.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_bench_core_smi_id", REPO / "baselines" / "_bench_core.py"
)
assert SPEC is not None and SPEC.loader is not None
CORE = importlib.util.module_from_spec(SPEC)
sys.modules["_bench_core_smi_id"] = CORE
SPEC.loader.exec_module(CORE)

smi_id = CORE.nvidia_smi_id


def test_ordinal_maps_through_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    assert smi_id(0) == "7"


def test_second_visible_device_maps_to_its_physical_index(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    assert smi_id(0) == "4"
    assert smi_id(1) == "6"


def test_unset_variable_leaves_the_ordinal_alone(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert smi_id(3) == "3"


def test_a_uuid_is_passed_through_because_nvidia_smi_accepts_one(monkeypatch):
    uuid = "GPU-8dff9bf3-654e-aebb-7118-7b35c61037e9"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuid)
    assert smi_id(0) == uuid


def test_whitespace_and_empty_entries_are_ignored(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " 5 , ")
    assert smi_id(0) == "5"


def test_an_ordinal_outside_the_visible_set_raises(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    with pytest.raises(RuntimeError, match="outside CUDA_VISIBLE_DEVICES"):
        smi_id(1)
