# Copyright (c) 2026 Donglin Zhuang
# SPDX-License-Identifier: MIT
"""The hardcoded e4m3 code tables equal torch's float8_e4m3fn bytes.

`int_fold_decode.py` carries `_E4M3_CODES`, the e4m3 byte of every raw code
q = 0..15, hardcoded so the kernel module stays torch-free. This check pins
them to torch on the CPU, reading the tuple out of the SOURCE so it needs no
cutlass.

Run: pytest tests/test_e4m3_code_table_cpu.py
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_SM100 = pathlib.Path(__file__).resolve().parents[1] / "src" / "foscar" / "_sm100"


def _codes(rel):
    """`_E4M3_CODES` as the module SOURCE spells it, parsed rather than imported."""
    tree = ast.parse((_SM100 / rel).read_text())
    node = next(n for n in tree.body
                if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "_E4M3_CODES" for t in n.targets))
    return list(ast.literal_eval(node.value))


def test_the_code_table_is_torch_float8_e4m3fn():
    torch = pytest.importorskip("torch")
    want = torch.arange(16, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8)
    assert _codes("int_fold_decode.py") == want.tolist()


def test_the_int2_k_table_is_e4m3_of_8_plus_q():
    """`LUT2C8` is `_E4M3_CODES[8:12]`: e4m3(8+q) for q in 0..3 must be 0x50..0x53,
    which is what `map_int2_k`'s PRMT gather and `map_int2_v`'s OR-magic assume."""
    torch = pytest.importorskip("torch")
    got = (torch.arange(4).float() + 8.0).to(torch.float8_e4m3fn).view(torch.uint8)
    assert got.tolist() == [0x50, 0x51, 0x52, 0x53]
    assert _codes("int_fold_decode.py")[8:12] == [0x50, 0x51, 0x52, 0x53]
