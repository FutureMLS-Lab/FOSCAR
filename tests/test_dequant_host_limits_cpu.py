# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: MIT
"""The host limits of `QuantKVCache.dequantize` keep its kernels' 32-bit indices in range.

The dequantize counts requests and grid CTAs in 32-bit integers. Its checking CTA c checks request
`c * THREADS + tidx`, so over the launch's ceil(B / THREADS) checking CTAs the last index it forms
is up to THREADS - 1 past the last request; its request search forms indices up to 31 past it,
and its step from a span of requests plus 32. The launch has one CTA per page entry and KV head
plus the checking CTAs. `kv_cache._dequant_size_refusal` refuses on the host, from the tensor
shapes, every call for which one of those would pass 2^31 - 1, bounding the checking CTAs with
the smaller thread count (`DEQUANT_MIN_CHECK_REQUESTS`). This file checks the refusal at its
boundaries and holds both kernels' thread counts to the request limit and to that bound, reading
them from the kernel source because importing the kernels imports the DSL. It does not run the
device check: it only matches the call sites that pass each kernel's THREADS to `_validate` and
to its launch, so a check that skips a chunk of requests passes here and is caught by the GPU
serving round-trip tests.

Run: pytest tests/test_dequant_host_limits_cpu.py
"""
import ast
import importlib.util
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "src/foscar"
KERNELS = (PKG / "_sm90/quant_dequant.py", PKG / "_sm100/quant_dequant.py")
INT32_MAX = 2 ** 31 - 1


def _load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def kv(monkeypatch):
    # kv_cache imports `_utils`, which imports the DSL; it is not reached by what is tested here.
    pkg = types.ModuleType("foscar")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "foscar", pkg)
    monkeypatch.setitem(sys.modules, "foscar._utils",
                        types.ModuleType("foscar._utils"))
    _load(monkeypatch, "foscar.quant_layout", PKG / "quant_layout.py")
    return _load(monkeypatch, "foscar.kv_cache", PKG / "kv_cache.py")


def _threads(path):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and [t.id for t in node.targets] == ["THREADS"]:
            return node.value.value
    raise AssertionError(f"{path} sets no THREADS")


def test_request_limit(kv):
    limit = kv.MAX_DEQUANT_REQUESTS
    assert kv._dequant_size_refusal(limit, 0, 8) is None
    assert "requests" in kv._dequant_size_refusal(limit + 1, 0, 8)
    assert "requests" in kv._dequant_size_refusal(2 ** 31 - 1, 0, 8)


def _checking(requests, threads):
    return -(-requests // threads)


@pytest.mark.parametrize("requests", [1, 128, 129, 2 ** 20 + 1, 2 ** 31 - 512])
@pytest.mark.parametrize("heads", [1, 8, 33])
def test_entry_limit(kv, heads, requests):
    """The last call the refusal admits fills the grid bound, and one more entry is refused."""
    checking = _checking(requests, kv.DEQUANT_MIN_CHECK_REQUESTS)
    last = (INT32_MAX - checking) // heads     # entries * heads + checking <= 2^31 - 1
    assert kv._dequant_size_refusal(requests, last, heads) is None
    assert "2^31 - 1 in all" in kv._dequant_size_refusal(requests, last + 1, heads)


@pytest.mark.parametrize("kernel", KERNELS, ids=["sm90", "sm100"])
def test_kernel_indices_stay_32_bit(kv, kernel):
    """At the request limit, the checking CTAs' last index, the search's largest sampled index
    and its step's numerator all fit in 32 bits; at every call the refusal admits, the kernel's
    own grid fits too, because its thread count is at least the host's bound; and the kernel's
    source passes its THREADS to `_validate` and to its launch binding (a match of the call
    sites, not a check of what the device does; module docstring)."""
    limit = kv.MAX_DEQUANT_REQUESTS
    threads = _threads(kernel)
    assert _checking(limit, threads) * threads - 1 <= INT32_MAX
    assert (limit - 1) + 31 <= INT32_MAX and (limit - 1) + 32 <= INT32_MAX
    assert threads >= kv.DEQUANT_MIN_CHECK_REQUESTS
    for requests in (1, limit):
        for heads in (1, 8, 33):
            last = (INT32_MAX - _checking(requests, kv.DEQUANT_MIN_CHECK_REQUESTS)) // heads
            assert kv._dequant_size_refusal(requests, last, heads) is None
            assert last * heads + _checking(requests, threads) <= INT32_MAX
    source = kernel.read_text()
    assert "self._validate(mPageIndptr, mPageIndices, mSeqLens, cta, tidx, THREADS)" in source
    assert "functools.partial(launch_dequant, threads=THREADS)" in source


def test_dequantize_applies_the_refusal(kv):
    source = (PKG / "kv_cache.py").read_text()
    body = source[source.index("    def dequantize("):source.index("def _dequant_size_refusal(")]
    assert "_dequant_size_refusal(batch, page_indices.shape[0], H)" in body
