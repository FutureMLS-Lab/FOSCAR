# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: MIT
"""foscar: standalone CuTe DSL GQA paged-decode kernels for SM90/SM100.

Public API is arch-agnostic; internal SM90/SM100 dispatch + feature gating are hidden.
`QuantKVLayout` resolves a layer's cache layout against the device's reader,
`QuantKVCache` holds that layer's pools and writes them, and `PagedDecodeWrapper`
plans and runs the decode over a cache of its planned layout.
"""
from ._utils import get_capabilities, is_supported
from .kv_cache import QuantKVCache, QuantKVLayout
from .wrapper import PagedDecodeWrapper, graph_workspace_bytes

__version__ = "0.2.0"
__all__ = ["PagedDecodeWrapper", "QuantKVCache", "QuantKVLayout", "get_capabilities",
           "graph_workspace_bytes", "is_supported", "__version__"]
