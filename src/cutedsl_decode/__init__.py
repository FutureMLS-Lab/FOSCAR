# Copyright (c) 2026 Donglin Zhuang. SPDX-License-Identifier: BSD-3-Clause
"""cutedsl-decode: standalone CuTe DSL GQA paged-decode kernels for SM90/SM100.

Public API is arch-agnostic; internal SM90/SM100 dispatch + feature gating are hidden.
"""
from ._utils import get_capabilities, is_supported
from .wrapper import PagedDecodeWrapper

__version__ = "0.1.0"
__all__ = ["PagedDecodeWrapper", "get_capabilities", "is_supported", "__version__"]
