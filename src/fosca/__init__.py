# Copyright 2026 Donglin Zhuang
# SPDX-License-Identifier: Apache-2.0
"""Quantized-KV paged GQA decode for Hopper (SM90) and Blackwell (SM100).

The public API is architecture-agnostic: `PagedDecodeWrapper` dispatches to the
right kernel and gates the features that kernel supports. Pool construction lives in
`fosca.pool` (SM100) and `fosca.pool_sm90` (SM90).
"""
from . import pool, pool_sm90
from ._utils import get_capabilities, is_supported
from .wrapper import PagedDecodeWrapper

__version__ = "0.1.0"
__all__ = [
    "PagedDecodeWrapper",
    "get_capabilities",
    "is_supported",
    "pool",
    "pool_sm90",
    "__version__",
]
