# tests/test_kernel_import.py
import torch
import pytest


def test_sm100_shared_reduction_library_imports():
    # Imports must succeed regardless of running arch (no device code runs at import).
    #
    # `gqa_decode` is the shared split-K reduction library, not only the non-paged
    # dense kernel: int_fold_decode takes 11 names from it at module level, and
    # `GroupedQueryAttentionDecode` is the namespace for the reduction members. The
    # class name matters too -- `launch_reduction` reaches the kernel via a
    # class-qualified `GroupedQueryAttentionDecode.reduction_kernel(...)` call.
    from fosca._sm100.gqa_decode import GroupedQueryAttentionDecode
    assert GroupedQueryAttentionDecode is not None
    for member in ("can_implement", "gqa_pack", "gemm_view", "gemm_view_bsh",
                   "reduction_none", "reduction_epilogue", "reduction_cluster",
                   "launch_reduction", "reduction_kernel"):
        assert hasattr(GroupedQueryAttentionDecode, member), member


def test_quantized_kernels_import():
    from fosca._sm100.int_fold_decode import IntDecodeSm100
    from fosca._sm90.int_decode_pingpong import IntPingPongDecodeSm90
    assert IntDecodeSm100 is not None
    assert IntPingPongDecodeSm90 is not None
