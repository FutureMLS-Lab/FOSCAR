# Copyright 2026 Donglin Zhuang
# SPDX-License-Identifier: Apache-2.0
"""Fused split-K (flash-decode) LSE combine kernel in CuTe DSL, following the trtllm multi-block reduction strategy.
Merges the OUTER split-K partials produced by IntPingPongDecodeSm90 (_sm90/int_decode_pingpong.py) into the
final output.
  op  : [OUTER, B, HQ, D] f32   (per-partition partial O, normalized by that partition's own softmax sum)
  lse : [OUTER, B, HQ]    f32   (per-partition log-sum-exp = localMax + log(localSum))
  out : [B, HQ, D]        bf16  = sum_o softmax_o(lse) * op_o   (online over OUTER, rescaling by the running max)
One block per (b, hq); D threads, each owns one head-dim element and reduces over the OUTER partitions online.
Entry point: `SplitKCombine(head_dim)`; `main()` compiles one configuration and checks it against torch.
Caller contract: `outer` equals the leading dimension of `op` and `lse`; `head_dim` is the trailing dimension of `op`.
"""
import os, sys
import torch, cutlass, cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack
import tvm_ffi

_LOG2E = 1.4426950408889634
dev = "cuda"


class SplitKCombine:
    """Flash-decode LSE combine of OUTER split-K partials. Grid [B, HQ]; block [D]."""

    def __init__(self, head_dim):
        self.head_dim = head_dim

    @cute.kernel
    def kernel(self, mOp, mLSE, mOut, outer: Int32):
        b, hq, _ = cute.arch.block_idx()
        dd, _, _ = cute.arch.thread_idx()
        log2e = Float32(_LOG2E)
        m = Float32(-1e30); wsum = Float32(0.0); acc = Float32(0.0)   # running max / weight-sum / weighted-acc
        o = Int32(0)
        while o < outer:
            l = mLSE[(o, b, hq)]                 # partition LSE (broadcast across the D threads)
            opd = mOp[(o, b, hq, dd)]            # this partition's partial-O for head-dim dd
            m_new = cutlass.max(m, l)
            corr = cute.arch.exp2((m - m_new) * log2e)   # rescale prior acc for the new running max
            w = cute.arch.exp2((l - m_new) * log2e)      # this partition's softmax weight
            acc = acc * corr + w * opd
            wsum = wsum * corr + w
            m = m_new
            o += 1
        mOut[(b, hq, dd)] = (acc / cutlass.max(wsum, Float32(1e-30))).to(cutlass.BFloat16)

    @cute.jit
    def __call__(self, mOp, mLSE, mOut, outer: Int32, b: Int32, hq: Int32, stream):
        self.kernel(mOp, mLSE, mOut, outer).launch(grid=[b, hq, 1], block=[self.head_dim, 1, 1], stream=stream)


def main():
    torch.manual_seed(0)
    OUTER, B, HQ, D = 8, 16, 32, 128
    op = torch.randn(OUTER, B, HQ, D, device=dev, dtype=torch.float32)
    lse = torch.randn(OUTER, B, HQ, device=dev, dtype=torch.float32) * 3
    out = torch.zeros(B, HQ, D, device=dev, dtype=torch.bfloat16)
    sf = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    comb = SplitKCombine(D)
    from cutedsl_decode._sm90.pingpong_runtime import sm90_compiler
    compile_fn, _ = sm90_compiler(os.environ.get('OPT_LEVEL', '3'))
    cd = compile_fn(comb, from_dlpack(op, assumed_align=16), from_dlpack(lse, assumed_align=16),
                    from_dlpack(out, assumed_align=16), Int32(OUTER), Int32(B), Int32(HQ), sf)
    with tvm_ffi.use_torch_stream():
        cd(op, lse, out, Int32(OUTER), Int32(B), Int32(HQ)); torch.cuda.synchronize()
    # torch ref (same as int_decode_pingpong.split_k_combine)
    M = lse.max(0, keepdim=True).values
    w = torch.exp(lse - M)
    ref = (w.unsqueeze(-1) * op).sum(0) / w.sum(0).unsqueeze(-1).clamp_min(1e-30)
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    print(f"SplitKCombine OUTER={OUTER} B={B} HQ={HQ} D={D}: rel_err={rel:.3e}  {'OK' if rel < 1e-2 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
