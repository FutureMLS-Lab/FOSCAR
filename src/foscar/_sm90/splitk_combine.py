# Copyright (c) 2026 Donglin Zhuang
# SPDX-License-Identifier: MIT
"""Fused split-K (flash-decode) LSE combine kernel — the trtllm multi-block-reduction strategy as a CuTe DSL kernel.
Merges the OUTER split-K partials produced by IntPingPongDecodeSm90 (_sm90/int_decode_pingpong.py) into the final
output, replacing the slow torch reduction.
  op  : [OUTER, B, HQ, D] f32   (per-partition partial O, normalized by that partition's own softmax sum)
  lse : [OUTER, B, HQ]    f32   (per-partition log-sum-exp = localMax + log(localSum))
  out : [B, HQ, D]        bf16 or f32 (the tensor's own dtype) = sum_o softmax_o(lse) * op_o
                                (online over OUTER; numerically safe)
One block per (b, hq); D threads, each owns one head-dim element and reduces over the OUTER partitions online.
`SplitKCombineLse` is the same merge that also returns the request's log-sum-exp, a separate class so
that `SplitKCombine` compiles to the same code whether or not a caller asks for one.
`SplitKCombineGraph` and `SplitKCombineGraphLse` are the graph split build's merges, which read each
request's split count from memory (`graph_plan`), so that a captured launch serves every step.
"""
import os
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack
import tvm_ffi

_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453
dev = "cuda"


class SplitKCombine:
    """Flash-decode LSE combine of OUTER split-K partials. Grid [B, HQ]; block [D]."""

    def __init__(self, head_dim):
        self.head_dim = head_dim

    @cute.jit
    def _merge(self, mOp, mLSE, b, hq, dd, outer):
        """`(m, wsum, acc)`: the running maximum, weight sum and weighted partial output of
        (b, hq) at channel dd after all `outer` partitions."""
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
        return m, wsum, acc

    @cute.kernel
    def kernel(self, mOp, mLSE, mOut, outer: Int32):
        b, hq, _ = cute.arch.block_idx()
        dd, _, _ = cute.arch.thread_idx()
        m, wsum, acc = self._merge(mOp, mLSE, b, hq, dd, outer)
        mOut[(b, hq, dd)] = (acc / cutlass.max(wsum, Float32(1e-30))).to(mOut.element_type)

    @cute.jit
    def __call__(self, mOp, mLSE, mOut, outer: Int32, b: Int32, hq: Int32, stream):
        self.kernel(mOp, mLSE, mOut, outer).launch(grid=[b, hq, 1], block=[self.head_dim, 1, 1], stream=stream)


class SplitKCombineLse(SplitKCombine):
    """`SplitKCombine` that also stores each (b, hq)'s merged natural-log LSE, m + ln(wsum), into
    `mLseOut` f32 [B, HQ], from the thread that owns channel 0, after its O store. The partition
    LSEs are natural-log, and every real request has a real partition 0, so wsum >= 1, which is
    why the one-instruction lg2.approx.ftz is exact enough (an error of about 2^-22)."""

    @cute.kernel
    def kernel(self, mOp, mLSE, mOut, mLseOut, outer: Int32):
        b, hq, _ = cute.arch.block_idx()
        dd, _, _ = cute.arch.thread_idx()
        m, wsum, acc = self._merge(mOp, mLSE, b, hq, dd, outer)
        mOut[(b, hq, dd)] = (acc / cutlass.max(wsum, Float32(1e-30))).to(mOut.element_type)
        if dd == 0:
            mLseOut[(b, hq)] = m + cute.math.log2(wsum, approx=True, ftz=True) * Float32(_LN2)

    @cute.jit
    def __call__(self, mOp, mLSE, mOut, mLseOut, outer: Int32, b: Int32, hq: Int32, stream):
        self.kernel(mOp, mLSE, mOut, mLseOut, outer).launch(grid=[b, hq, 1], block=[self.head_dim, 1, 1], stream=stream)


class SplitKCombineGraph(SplitKCombine):
    """The graph split build's external combine: `SplitKCombine` with each request's split count
    k_b read from `mK`, int32 [B], the work list's counts, instead of one launch argument. It
    merges the request's slots 0..k_b-1 only, because the slots above hold whatever an earlier
    replay left there. A request of one split was written directly by its decode CTA, so its
    blocks return at once."""

    @cute.kernel
    def kernel(self, mOp, mLSE, mOut, mK):
        b, hq, _ = cute.arch.block_idx()
        dd, _, _ = cute.arch.thread_idx()
        splits = mK[b]
        if splits > 1:
            m, wsum, acc = self._merge(mOp, mLSE, b, hq, dd, splits)
            mOut[(b, hq, dd)] = (acc / cutlass.max(wsum, Float32(1e-30))).to(mOut.element_type)

    @cute.jit
    def __call__(self, mOp, mLSE, mOut, mK, b: Int32, hq: Int32, stream):
        self.kernel(mOp, mLSE, mOut, mK).launch(grid=[b, hq, 1], block=[self.head_dim, 1, 1], stream=stream)


class SplitKCombineGraphLse(SplitKCombine):
    """`SplitKCombineGraph` that also stores each (b, hq)'s natural-log LSE into `mLseOut`: the
    merged one for a request of several splits, formed as `SplitKCombineLse` forms it, and for a
    request of one split its decode's own, `mLSE[0, b, hq]`, copied, since that decode wrote the
    output but not the caller's LSE."""

    @cute.kernel
    def kernel(self, mOp, mLSE, mOut, mK, mLseOut):
        b, hq, _ = cute.arch.block_idx()
        dd, _, _ = cute.arch.thread_idx()
        splits = mK[b]
        if splits > 1:
            m, wsum, acc = self._merge(mOp, mLSE, b, hq, dd, splits)
            mOut[(b, hq, dd)] = (acc / cutlass.max(wsum, Float32(1e-30))).to(mOut.element_type)
            if dd == 0:
                mLseOut[(b, hq)] = m + cute.math.log2(wsum, approx=True, ftz=True) * Float32(_LN2)
        elif dd == 0:
            mLseOut[(b, hq)] = mLSE[(0, b, hq)]

    @cute.jit
    def __call__(self, mOp, mLSE, mOut, mK, mLseOut, b: Int32, hq: Int32, stream):
        self.kernel(mOp, mLSE, mOut, mK, mLseOut).launch(grid=[b, hq, 1], block=[self.head_dim, 1, 1], stream=stream)


def main():
    torch.manual_seed(0)
    OUTER, B, HQ, D = 8, 16, 32, 128
    op = torch.randn(OUTER, B, HQ, D, device=dev, dtype=torch.float32)
    lse = torch.randn(OUTER, B, HQ, device=dev, dtype=torch.float32) * 3
    out = torch.zeros(B, HQ, D, device=dev, dtype=torch.bfloat16)
    sf = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    comb = SplitKCombine(D)
    from foscar._sm90.pingpong_runtime import sm90_compiler
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
