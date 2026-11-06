"""Every benchmarked implementation as a zero-argument callable.

Each builder receives one problem, converts it to the cache format its kernel
reads, pays every setup cost (JIT, plan, quantisation of the cache) outside the
returned callable, and returns the one call a decode step makes. Timing calls
that callable; nothing else runs inside the measurement.

The competitor builders import FlashInfer, which is a benchmark-only dependency:
`pip install "foscar[bench]"`. Nothing in the package imports it.
"""

from __future__ import annotations

import math

import torch

import foscar

#: The query scale the Blackwell kernel was measured at: the fp8 cast multiplies
#: by this and the softmax scale divides by it, which keeps the cast in range
#: without changing the scores.
Q_SCALE = 8.0

#: FlashInfer's own decode workspace size.
WORKSPACE_BYTES = 256 * 1024 * 1024


def _sm_scale(problem) -> float:
    return 1.0 / math.sqrt(problem["head_dim"])


def _slot_mapping(problem) -> torch.Tensor:
    """Where each request's tokens live, in the order the problem laid them out."""
    pages = problem["context"] // problem["page_size"]
    page = problem["page_size"]
    device = problem["device"]
    physical = problem["indices"].view(problem["batch"], pages)
    within = torch.arange(page, device=device)
    return (physical[:, :, None] * page + within[None, None, :]).reshape(-1)


# --- ours -------------------------------------------------------------------


def build_ours(problem, bits: int):
    """The package's own decode: a token-wise quantized cache and one wrapper."""
    device = problem["device"]
    head_dim = problem["head_dim"]
    batch, page = problem["batch"], problem["page_size"]
    layout = foscar.QuantKVLayout.create(
        bits=bits, head_dim=head_dim, num_kv_heads=problem["kv_heads"],
        k_quant=("token", head_dim), v_quant=("token", head_dim), device=device)
    cache = layout.allocate(batch * problem["context"] // page)
    cache.append(problem["k"], problem["v"], _slot_mapping(problem))

    q = problem["q"]
    if problem["arch"] == "sm100":
        q = (q.float() * Q_SCALE).clamp(-448, 448).to(torch.float8_e4m3fn)
        scale = _sm_scale(problem) / Q_SCALE
    else:
        scale = _sm_scale(problem)

    wrapper = foscar.PagedDecodeWrapper(
        torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device))
    wrapper.plan(problem["indptr"], problem["indices"], problem["seq_lens"],
                 num_qo_heads=problem["q_heads"], layout=layout)
    out = problem["out"]

    def fn() -> None:
        wrapper.run(q, cache, out=out, sm_scale=scale)

    return fn


# --- FlashInfer -------------------------------------------------------------


def _flashinfer():
    import flashinfer

    return flashinfer


def _bf16_pool(problem, hnd: bool) -> torch.Tensor:
    """The bf16 paged pool FlashInfer reads: [pages, 2, page|heads, heads|page, D]."""
    batch, pages, page = (problem["batch"], problem["context"] // problem["page_size"],
                          problem["page_size"])
    heads, head_dim = problem["kv_heads"], problem["head_dim"]
    k = problem["k"].view(batch, pages, page, heads, head_dim)
    v = problem["v"].view(batch, pages, page, heads, head_dim)
    if hnd:
        k, v = k.permute(0, 1, 3, 2, 4), v.permute(0, 1, 3, 2, 4)
        shape = (pages * batch, 2, heads, page, head_dim)
    else:
        shape = (pages * batch, 2, page, heads, head_dim)
    pool = torch.empty(shape, dtype=torch.bfloat16, device=problem["device"])
    physical = problem["indices"].view(batch, pages)
    pool[physical, 0] = k
    pool[physical, 1] = v
    return pool


def build_flashinfer_bf16(problem):
    """FlashInfer's paged bf16 decode, at the faster backend each GPU has.

    Blackwell runs the TensorRT-LLM backend, which is 6 to 14 percent faster than
    the fused-attention one at every shape measured here. Hopper has no
    TensorRT-LLM decode kernel, so it runs the fused-attention backend.
    """
    flashinfer = _flashinfer()
    page = problem["page_size"]
    hnd = problem["arch"] == "sm100"
    pool = _bf16_pool(problem, hnd)
    last_page = torch.full((problem["batch"],), page, dtype=torch.int32,
                           device=problem["device"])
    kwargs = {"backend": "trtllm-gen"} if hnd else {}
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=problem["device"]),
        kv_layout="HND" if hnd else "NHD", use_tensor_cores=True, **kwargs)
    wrapper.plan(problem["indptr"], problem["indices"], last_page,
                 problem["q_heads"], problem["kv_heads"], problem["head_dim"], page,
                 q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                 sm_scale=_sm_scale(problem))
    q, out = problem["q"], problem["out"]

    def fn() -> None:
        wrapper.run(q, pool, out=out)

    return fn


def _fp8_scale(tensor: torch.Tensor) -> float:
    top = float(tensor.abs().amax())
    return torch.finfo(torch.float8_e4m3fn).max / max(top, 1e-12) * 0.1


def _fp8(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    info = torch.finfo(torch.float8_e4m3fn)
    return (tensor.float() * scale).clamp(info.min, info.max).to(torch.float8_e4m3fn)


def build_trtllm(problem, kv_dtype: str):
    """TensorRT-LLM's Blackwell decode through FlashInfer, at fp8 or nvfp4 KV."""
    from flashinfer.decode import (get_trtllm_gen_fmha_module,
                                   trtllm_batch_decode_with_kv_cache)
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

    get_trtllm_gen_fmha_module()
    device = problem["device"]
    batch, pages, page = (problem["batch"], problem["context"] // problem["page_size"],
                          problem["page_size"])
    heads, head_dim = problem["kv_heads"], problem["head_dim"]
    total = batch * pages
    physical = problem["indices"].view(batch, pages)

    q = (problem["q"].float() * Q_SCALE).clamp(-448, 448).to(torch.float8_e4m3fn)
    q_scale = 1.0 / Q_SCALE
    k = problem["k"].view(batch, pages, page, heads, head_dim).permute(0, 1, 3, 2, 4)
    v = problem["v"].view(batch, pages, page, heads, head_dim).permute(0, 1, 3, 2, 4)
    k_scale, v_scale = 1.0 / _fp8_scale(k), 1.0 / _fp8_scale(v)

    if kv_dtype == "fp8":
        pool = torch.empty(total, 2, heads, page, head_dim,
                           dtype=torch.float8_e4m3fn, device=device)
        pool[physical, 0] = _fp8(k, 1.0 / k_scale)
        pool[physical, 1] = _fp8(v, 1.0 / v_scale)
        kv_cache, kv_cache_sf = pool, None
        out_dtype, o_scale = torch.bfloat16, 1.0
    elif kv_dtype == "nvfp4":
        from flashinfer.fp4_quantization import nvfp4_quantize_paged_kv_cache

        k_gsf = torch.tensor([448.0 / max(float(k.abs().amax()), 1e-12)],
                             dtype=torch.float32, device=device)
        v_gsf = torch.tensor([448.0 / max(float(v.abs().amax()), 1e-12)],
                             dtype=torch.float32, device=device)
        k_scale, v_scale = float((1.0 / k_gsf).item()), float((1.0 / v_gsf).item())
        k_fp4 = torch.empty(total, heads, page, head_dim // 2, dtype=torch.uint8,
                            device=device)
        v_fp4 = torch.empty_like(k_fp4)
        k_sf = torch.empty(total, heads, page, head_dim // 16,
                           dtype=torch.float8_e4m3fn, device=device)
        v_sf = torch.empty_like(k_sf)
        for lo in range(0, batch, 8):
            hi = min(lo + 8, batch)
            (kp, vp), (ks, vs), _, _ = nvfp4_quantize_paged_kv_cache(
                k[lo:hi].contiguous(), v[lo:hi].contiguous(), kv_layout="HND",
                k_global_sf=k_gsf, v_global_sf=v_gsf)
            here = physical[lo:hi]
            k_fp4[here] = kp
            v_fp4[here] = vp
            k_sf[here] = ks
            v_sf[here] = vs
        kv_cache, kv_cache_sf = (k_fp4, v_fp4), (k_sf, v_sf)
        out_dtype, o_scale = torch.float8_e4m3fn, v_scale
    else:
        raise ValueError(f"unknown kv dtype {kv_dtype!r}")

    block_tables = physical.contiguous()
    seq_lens = problem["seq_lens"].to(torch.uint32).contiguous()
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.int8, device=device)
    counter = torch.zeros(
        get_trtllm_gen_multi_ctas_kv_counter_bytes(
            batch, problem["q_heads"],
            torch.cuda.get_device_properties(device).multi_processor_count),
        dtype=torch.uint8, device=device)
    out = torch.empty(batch, problem["q_heads"], head_dim, dtype=out_dtype, device=device)
    bmm1 = q_scale * k_scale * _sm_scale(problem)
    bmm2 = v_scale / o_scale
    max_seq_len = problem["context"]

    def fn() -> None:
        trtllm_batch_decode_with_kv_cache(
            q, kv_cache, workspace, block_tables, seq_lens, max_seq_len,
            bmm1, bmm2, -1, out=out, kv_layout="HND", enable_pdl=True,
            backend="trtllm-gen", q_len_per_req=1, o_scale=o_scale,
            kv_cache_sf=kv_cache_sf, uses_shared_paged_kv_idx=True,
            multi_ctas_kv_counter_buffer=counter)

    fn()
    torch.cuda.synchronize()
    return fn


# --- registry ---------------------------------------------------------------

REGISTRY = {
    "ours_int2": lambda problem: build_ours(problem, 2),
    "ours_int4": lambda problem: build_ours(problem, 4),
    "flashinfer_bf16": build_flashinfer_bf16,
    "trtllm_fp8": lambda problem: build_trtllm(problem, "fp8"),
    "trtllm_nvfp4": lambda problem: build_trtllm(problem, "nvfp4"),
}


def supported(name: str, problem) -> bool:
    """Whether this kernel builds for this shape on this device."""
    if name.startswith("ours_"):
        accepted = [int(d) for d in foscar.get_capabilities()["head_dims"].split(",")
                    if d]
        if problem["head_dim"] not in accepted:
            return False
    if name.startswith("trtllm_"):
        return problem["arch"] == "sm100" and problem["page_size"] == 64
    return True


def build(name: str, problem):
    return REGISTRY[name](problem)
