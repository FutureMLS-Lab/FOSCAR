<p align="center">
  <img src="docs/foscar-logo.png" alt="FOSCAR logo" width="150">
</p>

<h1 align="center">FOSCAR: Accelerating Decode Attention over 2- and 4-Bit KV Caches through Affine Folding</h1>

<p align="center">NVIDIA Hopper and Blackwell · CuTe DSL</p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.09722"><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B.svg?logo=arxiv&amp;logoColor=white" alt="Paper on arXiv"></a>
  <a href="#quickstart"><img src="https://img.shields.io/badge/Quickstart-2563A6.svg?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyLjI1IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik04IDVsMTEgNy0xMSA3eiIgZmlsbD0id2hpdGUiIHN0cm9rZT0ibm9uZSIvPjwvc3ZnPg%3D%3D&amp;logoColor=white" alt="Quickstart"></a>
  <a href="#benchmarks"><img src="https://img.shields.io/badge/Benchmarks-16756D.svg?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyLjI1IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik00IDE5aDE2TTcgMTV2LTRtNSA0VjZtNSA5VjMiLz48L3N2Zz4%3D&amp;logoColor=white" alt="Benchmarks"></a>
  <a href="docs/integration.md"><img src="https://img.shields.io/badge/Integration-6554A3.svg?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyLjI1IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0xMiA3bDItMmE0IDQgMCAwIDEgNiA2bC0zIDNtLTUgMy0yIDJhNCA0IDAgMCAxLTYtNmwzLTNtMiA1IDYtNiIvPjwvc3ZnPg%3D%3D&amp;logoColor=white" alt="Integration guide"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Cite-BibTeX-555555.svg?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyLjI1IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik00IDdoNnY1YzAgMy0yIDUtNSA1di0zYzEuNSAwIDItMSAyLTJINHptMTAgMGg2djVjMCAzLTIgNS01IDV2LTNjMS41IDAgMi0xIDItMmgtM3oiIGZpbGw9IndoaXRlIiBzdHJva2U9Im5vbmUiLz48L3N2Zz4%3D&amp;logoColor=white" alt="Citation in BibTeX"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-32723F.svg?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyLjI1IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0xMiAzbDcgM3Y1YzAgNC0zIDctNyAxMC00LTMtNy02LTctMTBWNnpNOSAxMmwyIDIgNC00Ii8%2BPC9zdmc%2B&amp;logoColor=white" alt="MIT license"></a>
</p>

FOSCAR provides paged grouped-query attention (GQA) decode kernels for packed INT2 and INT4 KV caches.
Affine folding moves scale and offset arithmetic from individual KV values into the attention computation.
Both GPU backends use the same Python API.

## How FOSCAR works

![Four INT2 cells flow through B200. On the left, four scale and offset operations precede the dot product. On the right, one group correction turns the TMEM sum of 3.5 into the same logit of −1.5.](docs/foscar-overview.png)

[Download the B200 walkthrough (42 s, MP4)](docs/foscar-explainer.mp4?raw=true) · [Read the folding derivation](https://arxiv.org/html/2611.09722v1#S4.SS1)

<details>
<summary>Animate the INT2 cells</summary>

![Animated INT2 cells change into K operands, pair with Q, and converge to a scalar in TMEM](docs/foscar-folding-preview.gif)

</details>

The latency bars isolate folding on B200 with FP8 KV operands and equal split-KV counts. The hardware walkthrough shows the production FP4 path. [Paper §6.3, Fig. 10b–c](https://arxiv.org/html/2611.09722v1#S6.SS3)

## Install

Requires Linux, Python 3.10+, and a CUDA-enabled PyTorch build for your GPU (`torch>=2.4`).
Choose the extra that matches your environment.
The `cu13` extra supports Hopper (SM90) and Blackwell (SM100).
The `cu12` extra supports Hopper only.

```bash
git clone https://github.com/FutureMLS-Lab/FOSCAR.git
cd FOSCAR
pip install -e '.[cu13]'
```

For Hopper with CUDA 12, use `pip install -e '.[cu12]'` instead.
Kernels compile on first use and are cached for subsequent runs.

## Quickstart

Run one decode step for two requests with two fully populated cache pages per request.
Set `bits=4` in the layout to try INT4.

```python
import torch
from foscar import PagedDecodeWrapper, QuantKVLayout, get_capabilities

device = "cuda"
arch = get_capabilities()["arch"]
q_dtype = torch.float8_e4m3fn if arch == "sm100" else torch.bfloat16
batch, q_heads, kv_heads, head_dim = 2, 32, 8, 128
page_size, pages_per_request = 64, 2
num_pages = batch * pages_per_request
seq_len = page_size * pages_per_request
num_tokens = batch * seq_len

layout = QuantKVLayout.create(bits=2, head_dim=head_dim, num_kv_heads=kv_heads, device=device)
cache = layout.allocate(num_pages)
k = torch.randn(num_tokens, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
v = torch.randn_like(k)
slots = torch.arange(num_tokens, device=device, dtype=torch.int64)
cache.append(k, v, slots)

indptr = torch.arange(batch + 1, device=device, dtype=torch.int32) * pages_per_request
indices = torch.arange(num_pages, device=device, dtype=torch.int32)
seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
q = torch.randn(batch, q_heads, head_dim, device=device, dtype=torch.bfloat16).to(q_dtype)
out = torch.empty(batch, q_heads, head_dim, device=device, dtype=torch.bfloat16)
workspace = torch.empty(256 * 1024 * 1024, device=device, dtype=torch.uint8)
decode = PagedDecodeWrapper(workspace)
decode.plan(indptr, indices, seq_lens, num_qo_heads=q_heads, layout=layout)
decode.run(q, cache, out=out)
print(out.shape)  # torch.Size([2, 32, 128])
```

For page tables, cache writes, and CUDA graph capture, see the [integration guide](docs/integration.md).
See [design and internals](docs/design.md) for the backend architecture.

## Support matrix

| | Hopper (SM90) | Blackwell (SM100) |
|---|---|---|
| CUDA extra | `cu12` or `cu13` | `cu13` |
| KV storage | INT2, INT4 | INT2, INT4 |
| Query dtype | BF16 | FP8 E4M3 |
| Head dimensions | 64, 128, 256 | 64, 128, 256 |
| Page size | 64 tokens | 64 tokens |

The decode kernels support token-wise and channel-wise grouping.
The public cache writers and `dequantize()` support token-wise layouts.
To use channel-wise decode, supply a cache in the layout expected by the kernel.
Use `get_capabilities()` to check support on the current device.

<details>
<summary>Quantization groups and decode limits</summary>

| D | Tensor | layout | group size | supported? |
|---:|---|---|---:|:---:|
| 64 | K | token | 64 | Y |
| 64 | K | channel | 64, 128, 256 | Y |
| 64 | V | token | 64 | Y |
| 64 | V | channel | 64, 128, 256 | Y |
| 128 | K | token | 64, 128 | Y |
| 128 | K | channel | 64, 128, 256 | Y |
| 128 | V | token | 64 | N |
| 128 | V | token | 128 | Y |
| 128 | V | channel | 64, 128, 256 | Y |
| 256 | K | token | 64, 128, 256 | Y |
| 256 | K | channel | 64, 128, 256 | Y |
| 256 | V | token | 64, 128 | N |
| 256 | V | token | 256 | Y |
| 256 | V | channel | 64, 128, 256 | Y |

Both backends support split-K reduction and 1 to 128 KV heads at 128 query heads.

</details>

## Benchmarks

Geometric mean kernel speedup over FlashInfer BF16 for the Qwen3-8B attention shape on one GPU.
The repository harness measures batches of 1 and 8 at 32K, 128K, 256K, and 512K context lengths.
It measures GPU kernel time with a cold L2 cache and excludes host dispatch.

| GPU | FOSCAR INT2 | FOSCAR INT4 |
|---|---:|---:|
| H100 | **3.41×** | **3.00×** |
| B200 | **2.73×** | **2.33×** |

[H100 plots](docs/decode-latency-h100.png) · [B200 plots](docs/decode-latency-b200.png) · [Full results (CSV)](benchmarks/decode_latency.csv) · [Reproduction guide](benchmarks/README.md)

## Testing

After installing the CUDA extra, install the test dependencies with `pip install -e '.[test]'` and run `pytest -q`.
Device tests skip without a supported GPU.

## Citation

```bibtex
@article{zhuang2026foscar,
  title   = {{FOSCAR}: Accelerating Decode Attention over 2- and 4-Bit {KV} Caches through Affine Folding},
  author  = {Donglin Zhuang and Pragaash Ponnusamy and Xiaoxia Wu and Shuaiwen Leon Song and Zhongzhu Zhou},
  journal = {arXiv preprint arXiv:2611.09722},
  year    = {2026},
  url     = {https://arxiv.org/abs/2611.09722}
}
```

## License

[MIT](LICENSE).
See [NOTICE](NOTICE) for attribution and licenses for code derived from FlashInfer, QuACK, and FlashAttention.
