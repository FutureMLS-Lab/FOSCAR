"""Measure the decode grid in `bench.yaml` and write one CSV row per cell.

    python benchmarks/run_bench.py                       # every shape and kernel
    python benchmarks/run_bench.py --shape inkling
    python benchmarks/run_bench.py --kernel ours_int2 --batch 1 --context 32768
    python benchmarks/run_bench.py --out results.csv

A cell is one kernel at one (shape, batch, context). Every kernel of a cell sees
the same K, V and page table; each converts them to the cache format its own
kernel reads. Latency is `timer.measure`, which is GPU kernel time against a
cold L2, so a column is comparable with another column measured the same way and
with the tables in the README.

The competitor columns need FlashInfer; `pip install "foscar[bench]"`. Nothing
in the package imports it.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import subprocess
import sys

import torch

import kernels
import timer
from timer import measure

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "bench.yaml"


def load_config(path):
    import yaml

    with open(path) as handle:
        return yaml.safe_load(handle)


def arch_of(device):
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) == (9, 0):
        return "sm90"
    if major == 10:
        return "sm100"
    raise SystemExit(f"no decode kernel for compute capability {major}.{minor}")


def make_problem(shape, batch, context, page_size, device, arch):
    """The tensors every column of one cell decodes over."""
    heads, kv_heads, head_dim = (shape["num_qo_heads"], shape["num_kv_heads"],
                                 shape["head_dim"])
    pages = context // page_size
    if pages * page_size != context:
        raise SystemExit(f"context {context} is not a multiple of page {page_size}")

    generator = torch.Generator(device=device).manual_seed(0)
    k = torch.randn(batch * context, kv_heads, head_dim, device=device,
                    dtype=torch.bfloat16, generator=generator)
    v = torch.randn(batch * context, kv_heads, head_dim, device=device,
                    dtype=torch.bfloat16, generator=generator)
    q = torch.randn(batch, heads, head_dim, device=device, dtype=torch.bfloat16,
                    generator=generator)

    # One request's pages are contiguous, and the table is a permutation so that
    # no column is handed an identity gather.
    indptr = torch.arange(batch + 1, device=device, dtype=torch.int32) * pages
    indices = torch.randperm(batch * pages, device=device,
                             generator=generator).to(torch.int32)
    seq_lens = torch.full((batch,), context, device=device, dtype=torch.int32)
    out = torch.empty(batch, heads, head_dim, device=device, dtype=torch.bfloat16)
    return {
        "arch": arch, "batch": batch, "context": context, "page_size": page_size,
        "q_heads": heads, "kv_heads": kv_heads, "head_dim": head_dim,
        "k": k, "v": v, "q": q, "indptr": indptr, "indices": indices,
        "seq_lens": seq_lens, "out": out, "device": device,
    }


def run_isolated(args, names, wanted, batches, contexts):
    """Measure one cell per child process, then merge the rows.

    A CUDA context is shared by everything in a process, and a fault in one
    library's kernel has been seen to take down a later cell's build in the same
    process. A child per cell keeps every measurement in the state the published
    ones were taken in, and a cell that faults costs only itself.
    """
    parts, failed = [], []
    for kernel in wanted:
        for name in names:
            for batch in batches:
                for context in contexts:
                    part = args.out.with_name(
                        f"{args.out.stem}.{kernel}.{name}.{batch}.{context}.csv")
                    command = [sys.executable, str(pathlib.Path(__file__).resolve()),
                               "--config", str(args.config), "--kernel", kernel,
                               "--shape", name, "--batch", str(batch),
                               "--context", str(context), "--out", str(part),
                               "--in-process"]
                    if args.iterations:
                        command += ["--iterations", str(args.iterations)]
                    result = subprocess.run(command, check=False)
                    if result.returncode:
                        print(f"  {name} b={batch} kv={context} {kernel}: FAILED",
                              flush=True)
                        failed.append(f"{name} b={batch} kv={context} {kernel}")
                        part.unlink(missing_ok=True)
                        continue
                    parts.append(part)

    rows = []
    for part in parts:
        with open(part, newline="") as handle:
            rows.extend(csv.DictReader(handle))
        part.unlink()
    if not rows:
        raise SystemExit("no cell was measured")
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    if failed:
        print(f"{len(failed)} cell(s) failed and are absent from the table: "
              + ", ".join(failed))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--shape", action="append", help="shape name (repeatable)")
    parser.add_argument("--kernel", action="append", help="kernel name (repeatable)")
    parser.add_argument("--batch", type=int, action="append")
    parser.add_argument("--context", type=int, action="append")
    parser.add_argument("--out", type=pathlib.Path, default=HERE / "results.csv")
    parser.add_argument("--iterations", type=int, default=None,
                        help=f"measured iterations (default {timer.ITERS})")
    parser.add_argument("--in-process", action="store_true",
                        help="measure every kernel in this process instead of one "
                             "process per kernel")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    device = torch.device("cuda")
    arch = arch_of(device)
    shapes = config["shapes"]
    names = args.shape or list(shapes)
    wanted = args.kernel or list(config["kernels"])
    batches = args.batch or list(config["grid"]["batch"])
    contexts = args.context or list(config["grid"]["context"])
    page_size = int(config["page_size"])

    unknown = [n for n in wanted if n not in kernels.REGISTRY]
    if unknown:
        raise SystemExit(f"unknown kernel(s) {unknown}; known {sorted(kernels.REGISTRY)}")

    if len(wanted) > 1 and not args.in_process:
        return run_isolated(args, names, wanted, batches, contexts)

    rows = []
    print(f"{torch.cuda.get_device_name(device)} | {arch} | page {page_size} | "
          f"kernel time from CUPTI over a cold L2", flush=True)
    for name in names:
        shape = shapes[name]
        for batch in batches:
            for context in contexts:
                problem = make_problem(shape, batch, context, page_size, device, arch)
                for kernel in wanted:
                    if not kernels.supported(kernel, problem):
                        print(f"  {name} b={batch} kv={context} {kernel}: skipped",
                              flush=True)
                        continue
                    fn = kernels.build(kernel, problem)
                    result = measure(fn, iters=args.iterations) if args.iterations \
                        else measure(fn)
                    rows.append({
                        "gpu": torch.cuda.get_device_name(device),
                        "arch": arch,
                        "shape": name,
                        "batch": batch,
                        "context": context,
                        "q_heads": problem["q_heads"],
                        "kv_heads": problem["kv_heads"],
                        "head_dim": problem["head_dim"],
                        "page_size": page_size,
                        "kernel": kernel,
                        "us": round(result.median_us, 2),
                        "n_kernels": result.n_kernels,
                    })
                    print(f"  {name} b={batch} kv={context} {kernel}: "
                          f"{result.median_us:.2f} us "
                          f"({result.n_kernels} kernel(s), "
                          f"{result.iters_observed}/{result.iters})", flush=True)
                del problem
                torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("no cell was measured")
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
