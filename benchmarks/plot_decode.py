"""Render the decode-speedup figures from `decode_latency.csv`.

One figure per GPU, each with two panels, one per batch size. Context lengths are
placed at even spacing rather than by value, so the four points read at the same
interval. The y axis is the speedup over FlashInfer bf16 at the same shape,
which is bf16's latency divided by the kernel's, so the dashed line at 1.00 is
bf16 and higher is better. The latencies themselves are in the README tables.

Run from the repository root:

    python benchmarks/plot_decode.py
"""
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

HERE = pathlib.Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
BASE = "flashinfer_bf16"

# Fixed categorical order; identity is carried by position in the legend and by
# the marker, not by colour alone. bf16 is the reference line rather than a
# series, so it takes no hue.
SERIES = [
    ("ours_int2", "FOSCA int2", "#2a78d6", "o"),
    ("ours_int4", "FOSCA int4", "#eb6834", "s"),
    ("trtllm_nvfp4", "TRT-LLM NVFP4", "#1baf7a", "D"),
    ("trtllm_fp8", "TRT-LLM FP8", "#eda100", "^"),
]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
REF = "#8a8984"

rows = list(csv.DictReader((HERE / "decode_latency.csv").open()))


def us(gpu, kernel, batch, kv):
    hit = [r for r in rows if r["gpu"] == gpu and r["kernel"] == kernel
           and int(r["batch"]) == batch and int(r["kv"]) == kv]
    return float(hit[0]["us"]) if hit else None


def render(gpu, out):
    here = [r for r in rows if r["gpu"] == gpu]
    batches = sorted({int(r["batch"]) for r in here})
    kv_lens = sorted({int(r["kv"]) for r in here})
    present = [s for s in SERIES
               if any(r["kernel"] == s[0] for r in here)]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    fig.patch.set_facecolor("#ffffff")
    top = 1.0

    for ax, batch in zip(axes, batches):
        ax.set_facecolor("#ffffff")
        for key, label, color, marker in present:
            xs, ys = [], []
            for i, kv in enumerate(kv_lens):
                base = us(gpu, BASE, batch, kv)
                val = us(gpu, key, batch, kv)
                if base and val:
                    xs.append(i)
                    ys.append(base / val)
            ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=5,
                    markeredgecolor="#ffffff", markeredgewidth=1.2, label=label,
                    zorder=3)
            if ys:
                top = max(top, max(ys))

        ax.axhline(1.0, color=REF, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.annotate("FlashInfer bf16", xy=(0, 1.0), xytext=(0, 6),
                    textcoords="offset points", color=REF, fontsize=8, zorder=4)

        ax.set_xticks(range(len(kv_lens)))
        ax.set_xticklabels([f"{kv // 1024}K" for kv in kv_lens])
        ax.set_xlim(-0.25, len(kv_lens) - 0.75)
        ax.minorticks_off()
        ax.set_xlabel("context length", color=INK_2, fontsize=9)
        ax.set_title(f"batch {batch}", color=INK, fontsize=10, pad=8)
        ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.8)

    # Fit the axis to the data: the bf16 reference at 1.00 always stays in view, and
    # the top is the highest point plus a little headroom rather than a fixed 5.0.
    axes[0].set_ylim(0.95, top * 1.08)
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axes[0].set_ylabel("speedup over FlashInfer bf16", color=INK_2, fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(present), frameon=False,
               fontsize=8, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.8, columnspacing=1.8)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=200, facecolor="#ffffff")
    print(f"wrote {out}")


render("b200", DOCS / "decode-latency-b200.png")
render("h100", DOCS / "decode-latency-h100.png")
