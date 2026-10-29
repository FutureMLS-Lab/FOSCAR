"""Render the decode-latency figure from `decode_latency.csv`.

Two panels, one per KV length, with batch on a log axis. The y axis is each
kernel's decode latency divided by FlashInfer bf16's at the same shape, so the
dashed line at 1.00 is bf16 and every other series reads as a fraction of it.
The absolute latencies are in the README table.

Run from the repository root:

    python benchmarks/plot_decode.py
"""
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "docs" / "decode-latency.png"
BASE = "flashinfer_bf16"

# Fixed categorical order; identity is carried by position in the legend and by
# the marker, not by colour alone. bf16 is the reference line rather than a
# series, so it takes no hue.
SERIES = [
    ("ours_int2", "this package, int2", "#2a78d6", "o"),
    ("ours_int4", "this package, int4", "#eb6834", "s"),
    ("trtllm_nvfp4", "TRT-LLM NVFP4", "#1baf7a", "D"),
    ("trtllm_fp8", "TRT-LLM FP8", "#eda100", "^"),
]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
REF = "#8a8984"

rows = list(csv.DictReader((HERE / "decode_latency.csv").open()))
batches = sorted({int(r["batch"]) for r in rows})
kv_lens = sorted({int(r["kv"]) for r in rows})


def us(kernel, batch, kv):
    hit = [r for r in rows if r["kernel"] == kernel and int(r["batch"]) == batch
           and int(r["kv"]) == kv]
    return float(hit[0]["us"]) if hit else None


fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
fig.patch.set_facecolor("#ffffff")

for ax, kv in zip(axes, kv_lens):
    ax.set_facecolor("#ffffff")
    for key, label, color, marker in SERIES:
        xs, ys = [], []
        for b in batches:
            base = us(BASE, b, kv)
            val = us(key, b, kv)
            if base and val:
                xs.append(b)
                ys.append(val / base)
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=5,
                markeredgecolor="#ffffff", markeredgewidth=1.2, label=label,
                zorder=3)

    ax.axhline(1.0, color=REF, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("FlashInfer bf16", xy=(batches[0], 1.0), xytext=(0, 6),
                textcoords="offset points", color=REF, fontsize=8, zorder=4)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.3", "0.4", "0.5", "0.6", "0.8", "1.0"])
    ax.minorticks_off()
    ax.set_xlabel("batch size", color=INK_2, fontsize=9)
    ax.set_title(f"KV length {kv // 1024}K", color=INK, fontsize=10, pad=8)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.8)

axes[0].set_ylabel("decode latency / FlashInfer bf16", color=INK_2, fontsize=9)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
           fontsize=8, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02),
           handlelength=1.8, columnspacing=1.8)

fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT, dpi=200, facecolor="#ffffff")
print(f"wrote {OUT}")
