"""
Batch hallucination analysis for 64x64 DDIM samples.
Scans all num_channel_*/samling/*-ddim/ folders and counts hallucinations.
Outputs summary table, CSV, and comparison plot.

Hallucination cases:
  - empty image (0 shapes total)   -> HALLUCINATION
  - 2+ shapes in any column        -> HALLUCINATION

Detection uses shared hallucination_detector (OpenCV Otsu + connectedComponentsWithStats).
"""

import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

# ── Import shared hallucination detector ─────────────────────────────────────
_DETECTOR_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "flow_matching")
)
sys.path.insert(0, _DETECTOR_PATH)
from hallucination_detector import (  # noqa: E402
    analyze_batch, summarize, COLUMN_NAMES,
)

BASE_DIR = "/Users/admin/workspace/improved-diffusion/simple-shapes-5k-output"


def analyze_hallucinations(npz_path):
    """Return (n_hall, N, hall_indices, norm_indices, arr, col_blobs)."""
    arr     = np.load(npz_path)["arr_0"]          # (N, H, W, 3) uint8
    results = analyze_batch(arr)
    s       = summarize(results)

    col_blobs = {
        name: np.array([r["col_blobs"][name] for r in results], dtype=int)
        for name in COLUMN_NAMES
    }
    hall_indices = s["hall_indices"]
    norm_indices = [i for i in range(len(arr)) if not results[i]["is_hallucination"]]
    return s["n_hall"], len(arr), hall_indices, norm_indices, arr, col_blobs


def save_image_grid(images, indices, out_path, title, top_n=100):
    """Save a grid of images with column boundary lines."""
    n = min(top_n, len(images))
    if n == 0:
        return
    cols = 10
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
    axes = np.array(axes).flatten()
    for i in range(n):
        axes[i].imshow(images[i])
        axes[i].axvline(x=21, color="red", linewidth=0.8, alpha=0.8)
        axes[i].axvline(x=42, color="red", linewidth=0.8, alpha=0.8)
        axes[i].set_title(f"#{indices[i]}", fontsize=6)
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def save_blob_histogram(col_blobs, out_path, N):
    """Save per-column blob count histogram."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, name in zip(axes, COLUMN_NAMES):
        counts  = col_blobs[name]
        max_val = max(counts.max(), 2)
        bins    = np.arange(0, max_val + 2) - 0.5
        ax.hist(counts, bins=bins, color="steelblue", edgecolor="white", rwidth=0.8)
        ax.set_xlabel("shapes detected")
        ax.set_ylabel("# images")
        ax.set_title(f"{name} column")
        ax.set_xticks(range(int(max_val) + 1))
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Blob count distribution (N={N})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# --- Scan ---
results = {}   # (channel, step) -> (n_hall, N)

pattern = os.path.join(BASE_DIR,
    "num_channel_*", "sampling", "ddim", "*-ddim", "samples_*.npz")
files = sorted(glob.glob(pattern))

print(f"Found {len(files)} DDIM npz files\n")

for npz_path in files:
    parts   = npz_path.split(os.sep)
    ch_str  = parts[-5]                        # num_channel_8 etc.
    folder  = parts[-2]                        # checkpoint-020000s-...-ddim
    channel = int(re.search(r"num_channel_(\d+)", ch_str).group(1))
    step    = int(re.search(r"checkpoint-(\d+)s", folder).group(1))

    print(f"  ch={channel:>3}  step={step:>7}  {os.path.basename(npz_path)}", end=" ... ", flush=True)
    n_hall, N, hall_indices, norm_indices, arr, col_blobs = analyze_hallucinations(npz_path)
    results[(channel, step)] = (n_hall, N)
    print(f"{n_hall}/{N}  ({100*n_hall/N:.1f}%)")

    ckpt_dir = os.path.dirname(npz_path)
    tag = f"ch={channel}  step={step}  {n_hall}/{N} ({100*n_hall/N:.1f}%)"

    # Hallucination grid
    np.savetxt(os.path.join(ckpt_dir, "hallucination_indices.txt"),
               hall_indices, fmt="%d")
    save_image_grid(
        arr[hall_indices] if hall_indices else np.zeros((0,)+arr.shape[1:], dtype=np.uint8),
        hall_indices,
        out_path=os.path.join(ckpt_dir, "hallucinations.png"),
        title=f"Hallucinations — {tag}",
    )

    # Normal grid (first 100)
    save_image_grid(
        arr[norm_indices[:100]],
        norm_indices[:100],
        out_path=os.path.join(ckpt_dir, "normal.png"),
        title=f"Normal samples — {tag}",
    )

    # Blob histogram
    save_blob_histogram(
        col_blobs,
        out_path=os.path.join(ckpt_dir, "blob_histogram.png"),
        N=N,
    )

# --- Print table ---
channels = sorted(set(ch for ch, _ in results))
all_steps = sorted(set(st for _, st in results))

print("\n" + "="*70)
print(f"{'step':>8}", end="")
for ch in channels:
    print(f"  ch={ch:<3} rate%", end="")
print()
print("-"*70)

for step in all_steps:
    print(f"{step:>8}", end="")
    for ch in channels:
        if (ch, step) in results:
            n, N = results[(ch, step)]
            print(f"  {n:>5}/{N}  {100*n/N:>5.1f}%", end="")
        else:
            print(f"  {'---':>11}", end="")
    print()

# --- CSV ---
csv_path = os.path.join(BASE_DIR, "hallucination_ddim_summary.csv")
with open(csv_path, "w") as f:
    f.write("channel,step,n_hallucinations,total,rate_pct\n")
    for (ch, step) in sorted(results):
        n, N = results[(ch, step)]
        f.write(f"{ch},{step},{n},{N},{100*n/N:.2f}\n")
print(f"\nCSV saved: {csv_path}")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {8: "tab:red", 16: "tab:orange", 64: "tab:blue"}

for ch in channels:
    steps = sorted(st for (c, st) in results if c == ch)
    rates = [100 * results[(ch, st)][0] / results[(ch, st)][1] for st in steps]
    counts= [results[(ch, st)][0] for st in steps]
    col   = colors.get(ch, "gray")
    lbl   = f"ch={ch}"
    axes[0].plot(steps, rates,  marker="o", color=col, label=lbl, linewidth=2)
    axes[1].plot(steps, counts, marker="o", color=col, label=lbl, linewidth=2)

for ax, ylabel, title in zip(
    axes,
    ["Hallucination rate (%)", "# hallucinations"],
    ["Hallucination rate vs checkpoint (DDIM)",
     "Hallucination count vs checkpoint (DDIM)"]
):
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(all_steps)
    ax.set_xticklabels([f"{s//1000}k" for s in all_steps], rotation=45)

plt.tight_layout()
plot_path = os.path.join(BASE_DIR, "hallucination_ddim_summary.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {plot_path}")
