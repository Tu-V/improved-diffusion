"""
Detect hallucinations in generated simple-shapes images.

Training images are perfectly binary (0 or 255). Each image has 3 columns:
  col 0: [0:W//3]        -> triangle
  col 1: [W//3:2*W//3]   -> square
  col 2: [2*W//3:3*W//3] -> pentagon

Hallucination cases:
  - empty image (0 shapes total)   -> HALLUCINATION
  - 2+ shapes in any column        -> HALLUCINATION

Detection uses the shared hallucination_detector (OpenCV Otsu + connectedComponentsWithStats).

Usage:
  # Single npz:
  python scripts/detect_hallucinations.py --npz path/to/samples_2000x64x64x3.npz

  # Batch across checkpoints (scans base_dir for sampling folders):
  python scripts/detect_hallucinations.py --batch --base_dir /path/to/improved-diffusion
"""

import argparse
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

# ── Import shared hallucination detector ─────────────────────────────────────
_DETECTOR_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "flow_matching")
)
sys.path.insert(0, _DETECTOR_PATH)
from hallucination_detector import (  # noqa: E402
    analyze_batch, summarize, COLUMN_NAMES,
)

CHECKPOINTS = [20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000,
               100000, 110000, 120000, 125000, 130000, 140000, 150000, 160000]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def load_samples(npz_path):
    data = np.load(npz_path)
    arr  = data["arr_0"]   # (N, H, W, 3) uint8
    return arr


def run_analysis(rgb_arr):
    """Run shared detector on a batch; return (scores, col_blobs)."""
    results  = analyze_batch(rgb_arr)
    N        = len(results)
    scores   = np.array([r["score"] for r in results], dtype=float)
    col_blobs = {name: np.array([r["col_blobs"][name] for r in results], dtype=int)
                 for name in COLUMN_NAMES}
    return scores, col_blobs, results


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def visualize_hallucinations(rgb_arr, scores, indices, out_path, title, n=50):
    n = min(n, len(indices))
    if n == 0:
        print(f"No images to visualize for: {title}")
        return
    cols = 10
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
    axes = np.array(axes).flatten()
    W = rgb_arr.shape[2]
    col1, col2 = W // 3, 2 * W // 3
    for i, idx in enumerate(indices[:n]):
        axes[i].imshow(rgb_arr[idx])
        axes[i].axvline(x=col1, color="red", linewidth=0.8, alpha=0.8)
        axes[i].axvline(x=col2, color="red", linewidth=0.8, alpha=0.8)
        axes[i].set_title(f"#{idx} s={int(scores[idx])}", fontsize=6)
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def save_checkpoint_analysis(rgb_arr, scores, col_blobs, results, out_dir, top_n):
    """Save per-checkpoint analysis files inside the checkpoint's own folder."""
    os.makedirs(out_dir, exist_ok=True)
    s      = summarize(results)
    N      = s["n_total"]
    n_hall = s["n_hall"]

    sorted_indices = np.argsort(scores)[::-1]
    hall_sorted    = [i for i in sorted_indices if results[i]["is_hallucination"]]
    normal_sorted  = sorted_indices[::-1]

    print(f"  Hallucinations : {n_hall} / {N} ({100*n_hall/N:.1f}%)")
    print(f"    ├─ empty     : {s['n_empty']}")
    print(f"    └─ double_col: {s['n_double_col']}")

    # Grid: hallucinations
    visualize_hallucinations(
        rgb_arr, scores, hall_sorted,
        out_path=os.path.join(out_dir, "hallucinations.png"),
        title=f"Hallucinations ({n_hall}/{N})",
        n=top_n,
    )

    # Grid: hallucinations by type
    if s["empty_indices"]:
        visualize_hallucinations(
            rgb_arr, scores, s["empty_indices"],
            out_path=os.path.join(out_dir, "hallucinations_empty.png"),
            title=f"Empty images ({s['n_empty']}/{N})",
            n=top_n,
        )

    if s["double_col_indices"]:
        visualize_hallucinations(
            rgb_arr, scores, s["double_col_indices"],
            out_path=os.path.join(out_dir, "hallucinations_double_col.png"),
            title=f"Double-col ({s['n_double_col']}/{N})",
            n=top_n,
        )

    # Grid: normal
    visualize_hallucinations(
        rgb_arr, scores, normal_sorted,
        out_path=os.path.join(out_dir, "normal.png"),
        title="In-support samples",
        n=top_n,
    )

    # Save indices
    np.savetxt(os.path.join(out_dir, "hallucination_indices.txt"),
               s["hall_indices"], fmt="%d")
    np.savetxt(os.path.join(out_dir, "empty_indices.txt"),
               s["empty_indices"], fmt="%d")
    np.savetxt(os.path.join(out_dir, "double_col_indices.txt"),
               s["double_col_indices"], fmt="%d")

    return n_hall, N


# ---------------------------------------------------------------------------
# Batch mode: scan all checkpoint folders + plot summary
# ---------------------------------------------------------------------------

def run_batch(base_dir, top_n):
    results = {}   # step -> (n_hall, N)

    for step in CHECKPOINTS:
        pattern = os.path.join(base_dir, f"*{step:06d}s-sampling*", "samples_10000x*.npz")
        matches = glob.glob(pattern)
        if not matches:
            # fallback: any size
            pattern = os.path.join(base_dir, f"*{step:06d}s-sampling*", "samples_*.npz")
            matches = glob.glob(pattern)
        if not matches:
            print(f"[step {step:06d}] No npz found, skipping.")
            continue

        npz_path = matches[0]
        folder   = os.path.dirname(npz_path)
        out_dir  = os.path.join(folder, "hallucination_analysis")

        print(f"\n[step {step}] {npz_path}")
        rgb_arr = load_samples(npz_path)
        scores, col_blobs, batch_results = run_analysis(rgb_arr)
        n_hall, N = save_checkpoint_analysis(
            rgb_arr, scores, col_blobs, batch_results, out_dir, top_n
        )
        results[step] = (n_hall, N)

    if not results:
        print("No data found.")
        return

    # Plot hallucination rate across checkpoints
    steps  = sorted(results.keys())
    rates  = [100 * results[s][0] / results[s][1] for s in steps]
    counts = [results[s][0] for s in steps]
    labels = [f"{s//1000}k" for s in steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(steps, rates, marker="o", color="coral", linewidth=2)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Hallucination rate (%)")
    ax1.set_title("Hallucination rate vs checkpoint")
    ax1.set_xticks(steps)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.grid(axis="y", alpha=0.3)
    for s, r in zip(steps, rates):
        ax1.annotate(f"{r:.1f}%", (s, r), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)

    ax2.bar(labels, counts, color="steelblue", alpha=0.8)
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("# hallucinations")
    ax2.set_title("Hallucination count vs checkpoint")
    ax2.grid(axis="y", alpha=0.3)
    for i, c in enumerate(counts):
        ax2.text(i, c + max(counts) * 0.01, str(c), ha="center", fontsize=8)

    plt.tight_layout()
    out_plot = os.path.join(base_dir, "hallucination_summary.png")
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"\nSummary plot saved: {out_plot}")

    # Save summary CSV
    out_csv = os.path.join(base_dir, "hallucination_summary.csv")
    with open(out_csv, "w") as f:
        f.write("step,n_hallucinations,total,rate_pct\n")
        for s in steps:
            n, N = results[s]
            f.write(f"{s},{n},{N},{100*n/N:.2f}\n")
    print(f"Summary CSV saved: {out_csv}")

    print("\n=== Summary ===")
    print(f"{'Step':>8} {'Hall':>6} {'Total':>6} {'Rate':>8}")
    for s in steps:
        n, N = results[s]
        print(f"{s:>8} {n:>6} {N:>6} {100*n/N:>7.1f}%")


# ---------------------------------------------------------------------------
# Single mode
# ---------------------------------------------------------------------------

def run_single(npz_path, out_dir, top_n):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading {npz_path} ...")
    rgb_arr = load_samples(npz_path)
    N = len(rgb_arr)
    print(f"Total images: {N}")

    print("Analyzing...")
    scores, col_blobs, batch_results = run_analysis(rgb_arr)
    save_checkpoint_analysis(rgb_arr, scores, col_blobs, batch_results, out_dir, top_n)

    sorted_indices = np.argsort(scores)[::-1]
    print(f"\nTop 20:")
    print(f"{'idx':>6} {'score':>6} {'tri':>4} {'sq':>4} {'pe':>4} {'type':>12}")
    for idx in sorted_indices[:20]:
        r = batch_results[idx]
        cb = r["col_blobs"]
        print(f"{idx:>6} {int(scores[idx]):>6} "
              f"{cb['triangle']:>4} {cb['square']:>4} {cb['pentagon']:>4} "
              f"{r['hall_type']:>12}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch",    action="store_true", help="Scan all checkpoint folders")
    parser.add_argument("--base_dir", default="/Users/admin/workspace/improved-diffusion")
    parser.add_argument("--npz",      help="Single npz path (non-batch mode)")
    parser.add_argument("--out_dir",  default="hallucination_analysis")
    parser.add_argument("--top_n",    type=int, default=50)
    args = parser.parse_args()

    if args.batch:
        run_batch(args.base_dir, args.top_n)
    else:
        if not args.npz:
            parser.error("Provide --npz or use --batch")
        run_single(args.npz, args.out_dir, args.top_n)


if __name__ == "__main__":
    main()
