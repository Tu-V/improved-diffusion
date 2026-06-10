"""
Scan all DDIM checkpoint folders and report hallucination stats.

Usage:
    python scripts/scan_hallucinations_ddim.py \
        --ddim_dir simple-shapes-5k-output/num_channel_64/sampling/ddim \
        --out_dir  simple-shapes-5k-output/num_channel_64/hallucination_summary
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

# ── Import shared hallucination detector ──────────────────────────────────────
_DETECTOR_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "flow_matching")
)
sys.path.insert(0, _DETECTOR_PATH)
from hallucination_detector import analyze_batch, summarize, COLUMN_NAMES  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_rgb(npz_path: str) -> np.ndarray:
    """Load arr_0 from npz, ensure uint8 (N, H, W, 3)."""
    arr = np.load(npz_path)["arr_0"]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255)
        arr = arr.astype(np.uint8)
    return arr


def extract_step(folder_name: str) -> int:
    """'checkpoint-020000s-sampling-100steps-ddim' → 20000"""
    m = re.search(r"checkpoint-(\d+)s", folder_name)
    return int(m.group(1)) if m else -1


def save_image_grid(rgb_arr, results, indices, out_path, title, n=100):
    """Save a grid of images with red column dividers and per-image caption."""
    n = min(n, len(indices))
    if n == 0:
        print(f"    (no images for: {title})")
        return
    W = rgb_arr.shape[2]
    col_x = [W // 3, 2 * W // 3]   # red divider positions

    cols = 10
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
    axes = np.array(axes).flatten()

    for i, idx in enumerate(indices[:n]):
        r = results[idx]
        axes[i].imshow(rgb_arr[idx])
        for x in col_x:
            axes[i].axvline(x=x, color="red", linewidth=0.8, alpha=0.8)
        cb    = r["col_blobs"]
        b_str = f"t{cb['triangle']} s{cb['square']} p{cb['pentagon']}"
        h_str = r["hall_type"] if r["is_hallucination"] else "ok"
        axes[i].set_title(f"#{idx} [{h_str}]\n{b_str}", fontsize=5.5)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {os.path.basename(out_path)}")


def save_per_checkpoint(rgb_arr, results, s, scores, out_dir, step, top_n):
    """Save hallucination_analysis/ folder for one checkpoint."""
    os.makedirs(out_dir, exist_ok=True)

    hall_sorted   = sorted(s["hall_indices"],   key=lambda i: -results[i]["score"])
    normal_sorted = sorted(
        [i for i in range(len(results)) if not results[i]["is_hallucination"]],
        key=lambda i: scores[i],
    )

    save_image_grid(
        rgb_arr, results, hall_sorted,
        out_path=os.path.join(out_dir, "hallucinations.png"),
        title=f"All hallucinations  step={step}  ({s['n_hall']}/{s['n_total']}  {100*s['hall_rate']:.1f}%)",
        n=top_n,
    )
    save_image_grid(
        rgb_arr, results, s["empty_indices"],
        out_path=os.path.join(out_dir, "hallucinations_empty.png"),
        title=f"Empty images  step={step}  ({s['n_empty']}/{s['n_total']})",
        n=top_n,
    )
    save_image_grid(
        rgb_arr, results, s["double_col_indices"],
        out_path=os.path.join(out_dir, "hallucinations_double_col.png"),
        title=f"Double-col  step={step}  ({s['n_double_col']}/{s['n_total']})",
        n=top_n,
    )
    save_image_grid(
        rgb_arr, results, normal_sorted,
        out_path=os.path.join(out_dir, "normal.png"),
        title=f"Normal samples  step={step}  ({s['n_normal']}/{s['n_total']})",
        n=top_n,
    )

    np.savetxt(os.path.join(out_dir, "hallucination_indices.txt"),
               s["hall_indices"], fmt="%d")
    np.savetxt(os.path.join(out_dir, "empty_indices.txt"),
               s["empty_indices"], fmt="%d")
    np.savetxt(os.path.join(out_dir, "double_col_indices.txt"),
               s["double_col_indices"], fmt="%d")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddim_dir", required=True,
                        help="Parent folder containing checkpoint-*/ subdirs")
    parser.add_argument("--out_dir", default="hallucination_summary_ddim",
                        help="Where to write summary files")
    parser.add_argument("--top_n", type=int, default=50,
                        help="Max images per per-checkpoint grid (0 = skip grids)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Collect checkpoint folders ──
    entries = sorted(os.listdir(args.ddim_dir))
    ckpt_dirs = [
        e for e in entries
        if os.path.isdir(os.path.join(args.ddim_dir, e))
        and extract_step(e) > 0
    ]
    if not ckpt_dirs:
        print(f"No checkpoint folders found in {args.ddim_dir}")
        return

    print(f"Found {len(ckpt_dirs)} checkpoint folders.\n")

    rows = []   # list of result dicts

    for folder_name in ckpt_dirs:
        step     = extract_step(folder_name)
        folder   = os.path.join(args.ddim_dir, folder_name)

        # Find the npz
        npz_candidates = [
            f for f in os.listdir(folder)
            if f.startswith("samples_") and f.endswith(".npz")
        ]
        if not npz_candidates:
            print(f"  [step {step:>7}] no npz found — skip")
            continue
        npz_path = os.path.join(folder, npz_candidates[0])

        print(f"  [step {step:>7}] {npz_candidates[0]} ... ", end="", flush=True)
        arr     = load_rgb(npz_path)
        results = analyze_batch(arr)
        s       = summarize(results)
        scores  = np.array([r["score"] for r in results], dtype=float)

        print(f"hall={s['n_hall']}/{s['n_total']}  "
              f"({100*s['hall_rate']:.1f}%)  "
              f"empty={s['n_empty']}  double_col={s['n_double_col']}")

        # ── Save hallucination_analysis/ folder ──
        analysis_dir = os.path.join(folder, "hallucination_analysis")
        save_per_checkpoint(arr, results, s, scores, analysis_dir, step, args.top_n)

        # ── Save stats txt inside analysis folder ──
        stats_path = os.path.join(analysis_dir, "stats.txt")
        with open(stats_path, "w") as f:
            f.write(f"step            : {step}\n")
            f.write(f"total           : {s['n_total']}\n")
            f.write(f"hallucinations  : {s['n_hall']}  ({100*s['hall_rate']:.2f}%)\n")
            f.write(f"  empty         : {s['n_empty']}\n")
            f.write(f"  double_col    : {s['n_double_col']}\n")
            f.write(f"normal          : {s['n_normal']}\n\n")
            f.write("per-column blob distribution:\n")
            f.write(f"  {'column':10s}  {'0 shapes':>10}  {'1 shape':>10}  {'2+ shapes':>10}\n")
            for name in COLUMN_NAMES:
                cc = s["col_counts"][name]
                f.write(f"  {name:10s}  {cc['0']:>10}  {cc['1']:>10}  {cc['2+']:>10}\n")

        rows.append({
            "step":       step,
            "n_total":    s["n_total"],
            "n_hall":     s["n_hall"],
            "n_empty":    s["n_empty"],
            "n_double":   s["n_double_col"],
            "hall_rate":  s["hall_rate"],
        })

    if not rows:
        print("No data collected.")
        return

    rows.sort(key=lambda r: r["step"])

    # ── Save summary CSV ──
    csv_path = os.path.join(args.out_dir, "hallucination_summary.csv")
    with open(csv_path, "w") as f:
        f.write("step,n_total,n_hallucinations,n_empty,n_double_col,hall_rate_pct\n")
        for r in rows:
            f.write(f"{r['step']},{r['n_total']},{r['n_hall']},"
                    f"{r['n_empty']},{r['n_double']},{100*r['hall_rate']:.2f}\n")
    print(f"\nCSV saved → {csv_path}")

    # ── Print summary table ──
    print(f"\n{'Step':>8}  {'Hall':>6}  {'Total':>6}  {'Rate':>7}  {'Empty':>6}  {'DblCol':>6}")
    print("-" * 52)
    for r in rows:
        print(f"{r['step']:>8}  {r['n_hall']:>6}  {r['n_total']:>6}  "
              f"{100*r['hall_rate']:>6.1f}%  {r['n_empty']:>6}  {r['n_double']:>6}")

    # ── Plot ──
    steps  = [r["step"]     for r in rows]
    rates  = [100*r["hall_rate"] for r in rows]
    counts = [r["n_hall"]   for r in rows]
    empties= [r["n_empty"]  for r in rows]
    doubles= [r["n_double"] for r in rows]
    labels = [f"{s//1000}k" for s in steps]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: rate %
    ax = axes[0]
    ax.plot(steps, rates, marker="o", color="coral", linewidth=2, label="total hall rate")
    ax.fill_between(steps, rates, alpha=0.15, color="coral")
    for s, r in zip(steps, rates):
        ax.annotate(f"{r:.1f}%", (s, r), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Hallucination rate (%)")
    ax.set_title("Hallucination rate vs checkpoint (DDIM 100 steps)")
    ax.set_xticks(steps)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(rates) * 1.25 + 1)

    # Right: stacked count — empty vs double_col
    ax = axes[1]
    ax.bar(labels, empties, color="steelblue", label="empty (0 shapes)", alpha=0.85)
    ax.bar(labels, doubles, bottom=empties, color="tomato",
           label="double_col (2+ shapes)", alpha=0.85)
    for i, r in enumerate(rows):
        total = r["n_hall"]
        if total > 0:
            ax.text(i, total + max(counts) * 0.01, str(total),
                    ha="center", fontsize=7.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("# hallucinations")
    ax.set_title("Hallucination count vs checkpoint (stacked by type)")
    ax.legend(fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "hallucination_summary.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved  → {plot_path}")


if __name__ == "__main__":
    main()
