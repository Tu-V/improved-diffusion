"""
Detect hallucinations in generated 16x16 simple-shapes images.

Image layout (matches gen_simple_shapes_16x16.py):
  col 0 [x 0:5]  : triangle
  col 1 [x 5:10] : square
  col 2 [x 10:15]: pentagon
  col 3 [x 15:16]: padding (ignored)

Hallucination cases:
  1. double_col — >= 2 shapes trong cùng 1 cột
  2. empty      — không có shape nào trong toàn bộ ảnh

Detection logic được import từ:
  /Users/admin/workspace/flow_matching/hallucination_detector.py
  (OpenCV Otsu threshold + connectedComponentsWithStats — pixel-accurate area)

Usage:
  python scripts/detect_hallucinations_16x16.py \
      --npz path/to/samples_10000x16x16x3.npz \
      --out_dir hallucination_analysis_16x16
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# ── Import shared detector ────────────────────────────────────────────────────
DETECTOR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "flow_matching",
)
sys.path.insert(0, os.path.abspath(DETECTOR_PATH))

from hallucination_detector import (   # noqa: E402
    analyze_batch, summarize,
    COLUMN_NAMES, COLUMN_SLICES,
)

ZOOM = 10   # upscale factor for visualization (16→160 px per image)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(npz_path: str):
    """
    Load npz và trả về rgb_arr (N, H, W, 3) uint8.
    Hỗ trợ cả float [0,1] và uint8 [0,255].
    """
    data    = np.load(npz_path)
    rgb_arr = data["arr_0"]                    # (N, H, W, 3)

    if rgb_arr.dtype != np.uint8:
        # float [0,1] hoặc [0,255] → chuẩn hóa về uint8
        if rgb_arr.max() <= 1.0:
            rgb_arr = (rgb_arr * 255).clip(0, 255)
        rgb_arr = rgb_arr.astype(np.uint8)

    return rgb_arr


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def upscale(img_hw3: np.ndarray, zoom: int) -> np.ndarray:
    """Nearest-neighbour upscale (H, W, C) uint8 by integer factor."""
    return img_hw3.repeat(zoom, axis=0).repeat(zoom, axis=1)


def save_grid(rgb_arr: np.ndarray, results: list, indices: list,
              out_path: str, title: str, n: int = 100):
    """
    Lưu grid ảnh với:
      - scale-up ×ZOOM để nhìn rõ (16→160px)
      - 2 đường đỏ dọc chia 3 cột shape
      - caption: index + hall_type + blobs
    """
    n = min(n, len(indices))
    if n == 0:
        print(f"  No images for: {title}")
        return

    cols = 10
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.2))
    axes = np.array(axes).flatten()

    for i, idx in enumerate(indices[:n]):
        img_up = upscale(rgb_arr[idx], ZOOM)
        axes[i].imshow(img_up, vmin=0, vmax=255)

        # Đường đỏ chia cột
        for x_boundary in [5 * ZOOM, 10 * ZOOM]:
            axes[i].axvline(x=x_boundary, color="red", linewidth=0.8, alpha=0.9)

        r      = results[idx]
        blobs  = r["col_blobs"]
        b_str  = f"t{blobs['triangle']} s{blobs['square']} p{blobs['pentagon']}"
        h_type = r["hall_type"] if r["is_hallucination"] else "ok"
        axes[i].set_title(f"#{idx} [{h_type}]\n{b_str}", fontsize=5.5)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def save_blob_histogram(results: list, out_path: str):
    """Bar chart phân bố số blob ở từng cột."""
    N   = len(results)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, name in zip(axes, COLUMN_NAMES):
        counts  = np.array([r["col_blobs"][name] for r in results])
        max_val = max(int(counts.max()), 2)
        bins    = np.arange(0, max_val + 2) - 0.5
        ax.hist(counts, bins=bins, color="steelblue", edgecolor="white", rwidth=0.8)
        ax.set_xlabel("shapes detected")
        ax.set_ylabel("# images")
        ax.set_title(f"{name} column")
        ax.set_xticks(range(max_val + 1))
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Blob count distribution  (N={N})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",     required=True,
                        help="Path to samples .npz  (arr_0: N×H×W×3)")
    parser.add_argument("--out_dir", default="hallucination_analysis_16x16")
    parser.add_argument("--top_n",   type=int, default=100,
                        help="Max images per grid (default: 100)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load ──
    print(f"Loading {args.npz} ...")
    rgb_arr = load_samples(args.npz)
    N       = len(rgb_arr)
    print(f"Total images: {N}")

    # ── Detect ──
    print("Analysing ...")
    results = analyze_batch(rgb_arr)          # list[dict]
    s       = summarize(results)              # aggregate stats

    # ── Print results ──
    print(f"\n=== Results ===")
    print(f"Hallucinations : {s['n_hall']} / {N}  ({100*s['hall_rate']:.2f}%)")
    print(f"  ├─ empty image (0 shapes)  : {s['n_empty']}")
    print(f"  └─ double col (2+ in 1 col): {s['n_double_col']}")
    print(f"Normal         : {s['n_normal']} / {N}")
    print()
    print(f"  {'column':10s} {'0 shapes':>10} {'1 shape':>10} {'2+ shapes':>10}")
    for name in COLUMN_NAMES:
        cc = s["col_counts"][name]
        print(f"  {name:10s} {cc['0']:>10} {cc['1']:>10} {cc['2+']:>10}")

    # ── Grids ──
    scores = np.array([r["score"] for r in results])

    # Hallucination: tất cả loại (sort by score giảm dần)
    hall_sorted = sorted(s["hall_indices"], key=lambda i: -results[i]["score"])
    save_grid(
        rgb_arr, results, hall_sorted,
        out_path=os.path.join(args.out_dir, "hallucinations_all.png"),
        title=f"All hallucinations  ({s['n_hall']}/{N}  {100*s['hall_rate']:.2f}%)",
        n=args.top_n,
    )

    # Empty images
    save_grid(
        rgb_arr, results, s["empty_indices"],
        out_path=os.path.join(args.out_dir, "hallucinations_empty.png"),
        title=f"Empty images — no shapes at all  ({s['n_empty']}/{N})",
        n=args.top_n,
    )

    # Double-col images
    save_grid(
        rgb_arr, results, s["double_col_indices"],
        out_path=os.path.join(args.out_dir, "hallucinations_double_col.png"),
        title=f"Double-col — 2+ shapes in same column  ({s['n_double_col']}/{N})",
        n=args.top_n,
    )

    # Normal samples
    norm_sorted = sorted(
        [i for i, r in enumerate(results) if not r["is_hallucination"]],
        key=lambda i: scores[i],
    )
    save_grid(
        rgb_arr, results, norm_sorted,
        out_path=os.path.join(args.out_dir, "normal.png"),
        title=f"Normal samples  ({s['n_normal']}/{N})",
        n=args.top_n,
    )

    # ── Histogram ──
    save_blob_histogram(
        results,
        out_path=os.path.join(args.out_dir, "blob_histogram.png"),
    )

    # ── Indices files ──
    np.savetxt(
        os.path.join(args.out_dir, "hallucination_indices.txt"),
        s["hall_indices"], fmt="%d",
    )
    np.savetxt(
        os.path.join(args.out_dir, "empty_indices.txt"),
        s["empty_indices"], fmt="%d",
    )
    np.savetxt(
        os.path.join(args.out_dir, "double_col_indices.txt"),
        s["double_col_indices"], fmt="%d",
    )
    print(f"  Saved: hallucination_indices.txt / empty_indices.txt / double_col_indices.txt")


if __name__ == "__main__":
    main()
