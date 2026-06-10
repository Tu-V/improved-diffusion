"""
Phân tích onset timestep của hallucination trong trajectory.

Với mỗi hallucination sample:
  1. Tìm t* = bước DDIM đầu tiên pred_x0 bị hallucinate
  2. Đo giá trị pixel tại vị trí "proto-shape" lúc t*
  3. Đo binarization gap: phần trăm pixel nằm trong vùng ambiguous

Usage:
    python scripts/analyze_onset.py \
        --traj_root simple-shapes-5k-16x16-output/num_channel_64/hallucination_trajectories
"""

import argparse
import os
import sys
import numpy as np

_DETECTOR_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "flow_matching")
)
sys.path.insert(0, _DETECTOR_PATH)
from hallucination_detector import analyze_image, COLUMN_NAMES

# Column x-ranges cho 16x16
COL_SLICES = [(0, 5), (5, 10), (10, 15)]

AMBIGUOUS_LO = 20    # pixel > này mà < HI → ambiguous (không rõ shape hay background)
AMBIGUOUS_HI = 235


def binarization_gap(img_hw3: np.ndarray) -> float:
    """
    Đo khoảng cách trung bình giữa continuous output và binary ideal.
    img_hw3: uint8 [0,255] → convert về float → snap về {0,255} → đo gap.
    """
    f = img_hw3[:, :, 0].astype(np.float32)
    binary = np.where(f > 127.5, 255.0, 0.0)
    return float(np.abs(f - binary).mean())


def ambiguous_ratio(img_hw3: np.ndarray) -> float:
    """Tỉ lệ pixel nằm trong vùng không rõ ràng (không gần 0 hay 255)."""
    p = img_hw3[:, :, 0]
    return float(np.mean((p > AMBIGUOUS_LO) & (p < AMBIGUOUS_HI)))


def find_hallucinated_col(result: dict) -> str | None:
    """Trả về tên cột bị hallucinate (có >= 2 shapes), hoặc None."""
    for name in COLUMN_NAMES:
        if result["col_blobs"][name] >= 2:
            return name
    return None


def analyze_sample(sample_dir: str) -> dict:
    npz_path = os.path.join(sample_dir, "trajectory.npz")
    data = np.load(npz_path)

    x_t    = data["x_t"]      # (T, H, W, 3) uint8
    pred   = data["pred_x0"]  # (T, H, W, 3) uint8
    T = x_t.shape[0]

    sample_id = os.path.basename(sample_dir)

    # ── 1. Tìm onset timestep t* ──────────────────────────────────────────
    onset_t     = None
    onset_col   = None
    onset_blobs = None

    for t in range(T):
        res = analyze_image(pred[t])
        if res["is_hallucination"]:
            onset_t     = t
            onset_col   = find_hallucinated_col(res)
            onset_blobs = res["col_blobs"]
            break

    # ── 2. Pixel intensity của proto-shape tại t* ─────────────────────────
    proto_pixel_mean = None
    if onset_t is not None and onset_col is not None:
        col_idx = COLUMN_NAMES.index(onset_col)
        c0, c1  = COL_SLICES[col_idx]
        # Vùng pixel trong cột bị hallucinate, tại bước x_t (noisy image)
        col_region = x_t[onset_t, :, c0:c1, 0].astype(float)
        proto_pixel_mean = float(col_region.mean())

    # ── 3. Binarization gap và ambiguous ratio tại mỗi bước ──────────────
    gaps       = [binarization_gap(pred[t]) for t in range(T)]
    amb_ratios = [ambiguous_ratio(pred[t])  for t in range(T)]

    # ── 4. Final image ──
    final_res = analyze_image(pred[T - 1])

    return {
        "sample":           sample_id,
        "onset_t":          onset_t,           # None nếu không bao giờ hall trong pred_x0
        "onset_col":        onset_col,
        "onset_blobs":      onset_blobs,
        "proto_pixel_mean": proto_pixel_mean,  # mean intensity trong cột ở x_t tại t*
        "gap_at_onset":     gaps[onset_t] if onset_t is not None else None,
        "gap_final":        gaps[T - 1],
        "gap_mean":         float(np.mean(gaps)),
        "amb_at_onset":     amb_ratios[onset_t] if onset_t is not None else None,
        "amb_final":        amb_ratios[T - 1],
        "total_steps":      T,
        "final_hall":       final_res["is_hallucination"],
        "final_hall_type":  final_res["hall_type"],
        "final_col_blobs":  final_res["col_blobs"],
    }


def print_result(r: dict):
    print(f"\n{'='*60}")
    print(f"Sample: {r['sample']}")
    print(f"  Final hallucination : {r['final_hall']}  ({r['final_hall_type']})")
    print(f"  Final col_blobs     : {r['final_col_blobs']}")

    if r["onset_t"] is not None:
        pct = 100 * r["onset_t"] / r["total_steps"]
        print(f"\n  Onset step (t*)     : {r['onset_t']} / {r['total_steps']}  "
              f"({pct:.0f}% through denoising)")
        print(f"  Onset col           : {r['onset_col']}")
        print(f"  Onset blobs         : {r['onset_blobs']}")
        print(f"  x_t pixel mean in   ")
        print(f"    hallucinated col  : {r['proto_pixel_mean']:.1f}  "
              f"(uint8, higher → brighter proto-shape)")
    else:
        print(f"\n  Onset step (t*)     : not found in pred_x0 (hall only at final?)")

    print(f"\n  Binarization gap")
    print(f"    at onset (t*)     : {r['gap_at_onset']:.2f}" if r["gap_at_onset"] is not None
          else "    at onset (t*)     : N/A")
    print(f"    at final step     : {r['gap_final']:.2f}")
    print(f"    mean over traj    : {r['gap_mean']:.2f}")
    print(f"  Ambiguous ratio")
    print(f"    at onset (t*)     : {r['amb_at_onset']:.3f}" if r["amb_at_onset"] is not None
          else "    at onset (t*)     : N/A")
    print(f"    at final step     : {r['amb_final']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_root", required=True,
                        help="Folder chứa các sample_*/ subdirs")
    args = parser.parse_args()

    sample_dirs = sorted([
        os.path.join(args.traj_root, d)
        for d in os.listdir(args.traj_root)
        if os.path.isdir(os.path.join(args.traj_root, d))
        and os.path.exists(os.path.join(args.traj_root, d, "trajectory.npz"))
    ])

    if not sample_dirs:
        print(f"No trajectory samples found in {args.traj_root}")
        return

    print(f"Analyzing {len(sample_dirs)} hallucination trajectories...\n")

    all_results = []
    for d in sample_dirs:
        r = analyze_sample(d)
        print_result(r)
        all_results.append(r)

    # ── Tổng hợp ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    onset_steps = [r["onset_t"] for r in all_results if r["onset_t"] is not None]
    proto_means = [r["proto_pixel_mean"] for r in all_results if r["proto_pixel_mean"] is not None]
    gaps        = [r["gap_at_onset"] for r in all_results if r["gap_at_onset"] is not None]
    ambs        = [r["amb_at_onset"] for r in all_results if r["amb_at_onset"] is not None]

    if onset_steps:
        T = all_results[0]["total_steps"]
        print(f"Onset t* (% through denoising): "
              f"mean={np.mean(onset_steps)/T*100:.0f}%  "
              f"range=[{min(onset_steps)},{max(onset_steps)}]")
    if proto_means:
        print(f"x_t pixel mean at onset col   : "
              f"mean={np.mean(proto_means):.1f}  "
              f"range=[{min(proto_means):.1f},{max(proto_means):.1f}]")
        print(f"  (background should be ~0-50, shape ~200-255)")
    if gaps:
        print(f"Binarization gap at onset     : mean={np.mean(gaps):.2f}")
    if ambs:
        print(f"Ambiguous pixel ratio at onset: mean={np.mean(ambs):.3f}")


if __name__ == "__main__":
    main()
