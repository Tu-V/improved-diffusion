"""
Print trajectory.npz as formatted numpy arrays (1 channel) to a text file.

Usage:
    python scripts/print_trajectory.py \
        --traj path/to/trajectory.npz \
        --out  trajectory_arrays.txt \
        --channel 0
"""

import argparse
import os
import numpy as np


def fmt_matrix(arr_hw, step_label):
    """Format a (H, W) uint8 matrix as aligned text."""
    lines = [step_label]
    lines.append("-" * (arr_hw.shape[1] * 4))
    for row in arr_hw:
        lines.append(" ".join(f"{v:3d}" for v in row))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj",    required=True, help="Path to trajectory.npz")
    parser.add_argument("--out",     default="trajectory_arrays.txt")
    parser.add_argument("--channel", type=int, default=0,
                        help="Which RGB channel to print (0=R, 1=G, 2=B)")
    parser.add_argument("--every",   type=int, default=10,
                        help="Print every N steps (default: 10)")
    args = parser.parse_args()

    data = np.load(args.traj)
    print("Keys in npz:", list(data.keys()))
    for k in data.keys():
        print(f"  {k}: shape={data[k].shape}  dtype={data[k].dtype}")
    print()

    # ── x_t: (T, H, W, C) uint8 ──
    x_t    = data["x_t"]       # denoising steps
    pred_x0 = data["pred_x0"]  # model's x0 prediction at each step
    T = x_t.shape[0]
    ch = args.channel
    ch_name = ["R", "G", "B"][ch]

    blocks = []

    blocks.append(f"{'='*60}")
    blocks.append(f"Trajectory: {args.traj}")
    blocks.append(f"Steps: {T}   Image: {x_t.shape[1]}x{x_t.shape[2]}   Channel: {ch_name} ({ch})")
    blocks.append(f"{'='*60}\n")

    # Always include first, last, and every N-th step
    selected = sorted(set(
        list(range(0, T, args.every)) + [T - 1]
    ))

    for t in selected:
        xt_hw   = x_t[t,    :, :, ch]
        pred_hw = pred_x0[t, :, :, ch]

        blocks.append(fmt_matrix(xt_hw,   f"[step {t:>3d}/{T-1}]  x_t     (channel {ch_name})"))
        blocks.append("")
        blocks.append(fmt_matrix(pred_hw, f"[step {t:>3d}/{T-1}]  pred_x0 (channel {ch_name})"))
        blocks.append("")

    # ── final ──
    if "final" in data:
        final = data["final"][:, :, ch]
        blocks.append(fmt_matrix(final, f"[FINAL]  x_0  (channel {ch_name})"))
        blocks.append("")

    text = "\n".join(blocks)

    with open(args.out, "w") as f:
        f.write(text)

    print(text)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
