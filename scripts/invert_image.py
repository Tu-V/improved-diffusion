"""
DDIM inversion: recover the denoising trajectory of a known final image.

Given a final image x_0, DDIM inversion runs the reverse ODE forwards in
time (t=0 → t=T), recovering x_1, x_2, ..., x_T step by step.
Because DDIM (eta=0) is deterministic, this exactly reconstructs the noise
trajectory that would have produced x_0 during sampling.

At each step we also record pred_xstart = what the model currently predicts
x_0 to be.  This lets you see at which noise level the hallucination is
still "visible" to the model.

Outputs
-------
  trajectory.png          - visual grid: row 0 = x_t, row 1 = pred_xstart
  steps_arrays.txt        - numpy matrix of each x_t (grayscale, uint8)
  steps.npz               - all x_t and pred_xstart arrays
  hallucination_trace.txt - per-step hallucination status of pred_xstart

Usage
-----
  python scripts/invert_image.py \\
      --model_path ema_0.9999_200000.pt \\
      --npz samples_10000x16x16x3.npz --index 42 \\
      --image_size 16 --num_channels 64 --num_res_blocks 3 \\
      --diffusion_steps 1000 --noise_schedule linear \\
      --timestep_respacing 100 \\
      --out_dir inversion_out
"""

import argparse
import os
import sys

import numpy as np
import torch as th
from PIL import Image

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

# ── Import shared hallucination detector ─────────────────────────────────────
_DETECTOR_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "flow_matching")
)
sys.path.insert(0, _DETECTOR_PATH)
from hallucination_detector import (  # noqa: E402
    analyze_image as _analyze_image, COLUMN_NAMES,
)


def is_hallucination(img_uint8):
    """
    img_uint8: (H, W, 3) uint8 RGB.
    Returns (is_hall: bool, blobs: dict {col_name: count}).
    """
    result = _analyze_image(img_uint8)
    return result["is_hallucination"], result["col_blobs"]


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def to_uint8_hwc(tensor_chw):
    """(C,H,W) float [-1,1] → (H,W,C) uint8."""
    return (((tensor_chw + 1) * 127.5).clamp(0, 255)
            .to(th.uint8).permute(1, 2, 0).cpu().numpy())


def to_float_nchw(img_hw3_uint8, device):
    """(H,W,3) uint8 → (1,3,H,W) float32 in [-1,1]."""
    t = th.from_numpy(img_hw3_uint8).float() / 127.5 - 1.0  # (H,W,3)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)        # (1,3,H,W)


# ---------------------------------------------------------------------------
# DDIM inversion loop
# ---------------------------------------------------------------------------

def ddim_inversion(model, diffusion, x0_tensor, device):
    """
    Run DDIM reverse ODE: x_0 → x_1 → ... → x_T.

    Returns list of dicts with keys: t, x_t (uint8 HWC), pred_xstart (uint8 HWC).
    """
    steps = []
    img   = x0_tensor.clone()   # (1, C, H, W)
    T     = diffusion.num_timesteps

    for t_idx in range(T):
        t_tensor = th.tensor([t_idx] * img.shape[0], device=device)
        with th.no_grad():
            out = diffusion.ddim_reverse_sample(
                model, img, t_tensor,
                clip_denoised=False,   # don't clip — x_t will leave [-1,1] as noise grows
            )
        steps.append({
            "t":           t_idx,
            "x_t":         to_uint8_hwc(img[0]),
            "pred_xstart": to_uint8_hwc(out["pred_xstart"][0]),
        })
        img = out["sample"]

    # Final x_T (pure noise)
    steps.append({
        "t":           T,
        "x_t":         to_uint8_hwc(img[0].clamp(-1, 1)),
        "pred_xstart": to_uint8_hwc(img[0].clamp(-1, 1)),
    })
    return steps


# ---------------------------------------------------------------------------
# Save trajectory grid PNG  (upscaled for visibility)
# ---------------------------------------------------------------------------
ZOOM = 10

def upscale(img, zoom):
    return img.repeat(zoom, axis=0).repeat(zoom, axis=1)


def save_trajectory_png(steps, out_path, title="DDIM Inversion Trajectory"):
    import matplotlib.pyplot as plt

    n  = len(steps)
    H  = steps[0]["x_t"].shape[0] * ZOOM
    W  = steps[0]["x_t"].shape[1] * ZOOM

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 4))

    for col, s in enumerate(steps):
        xt   = upscale(s["x_t"],         ZOOM)
        pred = upscale(s["pred_xstart"], ZOOM)

        for row, img in enumerate([xt, pred]):
            ax = axes[row, col]
            ax.imshow(img[:, :, 0], cmap="gray", vmin=0, vmax=255)
            for xb in [5 * ZOOM, 10 * ZOOM]:
                ax.axvline(x=xb, color="red", linewidth=0.5, alpha=0.8)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"t={s['t']}", fontsize=6)

    axes[0, 0].set_ylabel("x_t",         fontsize=7)
    axes[1, 0].set_ylabel("pred_xstart", fontsize=7)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Print / save numpy arrays
# ---------------------------------------------------------------------------

def print_and_save_arrays(steps, txt_path):
    lines = []
    for s in steps:
        hall, blobs = is_hallucination(s["x_t"])   # s["x_t"] is (H, W, 3) uint8
        blob_str = "  ".join(f"{n}:{blobs[n]}" for n in COLUMN_NAMES)
        header = f"=== t={s['t']:>3}  |  blobs: {blob_str}  |  {'HALLUCINATION' if hall else 'ok'} ==="
        mat    = np.array2string(gray, separator=",",
                                 formatter={"int": lambda x: f"{x:3d}"})
        lines.append(header)
        lines.append(mat)
        lines.append("")

    text = "\n".join(lines)

    # Print to stdout
    print(text)

    # Save to file
    with open(txt_path, "w") as f:
        f.write(text)
    print(f"Saved: {txt_path}")


# ---------------------------------------------------------------------------
# Save hallucination trace
# ---------------------------------------------------------------------------

def save_hallucination_trace(steps, trace_path):
    lines = ["step  | triangle | square | pentagon | hallucination"]
    lines.append("-" * 55)
    for s in steps:
        hall, blobs = is_hallucination(s["pred_xstart"])  # (H, W, 3) uint8
        row = (f"{s['t']:>4}  | "
               f"{blobs['triangle']:>8} | "
               f"{blobs['square']:>6} | "
               f"{blobs['pentagon']:>8} | "
               f"{'*** YES ***' if hall else 'no'}")
        lines.append(row)

    text = "\n".join(lines)
    print("\n--- Hallucination trace (pred_xstart at each step) ---")
    print(text)
    with open(trace_path, "w") as f:
        f.write(text)
    print(f"\nSaved: {trace_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    device = dist_util.dev()

    # --- Load model ---
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(device)
    model.eval()

    # --- Load image ---
    if args.npz:
        data   = np.load(args.npz)
        arr    = data["arr_0"]            # (N, H, W, 3) uint8
        img_np = arr[args.index]          # (H, W, 3)
        print(f"Loaded image #{args.index} from {args.npz}  shape={img_np.shape}")
    elif args.png:
        img_np = np.array(Image.open(args.png).convert("RGB"))
        print(f"Loaded image from {args.png}  shape={img_np.shape}")
    else:
        sys.exit("Provide --npz+--index or --png")

    # Initial hallucination check
    hall0, blobs0 = is_hallucination(img_np)
    print(f"Input image hallucination: {'YES' if hall0 else 'NO'}  blobs={blobs0}")

    # --- DDIM inversion ---
    x0_tensor = to_float_nchw(img_np, device)
    logger.log(f"Running DDIM inversion ({diffusion.num_timesteps} steps)...")
    steps = ddim_inversion(model, diffusion, x0_tensor, device)
    logger.log(f"Inversion complete. Total steps recorded: {len(steps)}")

    # --- Save outputs ---
    os.makedirs(args.out_dir, exist_ok=True)

    save_trajectory_png(
        steps,
        out_path=os.path.join(args.out_dir, "trajectory.png"),
        title=f"DDIM Inversion  img#{args.index}  {'[HALLUCINATION]' if hall0 else '[normal]'}",
    )

    print_and_save_arrays(
        steps,
        txt_path=os.path.join(args.out_dir, "steps_arrays.txt"),
    )

    save_hallucination_trace(
        steps,
        trace_path=os.path.join(args.out_dir, "hallucination_trace.txt"),
    )

    # Save all arrays to npz for further analysis
    npz_out = os.path.join(args.out_dir, "steps.npz")
    np.savez(npz_out,
             x_t     =np.stack([s["x_t"]         for s in steps]),   # (T+1,H,W,3)
             pred_x0 =np.stack([s["pred_xstart"]  for s in steps]),   # (T+1,H,W,3)
             timesteps=np.array([s["t"]           for s in steps]))
    print(f"Saved: {npz_out}")


def create_argparser():
    defaults = dict(
        npz="",
        png="",
        index=0,
        out_dir="inversion_out",
        clip_denoised=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
