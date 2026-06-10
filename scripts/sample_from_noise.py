"""
Generate a denoising trajectory from a fixed input noise file.

Reads a .npy noise file (shape: C x H x W or H x W x C),
runs the full denoising process step by step, and saves every step
as a single PNG grid so you can see at which step hallucination appears.

With the same noise input:
  - Same trajectory only if --seed is fixed (p_sample adds noise each step).
  - Without --seed, each run produces a different trajectory even from
    the same starting noise.

Usage:
  python scripts/sample_from_noise.py \
      --model_path ema_0.9999_060000.pt \
      --noise_path hallucination_trajectories/sample_00042/noise.npy \
      --image_size 64 --num_channels 64 --num_res_blocks 3 \
      --diffusion_steps 1000 --noise_schedule linear \
      --timestep_respacing 100 \
      --seed 42 \
      --out_path traj_out.png
"""

import argparse
import os

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


def to_uint8_hwc(tensor):
    """Convert (C, H, W) float tensor in [-1,1] to (H, W, C) uint8."""
    return (((tensor + 1) * 127.5).clamp(0, 255)
            .to(th.uint8).permute(1, 2, 0).cpu().numpy())


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    # Load noise
    logger.log(f"loading noise from {args.noise_path} ...")
    noise_np = np.load(args.noise_path)
    noise_tensor = th.from_numpy(noise_np).float()

    # Normalize to (1, C, H, W) on device
    if noise_tensor.ndim == 3:
        # Could be (C,H,W) or (H,W,C)
        if noise_tensor.shape[-1] in (1, 3):   # (H,W,C)
            noise_tensor = noise_tensor.permute(2, 0, 1)
        noise_tensor = noise_tensor.unsqueeze(0)   # (1, C, H, W)
    noise_tensor = noise_tensor.to(dist_util.dev())

    # The saved noise is x_{T-1} (first output of progressive loop).
    # We scale it to reasonable noise range if needed.
    logger.log(f"noise shape: {noise_tensor.shape}, "
               f"range: [{noise_tensor.min():.2f}, {noise_tensor.max():.2f}]")

    logger.log(f"using DDIM with eta={args.eta} "
               f"({'deterministic' if args.eta == 0.0 else 'stochastic'})")

    # Collect all steps via DDIM — eta=0 means no noise added at each step
    logger.log("running DDIM denoising trajectory...")
    steps_xt   = []
    steps_pred = []

    import pdb; pdb.set_trace()


    for out in diffusion.ddim_sample_loop_progressive(
        model,
        noise_tensor.shape,
        noise=noise_tensor,
        clip_denoised=args.clip_denoised,
        device=dist_util.dev(),
        eta=args.eta,
    ):
        steps_xt.append(to_uint8_hwc(out["sample"][0]))
        steps_pred.append(to_uint8_hwc(out["pred_xstart"][0]))

    total = len(steps_xt)
    logger.log(f"total steps: {total}")

    # Pick num_frames evenly-spaced indices, always include last frame
    n_frames = min(args.num_frames, total)
    if n_frames >= total:
        indices = list(range(total))
    else:
        # evenly spaced, last frame always included
        indices = [int(round(i * (total - 1) / (n_frames - 1))) for i in range(n_frames)]
    logger.log(f"saving {len(indices)} frames: {indices}")

    # Build grid: row 0 = x_t, row 1 = pred_x0, one column per frame
    H, W, C = steps_xt[0].shape
    z       = args.zoom
    grid    = np.zeros((H * z * 2, W * z * len(indices), C), dtype=np.uint8)
    for col, idx in enumerate(indices):
        xt   = steps_xt[idx].repeat(z, axis=0).repeat(z, axis=1)
        pred = steps_pred[idx].repeat(z, axis=0).repeat(z, axis=1)
        grid[0:H*z,       col*W*z:(col+1)*W*z] = xt
        grid[H*z:H*z*2,   col*W*z:(col+1)*W*z] = pred

    out_path = args.out_path or os.path.join(logger.get_dir(), "trajectory.png")
    Image.fromarray(grid).save(out_path)
    logger.log(f"trajectory saved to {out_path}")

    # --- Debug: compare final image against saved trajectory.npz ---
    ref_npz = os.path.join(os.path.dirname(args.noise_path), "trajectory.npz")
    if os.path.exists(ref_npz):
        ref      = np.load(ref_npz)
        ref_final = ref["final"]                    # (H, W, C) uint8
        gen_final = steps_xt[-1]                    # (H, W, C) uint8
        np.set_printoptions(linewidth=200, formatter={"int": lambda x: f"{x:3d}"})
        print("[DEBUG] ref_final  channel-0:\n", ref_final[:, :, 0])
        print("[DEBUG] gen_final  channel-0:\n", gen_final[:, :, 0])
        match     = np.array_equal(ref_final, gen_final)
        max_diff  = int(np.abs(ref_final.astype(int) - gen_final.astype(int)).max())
        mean_diff = float(np.abs(ref_final.astype(int) - gen_final.astype(int)).mean())
        logger.log(
            f"[DEBUG] final vs trajectory.npz['final']:  "
            f"exact_match={match}  max_diff={max_diff}  mean_diff={mean_diff:.3f}"
        )
    else:
        logger.log(f"[DEBUG] no trajectory.npz found at {ref_npz}, skipping comparison")

    logger.log("done.")


def create_argparser():
    defaults = dict(
        noise_path="",
        out_path="",
        eta=0.0,           # 0.0 = fully deterministic DDIM, 1.0 = DDPM-like
        num_frames=4,      # how many steps to show in the output grid
        zoom=1,            # upscale factor for small images (e.g. 10 for 16x16)
        clip_denoised=True,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
