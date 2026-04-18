"""
Sample images from a 16x16 simple-shapes diffusion model using DDIM.

For every generated image:
  - Check if it is a hallucination (2+ shapes in the same column)
  - If yes, save the initial noise + full DDIM trajectory so you can replay it

Layout of saved trajectories:
  <OPENAI_LOGDIR>/
    samples_Nx16x16x3.npz          ← all N samples
    hallucination_trajectories/
      sample_00042/
        final.png                  ← final image  (upscaled 10×)
        noise.npy                  ← x_T  shape (3,16,16)  float32
        trajectory.png             ← grid: row0=x_t, row1=pred_xstart
        trajectory.npz             ← raw arrays for every step

Usage:
  mpiexec -n 4 python scripts/image_sample_16x16.py \\
      --model_path ema_0.9999_200000.pt \\
      --image_size 16 --num_channels 64 --num_res_blocks 3 \\
      --diffusion_steps 1000 --noise_schedule linear \\
      --num_samples 10000 --timestep_respacing 100 \\
      --batch_size 64
"""

import os

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image as PILImage
from scipy import ndimage

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


# ---------------------------------------------------------------------------
# Hallucination detection — tuned for 16x16  (5-px-wide columns)
# ---------------------------------------------------------------------------

COLUMN_SLICES     = [(0, 5), (5, 10), (10, 15)]
COLUMN_NAMES      = ["triangle", "square", "pentagon"]
BRIGHT_PERCENTILE = 80
MIN_SHAPE_AREA    = 3


def _count_blobs(col_img_f32):
    thresh = np.percentile(col_img_f32, BRIGHT_PERCENTILE)
    if thresh >= col_img_f32.max():
        return 0
    binary = (col_img_f32 >= thresh).astype(bool)
    struct = ndimage.generate_binary_structure(2, 2)
    labeled, n = ndimage.label(binary, structure=struct)
    return sum(1 for r in range(1, n + 1)
               if (labeled == r).sum() >= MIN_SHAPE_AREA)


def check_hallucination(img_uint8):
    """
    img_uint8: (H, W, 3) uint8.
    Returns (is_hall: bool, blobs: dict {col_name: count}).
    """
    gray  = img_uint8[:, :, 0].astype(np.float32)
    blobs = {name: _count_blobs(gray[:, c0:c1])
             for (c0, c1), name in zip(COLUMN_SLICES, COLUMN_NAMES)}
    return any(v >= 2 for v in blobs.values()), blobs


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

ZOOM = 10   # upscale factor for saved PNGs (16→160 px)


def _to_uint8(tensor_nchw):
    """(N,C,H,W) float [-1,1] → (N,H,W,C) uint8 numpy."""
    return (((tensor_nchw + 1) * 127.5).clamp(0, 255)
            .to(th.uint8).permute(0, 2, 3, 1).cpu().numpy())


def _upscale(img_hwc, zoom=ZOOM):
    return img_hwc.repeat(zoom, axis=0).repeat(zoom, axis=1)


# ---------------------------------------------------------------------------
# DDIM sampling with full trajectory capture
# ---------------------------------------------------------------------------

def sample_batch_ddim(diffusion, model, shape, clip_denoised, model_kwargs):
    """
    Run ddim_sample_loop_progressive for one batch.

    Returns
    -------
    final_uint8   : (N, H, W, 3) uint8 numpy — final samples
    initial_noise : (N, C, H, W) float32 numpy — x_T
    trajectory    : list of {"t", "x_t": (N,H,W,3) uint8, "pred_x0": same}
    """
    noise  = th.randn(*shape, device=dist_util.dev())   # x_T
    traj   = []
    sample = None

    for out in diffusion.ddim_sample_loop_progressive(
        model,
        shape,
        noise=noise,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
        device=dist_util.dev(),
        eta=0.0,
    ):
        traj.append({
            "t":      len(traj),
            "x_t":    _to_uint8(out["sample"]),
            "pred_x0": _to_uint8(out["pred_xstart"]),
        })
        sample = out["sample"]

    return _to_uint8(sample), noise.cpu().numpy(), traj


# ---------------------------------------------------------------------------
# Save one hallucinated sample's trajectory
# ---------------------------------------------------------------------------

def save_trajectory(out_dir, sample_id, final_uint8, noise_nchw, traj):
    """
    Save everything needed to replay the trajectory for one image.

    Parameters
    ----------
    final_uint8 : (H, W, 3) uint8
    noise_nchw  : (C, H, W) float32
    traj        : list from sample_batch_ddim, sliced to this image
    """
    folder = os.path.join(out_dir, f"sample_{sample_id:05d}")
    os.makedirs(folder, exist_ok=True)

    # --- Final image (upscaled) ---
    PILImage.fromarray(_upscale(final_uint8)).save(
        os.path.join(folder, "final.png"))

    # --- Initial noise ---
    np.save(os.path.join(folder, "noise.npy"), noise_nchw)

    # --- Trajectory PNG:  row0 = x_t,  row1 = pred_xstart ---
    steps_xt   = [s["x_t"]    for s in traj]
    steps_pred = [s["pred_x0"] for s in traj]
    n_steps    = len(steps_xt)
    H, W, C    = _upscale(steps_xt[0]).shape

    grid = np.zeros((H * 2, W * n_steps, C), dtype=np.uint8)
    for i, (xt, pred) in enumerate(zip(steps_xt, steps_pred)):
        grid[0:H,   i*W:(i+1)*W] = _upscale(xt)
        grid[H:H*2, i*W:(i+1)*W] = _upscale(pred)

    PILImage.fromarray(grid).save(os.path.join(folder, "trajectory.png"))

    # --- Raw arrays ---
    np.savez(
        os.path.join(folder, "trajectory.npz"),
        x_t      = np.stack(steps_xt),            # (T, H, W, C)
        pred_x0  = np.stack(steps_pred),
        timesteps= np.array([s["t"] for s in traj]),
        noise    = noise_nchw,
        final    = final_uint8,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    traj_dir   = None
    hall_count = 0
    if dist.get_rank() == 0:
        traj_dir = os.path.join(logger.get_dir(), "hallucination_trajectories")
        os.makedirs(traj_dir, exist_ok=True)
        logger.log(f"Hallucination trajectories → {traj_dir}")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")

    all_images  = []
    all_labels  = []
    save_interval = 1000
    last_saved    = 0
    global_idx    = 0    # sample index across all batches/ranks

    while len(all_images) * args.batch_size < args.num_samples:

        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES,
                size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        shape = (args.batch_size, 3, args.image_size, args.image_size)

        # ---- DDIM sampling with full trajectory ----
        final_uint8, initial_noise, traj = sample_batch_ddim(
            diffusion, model, shape,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        # final_uint8   : (N, H, W, 3)
        # initial_noise : (N, C, H, W)
        # traj          : list of T steps, each with (N,H,W,3) arrays

        # ---- Gather across GPUs ----
        sample_th = th.from_numpy(final_uint8).to(dist_util.dev()).to(th.int32)
        gathered  = [th.zeros_like(sample_th) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, sample_th)
        batch_imgs = [g.to(th.uint8).cpu().numpy() for g in gathered]
        all_images.extend(batch_imgs)

        if args.class_cond:
            gathered_lbl = [th.zeros_like(classes)
                            for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_lbl, classes)
            all_labels.extend([l.cpu().numpy() for l in gathered_lbl])

        # ---- Hallucination check + save trajectory (rank 0 only) ----
        if dist.get_rank() == 0:
            for rank_idx, rank_imgs in enumerate(batch_imgs):
                for i, img in enumerate(rank_imgs):
                    is_hall, blobs = check_hallucination(img)
                    if is_hall:
                        hall_count += 1
                        logger.log(
                            f"  Hallucination #{hall_count}  "
                            f"(global sample {global_idx})  "
                            f"blobs={blobs}"
                        )
                        # only rank-0's own batch has trajectory arrays;
                        # other ranks' images don't have trajectory on this process
                        if rank_idx == 0 and i < args.batch_size:
                            img_traj = [
                                {
                                    "t":       s["t"],
                                    "x_t":     s["x_t"][i],      # (H,W,3)
                                    "pred_x0": s["pred_x0"][i],  # (H,W,3)
                                }
                                for s in traj
                            ]
                            save_trajectory(
                                traj_dir, global_idx,
                                img,
                                initial_noise[i],   # (C,H,W)
                                img_traj,
                            )
                    global_idx += 1

        num_so_far = len(all_images) * args.batch_size
        logger.log(
            f"created {num_so_far} samples"
            f" | hallucinations so far: {hall_count}"
        )

        # ---- Periodic save ----
        if dist.get_rank() == 0 and num_so_far - last_saved >= save_interval:
            arr        = np.concatenate(all_images, axis=0)
            shape_str  = "x".join(str(x) for x in arr.shape)
            out_path   = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
            logger.log(f"periodic save → {out_path}")
            np.savez(out_path, arr)
            last_saved = num_so_far

    # ---- Final save ----
    arr = np.concatenate(all_images, axis=0)[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join(str(x) for x in arr.shape)
        out_path  = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"final save → {out_path}")
        np.savez(out_path, arr)
        logger.log(
            f"Done.  Total hallucinations: {hall_count} / {args.num_samples} "
            f"({100*hall_count/args.num_samples:.1f}%)"
        )

    if dist.get_world_size() > 1:
        dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    import argparse
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=64,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
