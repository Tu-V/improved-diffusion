"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.

Hallucination tracking (--save_hallucination_traj):
  If enabled, each generated image is checked for hallucination (2+ blobs in
  the same column). For hallucinated images, the full denoising trajectory
  (initial noise + every N steps) is saved to a separate folder for analysis.
"""

import argparse
import os
import sys

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

# ── Import shared hallucination detector ─────────────────────────────────────
_DETECTOR_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "flow_matching")
)
sys.path.insert(0, _DETECTOR_PATH)
from hallucination_detector import analyze_image as _analyze_image  # noqa: E402


def is_hallucination(img_uint8):
    """img_uint8: (H, W, 3) uint8. Returns True if hallucination detected."""
    return _analyze_image(img_uint8)["is_hallucination"]


# ---------------------------------------------------------------------------
# Trajectory sampling: collect intermediate steps for a single batch
# ---------------------------------------------------------------------------

def sample_with_trajectory(diffusion, model, shape, clip_denoised, model_kwargs, save_every_n_steps):
    """
    Run p_sample_loop_progressive, collect:
      - initial noise (x_T)
      - x_t every save_every_n_steps timesteps
      - pred_x0 every save_every_n_steps timesteps
      - final sample (x_0)

    Returns:
      final_sample : (N, C, H, W) float tensor
      trajectory   : list of {"t": int, "x_t": np.array (N,H,W,C) uint8,
                                         "pred_x0": np.array (N,H,W,C) uint8}
      initial_noise: (N, C, H, W) float tensor
    """
    trajectory = []
    initial_noise = None
    final_sample  = None
    step_count    = 0

    for out in diffusion.p_sample_loop_progressive(
        model, shape,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
        device=dist_util.dev(),
    ):
        img   = out["sample"]        # (N, C, H, W)
        pred  = out["pred_xstart"]   # (N, C, H, W)

        # First call: save initial noise (we already passed through T→T-1,
        # so back-compute: initial noise was the img before this first step)
        if initial_noise is None:
            # p_sample_loop_progressive starts from random noise → first img is x_{T-1}
            # We capture pred_xstart at T as a proxy for noise characteristics
            initial_noise = img.cpu()

        if step_count % save_every_n_steps == 0 or step_count == 0:
            def to_uint8_nhwc(t):
                return (((t + 1) * 127.5).clamp(0, 255)
                        .to(th.uint8).permute(0, 2, 3, 1).cpu().numpy())
            trajectory.append({
                "t": step_count,
                "x_t":     to_uint8_nhwc(img),
                "pred_x0": to_uint8_nhwc(pred),
            })

        final_sample = img
        step_count  += 1

    # Always include the very last step
    def to_uint8_nhwc(t):
        return (((t + 1) * 127.5).clamp(0, 255)
                .to(th.uint8).permute(0, 2, 3, 1).cpu().numpy())
    if trajectory[-1]["t"] != step_count - 1:
        trajectory.append({
            "t": step_count - 1,
            "x_t":     to_uint8_nhwc(final_sample),
            "pred_x0": to_uint8_nhwc(final_sample),
        })

    return final_sample, trajectory, initial_noise


# ---------------------------------------------------------------------------
# Save trajectory for a single hallucinated image
# ---------------------------------------------------------------------------

def save_trajectory(traj_dir, sample_id, img_uint8, trajectory, initial_noise):
    """
    Save trajectory of a single hallucinated image.
    Layout:
      traj_dir/
        sample_{id:05d}/
          final.png
          noise.npy          <- initial noise tensor
          step_000_xt.png    <- x_t at each saved step
          step_000_pred.png  <- pred_x0 at each saved step
          ...
          trajectory.npz     <- all arrays in one file
    """
    from PIL import Image as PILImage

    out = os.path.join(traj_dir, f"sample_{sample_id:05d}")
    os.makedirs(out, exist_ok=True)

    # Final image
    PILImage.fromarray(img_uint8).save(os.path.join(out, "final.png"))

    # Initial noise
    np.save(os.path.join(out, "noise.npy"), initial_noise.numpy())

    # Stitch all steps into 2 rows (x_t on top, pred_x0 on bottom), one column per step
    import math
    steps_xt   = [step["x_t"][0]    for step in trajectory]
    steps_pred = [step["pred_x0"][0] for step in trajectory]
    n_steps    = len(steps_xt)
    H, W, C    = steps_xt[0].shape

    grid = np.zeros((H * 2, W * n_steps, C), dtype=np.uint8)
    for i, (xt, pred) in enumerate(zip(steps_xt, steps_pred)):
        grid[0:H,   i*W:(i+1)*W] = xt
        grid[H:H*2, i*W:(i+1)*W] = pred

    PILImage.fromarray(grid).save(os.path.join(out, "trajectory.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    traj_dir = None
    if args.save_hallucination_traj and dist.get_rank() == 0:
        traj_dir = os.path.join(logger.get_dir(), "hallucination_trajectories")
        os.makedirs(traj_dir, exist_ok=True)
        logger.log(f"Hallucination trajectory tracking ON -> {traj_dir}")

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
    all_images   = []
    all_labels   = []
    save_interval = 1000
    last_saved    = 0
    hall_count    = 0
    global_idx    = 0   # tracks sample index across batches

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        # --- Sample (with or without trajectory) ---
        if args.save_hallucination_traj:
            sample_tensor, trajectory, initial_noise = sample_with_trajectory(
                diffusion, model,
                shape=(args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                save_every_n_steps=args.traj_save_every,
            )
            sample = ((sample_tensor + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1).contiguous()
        else:
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                progress=(dist.get_rank() == 0),
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1).contiguous()
            trajectory    = None
            initial_noise = None

        # --- Gather across GPUs ---
        gathered_samples = [th.zeros_like(sample.to(th.int32)) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample.to(th.int32))
        batch_imgs = [s.to(th.uint8).cpu().numpy() for s in gathered_samples]
        all_images.extend(batch_imgs)

        if args.class_cond:
            gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([l.cpu().numpy() for l in gathered_labels])

        # --- Hallucination detection + trajectory saving (rank 0 only) ---
        if args.save_hallucination_traj and dist.get_rank() == 0:
            for rank_imgs in batch_imgs:
                for i in range(len(rank_imgs)):
                    img = rank_imgs[i]
                    if is_hallucination(img):
                        hall_count += 1
                        if trajectory is not None and i < args.batch_size:
                            # Slice trajectory for this specific image in batch
                            img_traj = [{
                                "t": s["t"],
                                "x_t":     s["x_t"][i:i+1],
                                "pred_x0": s["pred_x0"][i:i+1],
                            } for s in trajectory]
                            save_trajectory(traj_dir, global_idx, img, img_traj, initial_noise[i])
                    global_idx += 1

        num_so_far = len(all_images) * args.batch_size
        logger.log(f"created {num_so_far} samples" +
                   (f" | hallucinations: {hall_count}" if args.save_hallucination_traj else ""))

        # --- Periodic save ---
        if dist.get_rank() == 0 and num_so_far - last_saved >= save_interval:
            arr_so_far = np.concatenate(all_images, axis=0)
            shape_str  = "x".join([str(x) for x in arr_so_far.shape])
            out_path   = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
            logger.log(f"saving checkpoint to {out_path}")
            if args.class_cond:
                np.savez(out_path, arr_so_far, np.concatenate(all_labels, axis=0))
            else:
                np.savez(out_path, arr_so_far)
            last_saved = num_so_far

    # --- Final save ---
    arr = np.concatenate(all_images, axis=0)[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path  = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, np.concatenate(all_labels, axis=0)[: args.num_samples])
        else:
            np.savez(out_path, arr)
        if args.save_hallucination_traj:
            logger.log(f"Total hallucinations saved: {hall_count} / {args.num_samples}")

    if dist.get_world_size() > 1:
        dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        save_hallucination_traj=False,  # flag: save trajectory for hallucinated images
        traj_save_every=10,             # save x_t every N denoising steps
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
