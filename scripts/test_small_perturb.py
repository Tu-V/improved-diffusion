"""
Test: noise rất gần x_T hallucination → có còn hallucinate không?

Usage:
    python scripts/test_small_perturb.py \
        --sample_dir simple-shapes-5k-16x16-output/num_channel_64/hallucination_trajectories/sample_02696 \
        --model_path simple-shapes-5k-16x16-output/num_channel_64/checkpoints/ema_0.9999_200000.pt \
        --image_size 16 --num_channels 64 --num_res_blocks 3 \
        --diffusion_steps 1000 --noise_schedule linear \
        --attention_resolutions 8 --timestep_respacing ddim25 \
        --n_trials 20
"""

import argparse, os, sys
import numpy as np
import torch as th
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DETECTOR_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "flow_matching")
)
sys.path.insert(0, _DETECTOR_PATH)

from improved_diffusion import dist_util
from improved_diffusion.script_util import (
    model_and_diffusion_defaults, create_model_and_diffusion,
    args_to_dict, add_dict_to_argparser,
)
from hallucination_detector import analyze_image, COLUMN_NAMES

ZOOM = 8


def ddim_sample(model, diffusion, x_T_np, device):
    x_T = th.from_numpy(x_T_np).unsqueeze(0).to(device)
    out = diffusion.ddim_sample_loop(
        model, shape=x_T.shape, noise=x_T,
        clip_denoised=True, model_kwargs={}, device=device, eta=0.0,
    )
    img = ((out + 1) * 127.5).clamp(0, 255).to(th.uint8)
    return img[0].permute(1, 2, 0).cpu().numpy()


def upscale(img, zoom=ZOOM):
    return img.repeat(zoom, axis=0).repeat(zoom, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--sigmas", type=str,
                        default="0.0,0.001,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--no_grid", action="store_true",
                        help="Skip saving image grid (faster for large n_trials)")
    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    args = parser.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",")]

    dist_util.setup_dist()
    device = dist_util.dev()

    print("Loading model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(device)
    model.eval()

    noise_orig = np.load(os.path.join(args.sample_dir, "noise.npy"))  # (3,16,16)
    sid = os.path.basename(args.sample_dir)
    print(f"\nSample: {sid}")
    print(f"{'sigma':>8}  {'hall':>5}/{args.n_trials}  {'rate':>6}  bar")
    print("-" * 50)

    # Lưu 1 ảnh đại diện mỗi sigma để visualize
    grid_imgs  = []   # list of (H,W,3) uint8
    grid_labels = []

    for sigma in sigmas:
        n_hall = 0
        sample_imgs = []

        for trial in range(args.n_trials):
            if sigma == 0.0:
                x_T = noise_orig.copy()
            else:
                perturb = np.random.randn(*noise_orig.shape).astype(np.float32) * sigma
                x_T = noise_orig + perturb

            img = ddim_sample(model, diffusion, x_T, device)
            r   = analyze_image(img)
            if r["is_hallucination"]:
                n_hall += 1
            if trial == 0 and not args.no_grid:
                sample_imgs.append((img, r))

            if (trial + 1) % 100 == 0:
                print(f"    ... {trial+1}/{args.n_trials}  hall so far: {n_hall}", flush=True)

        rate = n_hall / args.n_trials
        bar  = "█" * n_hall + "░" * (args.n_trials - n_hall)
        print(f"σ={sigma:>7.3f}  {n_hall:>5}/{args.n_trials}  {100*rate:>5.0f}%  {bar}")

        # Lấy ảnh trial đầu để grid
        if not args.no_grid and sample_imgs:
            img0, r0 = sample_imgs[0]
            grid_imgs.append(img0)
            h_str = r0["hall_type"] if r0["is_hallucination"] else "ok"
            cb    = r0["col_blobs"]
            grid_labels.append(
                f"σ={sigma}\n{h_str}\n"
                f"t{cb['triangle']}s{cb['square']}p{cb['pentagon']}\n"
                f"{n_hall}/{args.n_trials} hall"
            )

    # ── Save visualization grid ──
    if not args.no_grid and grid_imgs:
        n = len(sigmas)
        fig, axes = plt.subplots(1, n, figsize=(n * 2.2, 3.5), squeeze=False)
        axes = axes[0]   # (1, n) → (n,)
        W = grid_imgs[0].shape[1]
        col_x = [W // 3 * ZOOM, 2 * W // 3 * ZOOM]

        for ax, img, label in zip(axes, grid_imgs, grid_labels):
            ax.imshow(upscale(img))
            for x in col_x:
                ax.axvline(x=x, color="red", linewidth=1.0, alpha=0.8)
            ax.set_title(label, fontsize=7)
            ax.axis("off")

        fig.suptitle(f"{sid} — small perturbation test  (1 trial shown, n={args.n_trials} for rate)",
                     fontsize=9)
        plt.tight_layout()
        out_path = os.path.join(args.sample_dir, "small_perturb_test.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nGrid saved → {out_path}")


if __name__ == "__main__":
    main()
