"""
Sample 1000 perturbations từ noise gốc, đếm tỉ lệ hallucination,
và lưu ~10 representative samples (noise heatmap + final image) vào 1 ảnh.

Usage:
    python scripts/sample_perturbations.py \\
        --sample_dir simple-shapes-5k-16x16-output/num_channel_64/hallucination_trajectories/sample_02696 \\
        --model_path simple-shapes-5k-16x16-output/num_channel_64/checkpoints/ema_0.9999_200000.pt \\
        --image_size 16 --num_channels 64 --num_res_blocks 3 \\
        --diffusion_steps 1000 --noise_schedule linear \\
        --attention_resolutions 8 --timestep_respacing ddim25 \\
        --n_samples 1000 --sigma 0.05 --n_show 10
"""

import argparse
import os
import sys

import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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


@th.no_grad()
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
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--n_samples",  type=int,   default=1000,
                        help="Tổng số perturbation để sample")
    parser.add_argument("--sigma",      type=float, default=0.05,
                        help="Sigma của perturbation noise")
    parser.add_argument("--n_show",     type=int,   default=10,
                        help="Số samples đại diện để hiển thị")
    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    args = parser.parse_args()

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
    print(f"Model on {device}")

    sid      = os.path.basename(args.sample_dir)
    out_dir  = os.path.join(args.sample_dir, "perturbation_samples")
    os.makedirs(out_dir, exist_ok=True)

    eps0 = np.load(os.path.join(args.sample_dir, "noise.npy"))  # (3,16,16)

    # ── Verify eps0 ──────────────────────────────────────────────────────────
    img0 = ddim_sample(model, diffusion, eps0, device)
    r0   = analyze_image(img0)
    print(f"\neps0 (original): hall={r0['is_hallucination']}  "
          f"type={r0['hall_type']}  blobs={r0['col_blobs']}")

    # ── Main sweep ──────────────────────────────────────────────────────────
    print(f"\nSampling {args.n_samples} perturbations (σ={args.sigma})...")
    print("-" * 60)

    hall_records   = []   # (noise, img, result)  — hallucinations
    ok_records     = []   # (noise, img, result)  — non-hallucinations
    n_hall         = 0
    n_show_half    = args.n_show // 2   # bao nhiêu hall / ok mỗi loại

    for i in range(args.n_samples):
        perturb = np.random.randn(*eps0.shape).astype(np.float32) * args.sigma
        x_T     = eps0 + perturb
        img     = ddim_sample(model, diffusion, x_T, device)
        r       = analyze_image(img)

        if r["is_hallucination"]:
            n_hall += 1
            if len(hall_records) < n_show_half:
                hall_records.append((x_T.copy(), img, r))
        else:
            if len(ok_records) < n_show_half:
                ok_records.append((x_T.copy(), img, r))

        if (i + 1) % 100 == 0:
            print(f"  [{i+1:>4}/{args.n_samples}]  hall so far: {n_hall}", flush=True)

    rate = 100 * n_hall / args.n_samples
    print(f"\nResult: {n_hall}/{args.n_samples} = {rate:.1f}% hallucination  (σ={args.sigma})")

    # ── Build display list: hall samples first, then ok ─────────────────────
    display = []
    for noise, img, r in hall_records:
        display.append(("HALL", noise, img, r))
    for noise, img, r in ok_records:
        display.append(("ok", noise, img, r))

    # Thêm eps0 gốc ở đầu để so sánh
    display.insert(0, ("orig", eps0, img0, r0))

    n_disp = len(display)

    # ── Layout: 2 hàng × n_disp cột ─────────────────────────────────────────
    # Hàng 1: noise heatmap (channel 0)
    # Hàng 2: final image (upscaled)
    H_img = upscale(display[0][2]).shape[0]
    W_img = upscale(display[0][2]).shape[1]

    fig, axes = plt.subplots(
        2, n_disp,
        figsize=(n_disp * 1.8, 5.5),
        squeeze=False,
    )

    col_x = [W_img // 3, 2 * W_img // 3]   # column boundaries trên ảnh upscaled

    for col, (label, noise, img, r) in enumerate(display):
        # ── Row 0: noise heatmap ──
        ax_n = axes[0, col]
        im   = ax_n.imshow(noise[0], cmap="RdBu_r", vmin=-3, vmax=3,
                           interpolation="nearest")
        is_h  = r["is_hallucination"]
        color = "red" if is_h else ("gold" if label == "orig" else "limegreen")
        ax_n.set_title(label, fontsize=8, color=color, fontweight="bold")
        for spine in ax_n.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
        ax_n.set_xticks([])
        ax_n.set_yticks([])
        if col == 0:
            ax_n.set_ylabel("noise\n(ch0)", fontsize=8)

        # ── Row 1: final image ──
        ax_i = axes[1, col]
        ax_i.imshow(upscale(img))
        for x in col_x:
            ax_i.axvline(x=x, color="red", linewidth=0.8, alpha=0.7)
        cb   = r["col_blobs"]
        htype = r["hall_type"] if is_h else "ok"
        ax_i.set_title(
            f"{htype}\nt{cb['triangle']}s{cb['square']}p{cb['pentagon']}",
            fontsize=7,
            color=color,
        )
        for spine in ax_i.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
        ax_i.axis("off")
        if col == 0:
            axes[1, col].set_ylabel("image", fontsize=8)

    # ── Colorbar for noise ──
    plt.colorbar(im, ax=axes[0, -1], fraction=0.08, pad=0.04, label="z-score")

    # ── Title & legend ──
    hall_patch = mpatches.Patch(color="red",       label="hallucination")
    ok_patch   = mpatches.Patch(color="limegreen", label="ok")
    orig_patch = mpatches.Patch(color="gold",      label="original eps0")

    fig.suptitle(
        f"{sid}  |  σ={args.sigma}  |  "
        f"{n_hall}/{args.n_samples} = {rate:.1f}% hallucination\n"
        f"(showing {len(hall_records)} hall + {len(ok_records)} ok + original)",
        fontsize=10,
    )
    fig.legend(handles=[hall_patch, ok_patch, orig_patch],
               loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = os.path.join(out_dir, f"samples_sigma{args.sigma:.3f}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
