"""
DDIM inversion: trace ngược từ ảnh hallucination → noise x_T.

Pipeline:
  1. Load hallucination images từ npz (dùng hallucination_indices.txt)
  2. Chạy DDIM inversion (x_0 → x_T) dùng ddim_reverse_sample
  3. Verify: decode lại x_T → x_0, kiểm tra hallucination còn ko
  4. Lưu noise.npy + ảnh verify vào thư mục output

Usage:
    python scripts/invert_hallucination.py \\
        --npz simple-shapes-50k-16x16-output/num_channel_64/sampling/checkpoint-200000s-sampling-25steps-ddim/samples_30000x16x16x3.npz \\
        --indices_txt simple-shapes-50k-16x16-output/num_channel_64/sampling/checkpoint-200000s-sampling-25steps-ddim/hallucination_analysis/hallucination_indices.txt \\
        --model_path simple-shapes-50k-16x16-output/num_channel_64/checkpoints/ema_0.9999_200000.pt \\
        --image_size 16 --num_channels 64 --num_res_blocks 3 \\
        --diffusion_steps 1000 --noise_schedule linear \\
        --attention_resolutions 8 --timestep_respacing ddim25 \\
        --out_dir simple-shapes-50k-16x16-output/num_channel_64/sampling/checkpoint-200000s-sampling-25steps-ddim/inverted_noises
"""

import argparse
import os
import sys

import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

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
from hallucination_detector import analyze_image

ZOOM = 8


def upscale(img, zoom=ZOOM):
    return img.repeat(zoom, axis=0).repeat(zoom, axis=1)


@th.no_grad()
def ddim_invert(model, diffusion, x0_np, device):
    """
    DDIM inversion: x_0 (image) → x_T (noise).
    Chạy ddim_reverse_sample từ t=0 lên t=T-1.
    Returns x_T as numpy float32 (C, H, W), normalized ~ N(0,1).
    """
    # x0_np: (H, W, 3) uint8 → normalize về [-1, 1] rồi thành (1, 3, H, W)
    x0 = th.from_numpy(x0_np).float().permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    x0 = (x0 / 127.5 - 1.0).to(device)

    # Inversion: đi từ t=0 → t=T, dùng respaced index tăng dần (0, 1, ..., num_timesteps-1)
    # ddim_sample_loop dùng index ngược [T-1..0], inversion dùng [0..T-1]
    x = x0
    for t_val in range(diffusion.num_timesteps):   # 0, 1, ..., 24 với ddim25
        t_batch = th.tensor([t_val], device=device)
        out = diffusion.ddim_reverse_sample(
            model, x, t_batch,
            clip_denoised=False,
            model_kwargs={},
            eta=0.0,
        )
        x = out["sample"]

    # x bây giờ là x_T trong không gian [-1,1] scale của model
    # Chuyển về CHW numpy float32
    x_T = x[0].cpu().numpy()   # (3, H, W)
    return x_T


@th.no_grad()
def ddim_decode(model, diffusion, x_T_np, device):
    """x_T (C,H,W) float32 → decoded image (H,W,3) uint8."""
    x_T = th.from_numpy(x_T_np).unsqueeze(0).to(device)
    out = diffusion.ddim_sample_loop(
        model, shape=x_T.shape, noise=x_T,
        clip_denoised=True, model_kwargs={}, device=device, eta=0.0,
    )
    img = ((out + 1) * 127.5).clamp(0, 255).to(th.uint8)
    return img[0].permute(1, 2, 0).cpu().numpy()


def save_summary_grid(records, out_path, title):
    """
    records: list of dict với keys: idx, orig_img, recon_img, x_T, r_orig, r_recon
    Layout: mỗi sample = 3 cột: [noise heatmap | original | reconstructed]
    """
    n = len(records)
    fig, axes = plt.subplots(3, n, figsize=(n * 2.5, 8), squeeze=False)

    W = records[0]["orig_img"].shape[1]
    col_x = [W * ZOOM // 3, 2 * W * ZOOM // 3]

    for col, rec in enumerate(records):
        idx       = rec["idx"]
        orig_img  = rec["orig_img"]
        recon_img = rec["recon_img"]
        x_T       = rec["x_T"]
        r_orig    = rec["r_orig"]
        r_recon   = rec["r_recon"]

        # Row 0: noise heatmap (channel 0)
        ax = axes[0, col]
        im = ax.imshow(x_T[0], cmap="RdBu_r", vmin=-3, vmax=3,
                       interpolation="nearest")
        ax.set_title(f"#{idx}\ninverted noise\n(ch0)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        # Row 1: original hallucination image
        ax = axes[1, col]
        ax.imshow(upscale(orig_img))
        for x in col_x:
            ax.axvline(x=x, color="red", linewidth=1.0, alpha=0.8)
        cb = r_orig["col_blobs"]
        ax.set_title(
            f"original\n{r_orig['hall_type']}\n"
            f"t{cb['triangle']}s{cb['square']}p{cb['pentagon']}",
            fontsize=7, color="red",
        )
        ax.axis("off")

        # Row 2: reconstructed (verify)
        ax = axes[2, col]
        ax.imshow(upscale(recon_img))
        for x in col_x:
            ax.axvline(x=x, color="red", linewidth=1.0, alpha=0.8)
        cb2 = r_recon["col_blobs"]
        color = "red" if r_recon["is_hallucination"] else "green"
        ax.set_title(
            f"reconstructed\n{r_recon['hall_type']}\n"
            f"t{cb2['triangle']}s{cb2['square']}p{cb2['pentagon']}",
            fontsize=7, color=color,
        )
        ax.axis("off")

    axes[0, 0].set_ylabel("noise x_T", fontsize=9)
    axes[1, 0].set_ylabel("original", fontsize=9)
    axes[2, 0].set_ylabel("reconstructed", fontsize=9)

    plt.colorbar(im, ax=axes[0, -1], fraction=0.08, pad=0.04)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grid saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",          required=True,
                        help="Path to samples npz")
    parser.add_argument("--indices_txt",  default=None,
                        help="hallucination_indices.txt từ detect_hallucinations.py. "
                             "Nếu không có, dùng --indices trực tiếp.")
    parser.add_argument("--indices",      type=str, default=None,
                        help="Comma-separated indices, ví dụ: 123,456,789")
    parser.add_argument("--model_path",   required=True)
    parser.add_argument("--out_dir",      default="inverted_noises")
    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    args = parser.parse_args()

    # ── Load indices ──────────────────────────────────────────────────────────
    if args.indices_txt:
        hall_indices = np.loadtxt(args.indices_txt, dtype=int).tolist()
        if isinstance(hall_indices, int):
            hall_indices = [hall_indices]
    elif args.indices:
        hall_indices = [int(x) for x in args.indices.split(",")]
    else:
        raise ValueError("Cần --indices_txt hoặc --indices")

    print(f"Hallucination indices: {hall_indices}")

    # ── Load model ────────────────────────────────────────────────────────────
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

    # ── Load samples ──────────────────────────────────────────────────────────
    data = np.load(args.npz)
    imgs = data["arr_0"]   # (N, H, W, 3) uint8

    os.makedirs(args.out_dir, exist_ok=True)

    records = []
    for idx in hall_indices:
        orig_img = imgs[idx]   # (H, W, 3) uint8
        r_orig   = analyze_image(orig_img)
        print(f"\n[#{idx}] hall_type={r_orig['hall_type']}  "
              f"blobs={r_orig['col_blobs']}")

        # ── DDIM inversion ──
        print(f"  Inverting...")
        x_T = ddim_invert(model, diffusion, orig_img, device)
        print(f"  x_T: shape={x_T.shape}  "
              f"mean={x_T.mean():.3f}  std={x_T.std():.3f}  "
              f"norm={np.linalg.norm(x_T):.2f}")

        # ── Verify: decode lại ──
        print(f"  Decoding x_T → verify...")
        recon_img = ddim_decode(model, diffusion, x_T, device)
        r_recon   = analyze_image(recon_img)
        print(f"  Verify: hall={r_recon['is_hallucination']}  "
              f"type={r_recon['hall_type']}  blobs={r_recon['col_blobs']}")

        # ── Pixel diff ──
        diff = np.abs(orig_img.astype(int) - recon_img.astype(int)).mean()
        print(f"  Pixel diff (orig vs recon): {diff:.2f}")

        # ── Save noise ──
        sample_dir = os.path.join(args.out_dir, f"sample_{idx:05d}")
        os.makedirs(sample_dir, exist_ok=True)
        np.save(os.path.join(sample_dir, "noise.npy"), x_T)
        PILImage.fromarray(orig_img).save(
            os.path.join(sample_dir, "original.png"))
        PILImage.fromarray(recon_img).save(
            os.path.join(sample_dir, "reconstructed.png"))
        print(f"  Saved → {sample_dir}/")

        records.append({
            "idx":       idx,
            "orig_img":  orig_img,
            "recon_img": recon_img,
            "x_T":       x_T,
            "r_orig":    r_orig,
            "r_recon":   r_recon,
        })

    # ── Summary grid ──────────────────────────────────────────────────────────
    if records:
        n_verified = sum(1 for r in records if r["r_recon"]["is_hallucination"])
        save_summary_grid(
            records,
            out_path=os.path.join(args.out_dir, "inversion_summary.png"),
            title=(
                f"DDIM Inversion — {len(records)} hallucination samples\n"
                f"Verify: {n_verified}/{len(records)} still hallucinate after invert→decode"
            ),
        )

    print(f"\nDone. noise.npy files saved in {args.out_dir}/sample_XXXXX/")


if __name__ == "__main__":
    main()
