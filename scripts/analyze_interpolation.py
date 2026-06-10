"""
Interpolation test giữa hai hallucinating noises eps0 và eps1.

Tìm eps1 (cũng sinh hallucination), rồi sample các điểm trên đoạn thẳng:
    eps_alpha = alpha * eps0 + (1 - alpha) * eps1
Đếm bao nhiêu trong số n_alphas điểm đó vẫn bị hallucinate.

Lưu:
  interpolation_test/
    eps0.npy, eps1.npy
    eps0_heatmap.png, eps1_heatmap.png
    img_eps0.png, img_eps1.png
    hallucination_profile.png   ← hall/ok theo alpha
    interpolation_grid.png      ← 11 ảnh đại diện dọc interpolation
    all_samples_grid.png        ← toàn bộ n_alphas ảnh (tiny, xem pattern)

Usage:
    python scripts/analyze_interpolation.py \\
        --sample_dir simple-shapes-5k-16x16-output/num_channel_64/hallucination_trajectories/sample_02696 \\
        --model_path simple-shapes-5k-16x16-output/num_channel_64/checkpoints/ema_0.9999_200000.pt \\
        --image_size 16 --num_channels 64 --num_res_blocks 3 \\
        --diffusion_steps 1000 --noise_schedule linear \\
        --attention_resolutions 8 --timestep_respacing ddim25 \\
        --n_alphas 1000 --sigma_search 0.05
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
    return img[0].permute(1, 2, 0).cpu().numpy()   # (H,W,3) uint8


def upscale(img, zoom=ZOOM):
    return img.repeat(zoom, axis=0).repeat(zoom, axis=1)


def save_noise_heatmap(noise_chw, path, title="noise"):
    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(noise_chw[0], cmap="RdBu_r", vmin=-3, vmax=3,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def find_hallucinating_eps1(model, diffusion, eps0, device,
                            sigma=0.05, max_tries=500):
    """Tìm eps1 ≠ eps0 mà vẫn sinh hallucination."""
    print(f"Searching for eps1 (sigma={sigma}, max_tries={max_tries})...")
    for i in range(max_tries):
        perturb = np.random.randn(*eps0.shape).astype(np.float32) * sigma
        eps1 = eps0 + perturb
        img = ddim_sample(model, diffusion, eps1, device)
        r   = analyze_image(img)
        if r["is_hallucination"]:
            print(f"  Found eps1 at try #{i+1}  "
                  f"hall_type={r['hall_type']}  blobs={r['col_blobs']}")
            return eps1, img, r
    raise RuntimeError(
        f"Could not find hallucinating eps1 after {max_tries} tries. "
        "Try a smaller sigma_search."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir",   required=True)
    parser.add_argument("--model_path",   required=True)
    parser.add_argument("--n_alphas",     type=int,   default=1000,
                        help="Number of interpolation points in [0,1]")
    parser.add_argument("--sigma_search", type=float, default=0.05,
                        help="Perturbation sigma dùng để tìm eps1")
    parser.add_argument("--eps1_path",    type=str,   default=None,
                        help="Path tới eps1.npy có sẵn (bỏ qua bước search)")
    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    args = parser.parse_args()

    dist_util.setup_dist()
    device = dist_util.dev()

    # ── Load model ──────────────────────────────────────────────────────────
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

    sid     = os.path.basename(args.sample_dir)
    out_dir = os.path.join(args.sample_dir, "interpolation_test")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load & verify eps0 ──────────────────────────────────────────────────
    eps0 = np.load(os.path.join(args.sample_dir, "noise.npy"))  # (3,16,16)
    img0 = ddim_sample(model, diffusion, eps0, device)
    r0   = analyze_image(img0)
    print(f"\neps0: hall={r0['is_hallucination']}  "
          f"type={r0['hall_type']}  blobs={r0['col_blobs']}")
    if not r0["is_hallucination"]:
        raise RuntimeError("eps0 không hallucinate — kiểm tra lại sample_dir")

    # ── Find / load eps1 ────────────────────────────────────────────────────
    if args.eps1_path:
        eps1 = np.load(args.eps1_path)
        img1 = ddim_sample(model, diffusion, eps1, device)
        r1   = analyze_image(img1)
        print(f"eps1 (loaded): hall={r1['is_hallucination']}  "
              f"type={r1['hall_type']}  blobs={r1['col_blobs']}")
    else:
        eps1, img1, r1 = find_hallucinating_eps1(
            model, diffusion, eps0, device, sigma=args.sigma_search
        )

    l2 = float(np.linalg.norm(eps1 - eps0))
    print(f"\nL2(eps1 - eps0) = {l2:.4f}  "
          f"(‖eps0‖ = {np.linalg.norm(eps0):.2f})")

    # ── Save eps0, eps1 ─────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "eps0.npy"), eps0)
    np.save(os.path.join(out_dir, "eps1.npy"), eps1)
    save_noise_heatmap(eps0, os.path.join(out_dir, "eps0_heatmap.png"), "eps0")
    save_noise_heatmap(eps1, os.path.join(out_dir, "eps1_heatmap.png"), "eps1")
    PILImage.fromarray(img0).save(os.path.join(out_dir, "img_eps0.png"))
    PILImage.fromarray(img1).save(os.path.join(out_dir, "img_eps1.png"))

    # ── Interpolation sweep ─────────────────────────────────────────────────
    alphas = np.linspace(0.0, 1.0, args.n_alphas)
    records = []   # (alpha, is_hall, hall_type, col_blobs, img)

    print(f"\nInterpolation sweep: {args.n_alphas} points, α ∈ [0,1]")
    print(f"  eps_α = α·eps0 + (1-α)·eps1")
    print("-" * 60)

    n_hall = 0
    for i, alpha in enumerate(alphas):
        eps_a = alpha * eps0 + (1.0 - alpha) * eps1
        img_a = ddim_sample(model, diffusion, eps_a, device)
        r_a   = analyze_image(img_a)
        is_h  = r_a["is_hallucination"]
        if is_h:
            n_hall += 1
        records.append((float(alpha), is_h, r_a["hall_type"],
                        r_a["col_blobs"], img_a))

        if (i + 1) % 100 == 0:
            print(f"  [{i+1:>4}/{args.n_alphas}]  α={alpha:.3f}  "
                  f"hall so far: {n_hall}", flush=True)

    total = args.n_alphas
    rate  = 100 * n_hall / total
    print(f"\nResult: {n_hall}/{total} = {rate:.1f}% hallucination along segment")

    # ── 1. Hallucination profile ─────────────────────────────────────────────
    alpha_vals = [r[0] for r in records]
    hall_flags = [int(r[1]) for r in records]

    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.fill_between(alpha_vals, hall_flags, step="mid",
                    alpha=0.65, color="crimson", label="hallucination")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.2)
    ax.set_xlabel("α  (0 = eps1 side, 1 = eps0 side)")
    ax.set_ylabel("hall")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["ok", "hall"])
    ax.set_title(
        f"{sid} — interpolation α·eps0 + (1-α)·eps1\n"
        f"{n_hall}/{total} = {rate:.1f}% hallucinate  |  "
        f"L2(eps1-eps0)={l2:.3f}"
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hallucination_profile.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2. Image grid at 11 key alpha values ─────────────────────────────────
    vis_targets = np.linspace(0.0, 1.0, 11)
    key_idx = [int(np.argmin(np.abs(np.array(alpha_vals) - v)))
               for v in vis_targets]

    n_vis = len(key_idx)
    fig, axes = plt.subplots(1, n_vis, figsize=(n_vis * 2.2, 3.5))
    W      = records[0][4].shape[1]
    col_x  = [W // 3 * ZOOM, 2 * W // 3 * ZOOM]

    for ax, idx in zip(axes, key_idx):
        alpha, is_h, htype, blobs, img_a = records[idx]
        ax.imshow(upscale(img_a))
        for x in col_x:
            ax.axvline(x=x, color="red", linewidth=1.0, alpha=0.8)
        h_str = htype if is_h else "ok"
        cb    = blobs
        ax.set_title(
            f"α={alpha:.2f}\n{h_str}\n"
            f"t{cb['triangle']}s{cb['square']}p{cb['pentagon']}",
            fontsize=7,
        )
        # Red/green border để dễ thấy
        for spine in ax.spines.values():
            spine.set_edgecolor("red" if is_h else "green")
            spine.set_linewidth(3)
        ax.axis("off")

    fig.suptitle(
        f"{sid} — interpolation grid  ({n_hall}/{total} hall)",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "interpolation_grid.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── 3. All-samples overview grid ─────────────────────────────────────────
    # Xếp tất cả n_alphas ảnh theo hàng (50 ảnh/hàng), border đỏ/xanh
    row_size = 50
    n_rows   = (total + row_size - 1) // row_size
    fig, axes = plt.subplots(n_rows, row_size,
                             figsize=(row_size * 0.45, n_rows * 0.55))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for ri in range(n_rows):
        for ci in range(row_size):
            idx = ri * row_size + ci
            ax  = axes[ri, ci]
            if idx < total:
                _, is_h, _, _, img_a = records[idx]
                ax.imshow(img_a)
                color = "red" if is_h else "lime"
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
            ax.axis("off")

    fig.suptitle(
        f"{sid} — all {total} samples along interpolation\n"
        f"red=hall  green=ok   {n_hall}/{total} ({rate:.0f}%) hall",
        fontsize=9,
    )
    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join(out_dir, "all_samples_grid.png"),
                dpi=100, bbox_inches="tight")
    plt.close()

    print(f"\nFiles saved → {out_dir}/")
    print(f"  eps0.npy  eps1.npy")
    print(f"  eps0_heatmap.png  eps1_heatmap.png")
    print(f"  img_eps0.png  img_eps1.png")
    print(f"  hallucination_profile.png  ← tỉ lệ hall theo α")
    print(f"  interpolation_grid.png     ← 11 ảnh key alpha")
    print(f"  all_samples_grid.png       ← tất cả {total} ảnh")


if __name__ == "__main__":
    main()
