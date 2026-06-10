"""
Phân tích x_T của hallucination cases:
  1. Column-wise noise statistics — cột hallucinate có noise bất thường không?
  2. Basin of attraction — perturbation nhỏ bao nhiêu thì "chữa" được hallucination?

Usage:
    python scripts/analyze_noise_basin.py \
        --traj_root simple-shapes-5k-16x16-output/num_channel_64/hallucination_trajectories \
        --model_path simple-shapes-5k-16x16-output/num_channel_64/checkpoints/ema_0.9999_200000.pt \
        --image_size 16 --num_channels 64 --num_res_blocks 3 \
        --diffusion_steps 1000 --noise_schedule linear \
        --timestep_respacing 100
"""

import argparse
import os
import sys

import numpy as np
import torch as th
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DETECTOR_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "flow_matching")
)
sys.path.insert(0, _DETECTOR_PATH)

from improved_diffusion import dist_util
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from hallucination_detector import analyze_image, COLUMN_NAMES

COL_SLICES = [(0, 5), (5, 10), (10, 15)]   # triangle, square, pentagon


# ── DDIM inference ────────────────────────────────────────────────────────────

@th.no_grad()
def ddim_sample(model, diffusion, x_T_np, device):
    """Run DDIM from given x_T (numpy float32 CHW). Returns uint8 HWC numpy."""
    x_T = th.from_numpy(x_T_np).unsqueeze(0).to(device)   # (1,3,16,16)
    out = diffusion.ddim_sample_loop(
        model,
        shape=x_T.shape,
        noise=x_T,
        clip_denoised=True,
        model_kwargs={},
        device=device,
        eta=0.0,
    )
    img = ((out + 1) * 127.5).clamp(0, 255).to(th.uint8)
    return img[0].permute(1, 2, 0).cpu().numpy()   # (H,W,3)


# ── Column-wise noise analysis ────────────────────────────────────────────────

def col_noise_stats(noise_chw: np.ndarray, hall_col: str | None):
    """
    noise_chw: (3,16,16) float32 ~ N(0,1)
    Trả về stats per column (mean |z|, max |z|, cluster score)
    và highlight cột bị hallucinate.
    """
    # Dùng channel 0 để phân tích (3 channel giống nhau vì unstructured noise)
    noise_hw = noise_chw[0]   # (16,16)

    print(f"  {'col':12s} {'mean|z|':>8} {'max|z|':>8} {'cluster':>8}  ← hall?")
    for (c0, c1), name in zip(COL_SLICES, COLUMN_NAMES):
        col   = noise_hw[:, c0:c1].flatten()
        mz    = float(np.abs(col).mean())
        maxz  = float(np.abs(col).max())

        # Cluster score: liệu các pixel dương có tập trung thành 1 vùng không?
        # → cộng nhau những pixel có z > 0.5 tạo thành blob
        positive = (noise_hw[:, c0:c1] > 0.5).astype(np.uint8)
        from scipy import ndimage
        labeled, n_blobs = ndimage.label(positive)
        max_blob = max(
            (labeled == i).sum() for i in range(1, n_blobs + 1)
        ) if n_blobs > 0 else 0
        cluster = float(max_blob)   # lớn → noise có cụm pixel dương lớn

        is_hall = "  ← HALL" if name == hall_col else ""
        print(f"  {name:12s} {mz:>8.3f} {maxz:>8.3f} {cluster:>8.0f}{is_hall}")


# ── Basin of attraction sweep ─────────────────────────────────────────────────

def basin_sweep(model, diffusion, x_T_orig: np.ndarray,
                device, n_trials: int = 10,
                sigmas=(0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)):
    """
    Với mỗi sigma, sample n_trials perturbations:
        x_T_perturbed = x_T_orig + N(0, sigma^2)
    Đếm bao nhiêu vẫn còn hallucinate.

    Trả về dict: sigma → (n_hall, n_trials)
    """
    results = {}
    for sigma in sigmas:
        n_hall = 0
        for _ in range(n_trials):
            perturb = np.random.randn(*x_T_orig.shape).astype(np.float32) * sigma
            x_T_new = x_T_orig + perturb
            img = ddim_sample(model, diffusion, x_T_new, device)
            if analyze_image(img)["is_hallucination"]:
                n_hall += 1
        results[sigma] = (n_hall, n_trials)
    return results


# ── Column-targeted perturbation ──────────────────────────────────────────────

def col_targeted_sweep(model, diffusion, x_T_orig: np.ndarray,
                       hall_col: str, device, n_trials: int = 10,
                       sigmas=(0.1, 0.3, 0.5, 1.0, 2.0)):
    """
    Chỉ perturb pixels trong cột bị hallucinate (col-specific).
    → Xem cột đó có "nhạy cảm" đặc biệt không.
    """
    col_idx = COLUMN_NAMES.index(hall_col)
    c0, c1  = COL_SLICES[col_idx]

    results = {}
    for sigma in sigmas:
        n_hall = 0
        for _ in range(n_trials):
            x_T_new = x_T_orig.copy()
            # Chỉ perturb column bị hallucinate
            x_T_new[:, :, c0:c1] += (
                np.random.randn(3, 16, c1 - c0).astype(np.float32) * sigma
            )
            img = ddim_sample(model, diffusion, x_T_new, device)
            if analyze_image(img)["is_hallucination"]:
                n_hall += 1
        results[sigma] = (n_hall, n_trials)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_root",  required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--n_trials",   type=int, default=20,
                        help="Perturbation trials per sigma level")
    defaults = model_and_diffusion_defaults()
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()

    dist_util.setup_dist()
    device = dist_util.dev()

    # ── Load model ──
    print("Loading model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}\n")

    # ── Load trajectories ──
    sample_dirs = sorted([
        os.path.join(args.traj_root, d)
        for d in os.listdir(args.traj_root)
        if os.path.isdir(os.path.join(args.traj_root, d))
        and os.path.exists(os.path.join(args.traj_root, d, "trajectory.npz"))
    ])

    for sample_dir in sample_dirs:
        sid      = os.path.basename(sample_dir)
        noise    = np.load(os.path.join(sample_dir, "noise.npy"))   # (3,16,16)
        traj     = np.load(os.path.join(sample_dir, "trajectory.npz"))
        final_img = traj["final"]   # (16,16,3) uint8
        final_res = analyze_image(final_img)

        # Xác định cột bị hallucinate
        hall_col = next(
            (n for n in COLUMN_NAMES if final_res["col_blobs"][n] >= 2), None
        )

        print(f"{'='*65}")
        print(f"Sample: {sid}")
        print(f"  hall_type={final_res['hall_type']}  "
              f"col_blobs={final_res['col_blobs']}")
        print(f"  Hallucinating column: {hall_col}")

        # ── Verify: re-run DDIM với noise gốc ──
        img_verify = ddim_sample(model, diffusion, noise, device)
        r_verify   = analyze_image(img_verify)
        print(f"  Verify re-run: hall={r_verify['is_hallucination']}  "
              f"blobs={r_verify['col_blobs']}")

        # ── 1. Column-wise noise stats ──
        print(f"\n  [1] Column-wise noise statistics (x_T channel 0):")
        col_noise_stats(noise, hall_col)

        # ── 2. Global perturbation sweep ──
        print(f"\n  [2] Global perturbation basin  (n_trials={args.n_trials} per sigma):")
        print(f"  {'sigma':>8}  {'hall':>6} / {'total':>5}  {'hall%':>7}")
        global_res = basin_sweep(
            model, diffusion, noise, device,
            n_trials=args.n_trials,
            sigmas=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
        )
        for sigma, (nh, nt) in global_res.items():
            bar = "█" * nh + "░" * (nt - nh)
            print(f"  σ={sigma:>5.2f}  {nh:>6} / {nt:>5}  {100*nh/nt:>6.0f}%  {bar}")

        # ── 3. Column-targeted perturbation ──
        if hall_col is not None:
            print(f"\n  [3] Column-targeted perturbation  (only '{hall_col}' col):")
            print(f"  {'sigma':>8}  {'hall':>6} / {'total':>5}  {'hall%':>7}")
            col_res = col_targeted_sweep(
                model, diffusion, noise, hall_col, device,
                n_trials=args.n_trials,
                sigmas=(0.1, 0.3, 0.5, 1.0, 2.0, 3.0),
            )
            for sigma, (nh, nt) in col_res.items():
                bar = "█" * nh + "░" * (nt - nh)
                print(f"  σ={sigma:>5.2f}  {nh:>6} / {nt:>5}  {100*nh/nt:>6.0f}%  {bar}")

        print()


if __name__ == "__main__":
    main()
