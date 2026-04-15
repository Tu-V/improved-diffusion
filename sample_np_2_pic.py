import os
import glob
import numpy as np
from PIL import Image

base_dir = "/Users/admin/workspace/improved-diffusion"

for npz_path in sorted(glob.glob(os.path.join(base_dir, "*sampling*", "samples_2000x*.npz"))):
    folder = os.path.dirname(npz_path)
    out_dir = os.path.join(folder, "images")
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(npz_path)
    arr = data["arr_0"]
    print(f"{npz_path} -> {len(arr)} images -> {out_dir}")

    for i, img in enumerate(arr):
        Image.fromarray(img).save(os.path.join(out_dir, f"sample_{i:05d}.png"))
