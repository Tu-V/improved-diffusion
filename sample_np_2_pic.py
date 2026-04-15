from PIL import Image
import numpy as np

data = np.load("/Users/admin/workspace/improved-diffusion/simple-shapes-5k-checkpoints-sampling/samples_2000x64x64x3.npz")
arr = data["arr_0"]

for i, img in enumerate(arr):
    Image.fromarray(img).save(f"/Users/admin/workspace/improved-diffusion/simple-shapes-5k-checkpoints-sampling/sample_{i:03d}.png")
