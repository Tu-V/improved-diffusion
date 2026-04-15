from PIL import Image
import numpy as np

data = np.load("/Users/admin/workspace/improved-diffusion/simple-shapes-5k-checkpoints-10000s-sampling-250steps/samples_2000x64x64x3.npz")
arr = data["arr_0"]

for i, img in enumerate(arr):
    Image.fromarray(img).save(f"/Users/admin/workspace/improved-diffusion/simple-shapes-5k-checkpoints-10000s-sampling-250steps/sample_{i:03d}.png")
