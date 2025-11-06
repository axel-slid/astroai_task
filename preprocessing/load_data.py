# %%

import os
from tensorflow.keras.datasets import cifar10
from PIL import Image


# %%

(x_train, _), (x_test, _) = cifar10.load_data()

images = list(x_train) + list(x_test)

out_dir = "raw"
os.makedirs(out_dir, exist_ok=True)

for i, img in enumerate(images, start=1):
    Image.fromarray(img).save(os.path.join(out_dir, f"{i}.png"))

# %%










# visualize random images import os, random
from PIL import Image
import matplotlib.pyplot as plt
import random

folder = "/home/dils/astroai/task/data/raw"
files = random.sample(os.listdir(folder), 9)

plt.figure(figsize=(6,6))
for i, fname in enumerate(files):
    img = Image.open(os.path.join(folder, fname))
    plt.subplot(3,3,i+1)
    plt.imshow(img)
    plt.title(fname)
    plt.axis('off')
plt.tight_layout()
plt.show()

# %%

# check if all are 32x32x3
import os
from PIL import Image
import numpy as np

folder = "/home/dils/astroai/task/data/raw"

all_valid = all(
    np.array(Image.open(os.path.join(folder, f)).convert("RGB")).shape == (32, 32, 3)
    for f in os.listdir(folder) if f.lower().endswith(".png")
)

print(all_valid)
# %%
