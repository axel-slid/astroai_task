# %%

import os
import numpy as np
from PIL import Image

raw_dir = "/home/dils/astroai/task/data/raw"
split_dir = "/home/dils/astroai/task/data/split"
x_dir = os.path.join(split_dir, "x")
y_dir = os.path.join(split_dir, "y")

os.makedirs(x_dir, exist_ok=True)
os.makedirs(y_dir, exist_ok=True)

for fname in sorted(os.listdir(raw_dir)):

    path = os.path.join(raw_dir, fname)
    img = np.array(Image.open(path).convert("RGB"))  # shape: (32, 32, 3)

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    rb = np.stack((r, b), axis=-1)

    Image.fromarray(np.dstack((r, np.zeros_like(r), b)).astype(np.uint8)).save(
        os.path.join(x_dir, fname)
    )

    Image.fromarray(g.astype(np.uint8)).save(os.path.join(y_dir, fname))


# %%


# visualize the RB and G images



import os, random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

x_dir = "/home/dils/astroai/task/data/split/x"
y_dir = "/home/dils/astroai/task/data/split/y"

files = sorted(os.listdir(x_dir))
sample = random.sample(files, 9)

# --- R+B images ---
plt.figure(figsize=(6,6))
for i, fname in enumerate(sample):
    img = np.array(Image.open(os.path.join(x_dir, fname)))
    plt.subplot(3,3,i+1)
    plt.imshow(img)
    plt.title(fname)
    plt.axis('off')
plt.suptitle("R + B Channel Images", fontsize=14)
plt.tight_layout()
plt.show()

# --- G images ---
plt.figure(figsize=(6,6))
for i, fname in enumerate(sample):
    img = np.array(Image.open(os.path.join(y_dir, fname)))
    plt.subplot(3,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(fname)
    plt.axis('off')
plt.suptitle("G Channel Images", fontsize=14)
plt.tight_layout()
plt.show()

# %%
