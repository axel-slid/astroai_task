# %%

import os, random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


RAW_DIR   = "/home/dils/astroai/task/data/raw"
X_TEST    = "/home/dils/astroai/task/data/split/x_test"
Y_TEST    = "/home/dils/astroai/task/data/split/y_test"

DIFF_PRED = "/home/dils/astroai/task/outputs/diffusion_predicted_y_test"      # predicted G (grayscale)
POLY_PRED = "/home/dils/astroai/task/outputs/polynomial_reg_predicted_y_test" # predicted G (grayscale)

OUT_FIG   = "/home/dils/astroai/task/code/experiments/compare_diffusion_vs_poly.png"
N_SHOW    = 9  # number of images to visualize

# ==== helpers ====
def imread_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

def imread_gray(path):
    return np.array(Image.open(path).convert("L"))

def safe32x32(arr, mode="RGB"):
    if mode == "RGB":
        if arr.shape != (32,32,3):
            arr = np.array(Image.fromarray(arr).resize((32,32))).astype(np.uint8)
            if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
        return arr
    else:
        if arr.shape != (32,32):
            arr = np.array(Image.fromarray(arr).resize((32,32)))
        return arr

def list_pngs(d): return {f for f in os.listdir(d) if f.lower().endswith(".png")}
common = list(
    list_pngs(DIFF_PRED)
    & list_pngs(POLY_PRED)
    & list_pngs(X_TEST)
    & list_pngs(Y_TEST)
)
if not common:
    raise RuntimeError("No overlapping PNG filenames found across prediction/test folders.")

random.shuffle(common)
show = common[:min(N_SHOW, len(common))]

orig_list, actG_list = [], []
diffG_list, polyG_list = [], []
diffRGB_list, polyRGB_list = [], []

for fname in show:
    raw_path = os.path.join(RAW_DIR, fname)
    if os.path.isfile(raw_path):
        orig = imread_rgb(raw_path)
    else:
        # reconstruct original from test RB + true G
        x = imread_rgb(os.path.join(X_TEST, fname))  # RB stored as RGB (G is 0)
        R, B = x[:,:,0], x[:,:,2]
        Gtrue = imread_gray(os.path.join(Y_TEST, fname))
        orig = np.stack([R, Gtrue, B], axis=-1)
    orig = safe32x32(orig, "RGB").astype(np.uint8)

    # true G
    G = safe32x32(imread_gray(os.path.join(Y_TEST, fname)), "L").astype(np.uint8)

    G_diff = safe32x32(imread_gray(os.path.join(DIFF_PRED, fname)), "L").astype(np.uint8)
    G_poly = safe32x32(imread_gray(os.path.join(POLY_PRED, fname)), "L").astype(np.uint8)

    x = imread_rgb(os.path.join(X_TEST, fname))
    R, B = x[:,:,0].astype(np.uint8), x[:,:,2].astype(np.uint8)
    predRGB_diff = np.stack([R, G_diff, B], axis=-1)
    predRGB_poly = np.stack([R, G_poly, B], axis=-1)

    orig_list.append(orig)
    actG_list.append(G)
    diffG_list.append(G_diff)
    polyG_list.append(G_poly)
    diffRGB_list.append(predRGB_diff)
    polyRGB_list.append(predRGB_poly)

diff_abs = [np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
            for a, b in zip(actG_list, diffG_list)]
poly_abs = [np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
            for a, b in zip(actG_list, polyG_list)]

# ==== plot 8 rows × N columns ====
N = len(show)
plt.figure(figsize=(2.2 * N, 15))

for i in range(N):
    ax = plt.subplot(8, N, 0*N + i + 1); ax.imshow(orig_list[i]);            ax.set_title(f"Original\n{show[i]}"); ax.axis('off')
    ax = plt.subplot(8, N, 1*N + i + 1); ax.imshow(actG_list[i], cmap='gray', vmin=0, vmax=255); ax.set_title("Actual G"); ax.axis('off')
    ax = plt.subplot(8, N, 2*N + i + 1); ax.imshow(diffG_list[i], cmap='gray', vmin=0, vmax=255); ax.set_title("Diffusion Ĝ"); ax.axis('off')
    ax = plt.subplot(8, N, 3*N + i + 1); ax.imshow(polyG_list[i], cmap='gray', vmin=0, vmax=255); ax.set_title("Polynomial Ĝ"); ax.axis('off')
    ax = plt.subplot(8, N, 4*N + i + 1); ax.imshow(diff_abs[i], cmap='gray', vmin=0, vmax=255);  ax.set_title("|G − Ĝ| (diff)"); ax.axis('off')
    ax = plt.subplot(8, N, 5*N + i + 1); ax.imshow(poly_abs[i], cmap='gray', vmin=0, vmax=255);  ax.set_title("|G − Ĝ| (poly)"); ax.axis('off')
    ax = plt.subplot(8, N, 6*N + i + 1); ax.imshow(diffRGB_list[i]);         ax.set_title("Pred RGB (diff)"); ax.axis('off')
    ax = plt.subplot(8, N, 7*N + i + 1); ax.imshow(polyRGB_list[i]);         ax.set_title("Pred RGB (poly)"); ax.axis('off')

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=200)
plt.show()

print(f"Saved comparison figure → {OUT_FIG}")

# %%
