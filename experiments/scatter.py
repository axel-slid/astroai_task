# %%

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from tqdm import tqdm
import random

# ---------- paths ----------
RAW_DIR     = "/home/dils/astroai/task/data/raw"  # not used here, just for reference
Y_TEST_DIR  = "/home/dils/astroai/task/data/split/y_test"

DIFF_DIR    = "/home/dils/astroai/task/outputs/diffusion_predicted_y_test"
POLY_DIR    = "/home/dils/astroai/task/outputs/polynomial_reg_predicted_y_test"

OUT_FIG     = "/home/dils/astroai/task/code/experiments/scatter_pred_vs_true.png"

# ---------- config ----------
# To keep scatter responsive, sample up to this many pixels (per model; both use the same mask)
MAX_POINTS = 200_000
RNG_SEED   = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# ---------- helpers ----------
def list_pngs(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])

def read_gray(path):
    # returns HxW uint8
    return np.array(Image.open(path).convert("L"))

# ---------- gather filenames: ONLY what diffusion has produced ----------
diff_files = set(list_pngs(DIFF_DIR))
if not diff_files:
    raise RuntimeError("No PNGs found in diffusion prediction folder.")

# We also need those files to exist in y_test and polynomial predictions
poly_files = set(list_pngs(POLY_DIR))
y_files    = set(list_pngs(Y_TEST_DIR))
files = sorted(diff_files & poly_files & y_files)
if not files:
    raise RuntimeError("No overlapping filenames across diffusion, polynomial and y_test.")

print(f"Using {len(files)} images (intersection of diffusion/poly/y_test).")

# ---------- load and concatenate pixels ----------
true_list = []
pred_diff_list = []
pred_poly_list = []

for fn in tqdm(files, desc="Loading images"):
    y_true  = read_gray(os.path.join(Y_TEST_DIR, fn)).astype(np.uint8)
    g_diff  = read_gray(os.path.join(DIFF_DIR, fn)).astype(np.uint8)
    g_poly  = read_gray(os.path.join(POLY_DIR, fn)).astype(np.uint8)

    # safety: enforce 32x32
    if y_true.shape != (32, 32):
        y_true = np.array(Image.fromarray(y_true).resize((32, 32)))
    if g_diff.shape != (32, 32):
        g_diff = np.array(Image.fromarray(g_diff).resize((32, 32)))
    if g_poly.shape != (32, 32):
        g_poly = np.array(Image.fromarray(g_poly).resize((32, 32)))

    true_list.append(y_true.reshape(-1))
    pred_diff_list.append(g_diff.reshape(-1))
    pred_poly_list.append(g_poly.reshape(-1))

Y_true   = np.concatenate(true_list).astype(np.float32)   # (N,)
X_diff   = np.concatenate(pred_diff_list).astype(np.float32)
X_poly   = np.concatenate(pred_poly_list).astype(np.float32)

# ---------- (optional) subsample for plotting ----------
N = Y_true.size
if N > MAX_POINTS:
    idx = np.random.choice(N, size=MAX_POINTS, replace=False)
    Y_plot = Y_true[idx]
    Xd_plot = X_diff[idx]
    Xp_plot = X_poly[idx]
else:
    Y_plot = Y_true
    Xd_plot = X_diff
    Xp_plot = X_poly

# ---------- metrics (on ALL points that diffusion produced, not just the plotted subset) ----------
mse_diff = float(np.mean((Y_true - X_diff) ** 2))
mse_poly = float(np.mean((Y_true - X_poly) ** 2))

# ---------- plotting ----------
plt.figure(figsize=(14, 6))

common_kwargs = dict(s=2, alpha=0.25, edgecolors='none')

# left: diffusion
ax1 = plt.subplot(1, 2, 1)
ax1.scatter(Xd_plot, Y_plot, **common_kwargs)
ax1.plot([0,255],[0,255], linestyle='--')  # y=x
ax1.set_xlim(0,255); ax1.set_ylim(0,255)
ax1.set_aspect('equal', adjustable='box')
ax1.set_title("Diffusion: Predicted vs True (G channel)")
ax1.set_xlabel("Predicted pixel (Diffusion)")
ax1.set_ylabel("True pixel (G)")
txt1 = AnchoredText(f"MSE = {mse_diff:.2f}\nN pixels = {Y_true.size:,}", loc="lower right",
                    prop=dict(size=10), frameon=True, bbox_to_anchor=(1,0), bbox_transform=ax1.transAxes)
ax1.add_artist(txt1)
ax1.grid(True, linestyle=':', linewidth=0.5)

# right: polynomial
ax2 = plt.subplot(1, 2, 2)
ax2.scatter(Xp_plot, Y_plot, **common_kwargs)
ax2.plot([0,255],[0,255], linestyle='--')
ax2.set_xlim(0,255); ax2.set_ylim(0,255)
ax2.set_aspect('equal', adjustable='box')
ax2.set_title("Polynomial: Predicted vs True (G channel)")
ax2.set_xlabel("Predicted pixel (Polynomial)")
ax2.set_ylabel("True pixel (G)")
txt2 = AnchoredText(f"MSE = {mse_poly:.2f}\nN pixels = {Y_true.size:,}", loc="lower right",
                    prop=dict(size=10), frameon=True, bbox_to_anchor=(1,0), bbox_transform=ax2.transAxes)
ax2.add_artist(txt2)
ax2.grid(True, linestyle=':', linewidth=0.5)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
plt.savefig(OUT_FIG, dpi=200)
plt.show()

print(f"Saved scatter comparison → {OUT_FIG}")
