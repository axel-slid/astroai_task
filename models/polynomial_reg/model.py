# %%

import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import SGDRegressor

# ---------------------------
# Paths
# ---------------------------
ROOT = "/home/dils/astroai/task/data"
RAW_DIR    = os.path.join(ROOT, "raw")

X_TRAIN    = os.path.join(ROOT, "split", "x_train")  # RB as RGB (G=0)
Y_TRAIN    = os.path.join(ROOT, "split", "y_train")  # G grayscale
X_TEST     = os.path.join(ROOT, "split", "x_test")
Y_TEST     = os.path.join(ROOT, "split", "y_test")

for p in [RAW_DIR, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST]:
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Missing directory: {p}")

# ---------------------------
# Settings
# ---------------------------
DEGREE = 2            # polynomial degree for f(R,B)->G
BATCH_IMAGES = 200    # images per batch for streaming fit
SHOW_N = 9            # number of test images to visualize
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------------------------
# Utilities
# ---------------------------
def common_pngs(dir_a, dir_b):
    a = {f for f in os.listdir(dir_a) if f.lower().endswith(".png")}
    b = {f for f in os.listdir(dir_b) if f.lower().endswith(".png")}
    return sorted(a & b)

def load_rb(fname, x_dir):
    """Load RB from x_* folder (RGB with G=0). Returns R, B as float32 (32x32)."""
    path = os.path.join(x_dir, fname)
    with Image.open(path) as im:
        arr = np.array(im)  # (32,32,3)
    if arr.shape != (32, 32, 3):
        raise ValueError(f"{path} shape {arr.shape} != (32,32,3)")
    R = arr[:, :, 0].astype(np.float32)
    B = arr[:, :, 2].astype(np.float32)
    return R, B

def load_g(fname, y_dir):
    """Load G channel from y_* folder (grayscale). Returns G as float32 (32x32)."""
    path = os.path.join(y_dir, fname)
    with Image.open(path) as im:
        G = np.array(im.convert("L"))
    if G.shape != (32, 32):
        raise ValueError(f"{path} shape {G.shape} != (32,32)")
    return G.astype(np.float32)

def load_original_rgb(fname):
    """Load original RGB for visualization (from raw). If missing, reconstruct from test split."""
    raw_path = os.path.join(RAW_DIR, fname)
    if os.path.isfile(raw_path):
        with Image.open(raw_path) as im:
            arr = np.array(im.convert("RGB"))
        if arr.shape == (32, 32, 3):
            return arr
    # Fallback (only if raw missing for this fname) — reconstruct from test split
    R, B = load_rb(fname, X_TEST)
    G = load_g(fname, Y_TEST)
    return np.stack([R, G, B], axis=-1).astype(np.uint8)

# ---------------------------
# Discover files
# ---------------------------
train_files = common_pngs(X_TRAIN, Y_TRAIN)
test_files  = common_pngs(X_TEST, Y_TEST)

if not train_files:
    raise RuntimeError("No paired PNGs in x_train & y_train.")
if not test_files:
    raise RuntimeError("No paired PNGs in x_test & y_test.")

print(f"Train pairs: {len(train_files)}   |   Test pairs: {len(test_files)}")

# ---------------------------
# Model pieces
# ---------------------------
poly   = PolynomialFeatures(degree=DEGREE, include_bias=True)
scaler = StandardScaler(with_mean=True, with_std=True)
reg    = SGDRegressor(
    loss="squared_error", penalty="l2", alpha=1e-6,
    learning_rate="invscaling", eta0=1e-2, max_iter=1,
    tol=None, random_state=RANDOM_SEED
)

# ---------------------------
# Pass 1: Fit scaler on ALL train pixels (streaming)
# ---------------------------
print("Pass 1/2: Fitting scaler on ALL train pixels (R,B)…")
for start in range(0, len(train_files), BATCH_IMAGES):
    batch = train_files[start:start+BATCH_IMAGES]
    R_list, B_list = [], []
    for fname in batch:
        R, B = load_rb(fname, X_TRAIN)
        R_list.append(R.reshape(-1))
        B_list.append(B.reshape(-1))
    RB = np.stack([np.concatenate(R_list), np.concatenate(B_list)], axis=1)  # (N,2)
    scaler.partial_fit(RB)
print("Scaler fitted.")

# ---------------------------
# Pass 2: Train on ALL train pixels (streaming)
# ---------------------------
print("Pass 2/2: Training polynomial SGD on ALL train pixels…")
poly_fitted = False
for start in range(0, len(train_files), BATCH_IMAGES):
    batch = train_files[start:start+BATCH_IMAGES]
    R_list, B_list, G_list = [], [], []
    for fname in batch:
        R, B = load_rb(fname, X_TRAIN)
        G    = load_g(fname, Y_TRAIN)
        R_list.append(R.reshape(-1))
        B_list.append(B.reshape(-1))
        G_list.append(G.reshape(-1))
    R_all = np.concatenate(R_list)
    B_all = np.concatenate(B_list)
    G_all = np.concatenate(G_list)

    X = np.stack([R_all, B_all], axis=1)           # (N,2)
    Xs = scaler.transform(X)                       # standardized (R,B)
    if not poly_fitted:
        Xp = poly.fit_transform(Xs)                # fit once to lock feature mapping
        poly_fitted = True
    else:
        Xp = poly.transform(Xs)

    reg.partial_fit(Xp, G_all)
    print(f"  Trained on batch {(start//BATCH_IMAGES)+1}/{(len(train_files)+BATCH_IMAGES-1)//BATCH_IMAGES} "
          f"with {len(G_all):,} pixels")

print("✅ Training complete.")

# ---------------------------
# Evaluate on the FULL test set (all pixels)
# ---------------------------
print("Evaluating on ALL test pixels…")
mae_sum = 0.0
mse_sum = 0.0
pix_count = 0

for start in range(0, len(test_files), BATCH_IMAGES):
    batch = test_files[start:start+BATCH_IMAGES]
    R_list, B_list, G_list = [], [], []
    for fname in batch:
        R, B = load_rb(fname, X_TEST)
        G    = load_g(fname, Y_TEST)
        R_list.append(R.reshape(-1))
        B_list.append(B.reshape(-1))
        G_list.append(G.reshape(-1))

    R_all = np.concatenate(R_list)
    B_all = np.concatenate(B_list)
    G_all = np.concatenate(G_list)

    X = np.stack([R_all, B_all], axis=1)
    Xp = poly.transform(scaler.transform(X))
    G_hat = reg.predict(Xp)

    diff = (G_all - G_hat).astype(np.float64)
    mae_sum += np.abs(diff).sum()
    mse_sum += (diff**2).sum()
    pix_count += len(G_all)

mae = mae_sum / pix_count
mse = mse_sum / pix_count
print(f"Test MAE (all pixels): {mae:.2f}")
print(f"Test MSE (all pixels): {mse:.2f}")

# ---------------------------
# Visualization on 9 test images
# ---------------------------
show_files = random.sample(test_files, min(SHOW_N, len(test_files)))

orig_list, actG_list, predG_list, diff_list, predRGB_list, rb_list = [], [], [], [], [], []

for fname in show_files:
    R, B = load_rb(fname, X_TEST)
    G    = load_g(fname, Y_TEST)

    X = np.stack([R.reshape(-1), B.reshape(-1)], axis=1)
    Xp = poly.transform(scaler.transform(X))
    G_hat = reg.predict(Xp).reshape(32, 32)
    G_hat_u8 = np.clip(G_hat, 0, 255).astype(np.uint8)

    orig = load_original_rgb(fname).astype(np.uint8)
    rb   = np.stack([R, np.zeros_like(R), B], axis=-1).astype(np.uint8)
    pred_rgb = np.stack([R, G_hat_u8, B], axis=-1).astype(np.uint8)
    diff = np.abs(G.astype(np.int16) - G_hat_u8.astype(np.int16)).astype(np.uint8)

    orig_list.append(orig)
    actG_list.append(G.astype(np.uint8))
    predG_list.append(G_hat_u8)
    diff_list.append(diff)
    predRGB_list.append(pred_rgb)
    rb_list.append(rb)

N = len(show_files)
plt.figure(figsize=(2.2 * N, 12))
for i in range(N):
    ax = plt.subplot(6, N, 0*N + i + 1); ax.imshow(orig_list[i]);            ax.set_title(f"Original\n{show_files[i]}"); ax.axis('off')
    ax = plt.subplot(6, N, 1*N + i + 1); ax.imshow(actG_list[i],  cmap='gray', vmin=0, vmax=255); ax.set_title("Actual G");    ax.axis('off')
    ax = plt.subplot(6, N, 2*N + i + 1); ax.imshow(predG_list[i], cmap='gray', vmin=0, vmax=255); ax.set_title("Predicted G"); ax.axis('off')
    ax = plt.subplot(6, N, 3*N + i + 1); ax.imshow(diff_list[i],  cmap='gray', vmin=0, vmax=255); ax.set_title("|G - Ĝ|");    ax.axis('off')
    ax = plt.subplot(6, N, 4*N + i + 1); ax.imshow(predRGB_list[i]);         ax.set_title("Predicted RGB");                    ax.axis('off')
    ax = plt.subplot(6, N, 5*N + i + 1); ax.imshow(rb_list[i]);              ax.set_title("RB (G=0)");                         ax.axis('off')

plt.tight_layout()
plt.show()

# %%





# save "weights"

from joblib import dump
import time

ARTIFACT_DIR = os.path.join(ROOT, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

artifact = {
    "scaler": scaler,   # fitted StandardScaler
    "poly": poly,       # fitted PolynomialFeatures
    "reg": reg,         # fitted SGDRegressor
    "meta": {
        "degree": DEGREE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": RANDOM_SEED,
        "image_size": (32, 32),
        "channels_in": ["R","B"],
        "channel_out": "G",
    },
}

model_path = os.path.join(ARTIFACT_DIR, f"rb2g_deg{DEGREE}.joblib")
dump(artifact, model_path)
# %%
