# %%

import os
import numpy as np
from PIL import Image
from joblib import load


MODEL_PATH = "/home/dils/astroai/task/code/models/polynomial_reg/artifacts/rb2g_deg2.joblib"
X_TEST_DIR = "/home/dils/astroai/task/data/split/x_test"
OUTPUT_DIR = "/home/dils/astroai/task/outputs/polynomial_reg_predicted_y_test"

os.makedirs(OUTPUT_DIR, exist_ok=True)

artifact = load(MODEL_PATH)
scaler, poly, reg = artifact["scaler"], artifact["poly"], artifact["reg"]
print(f"✅ Loaded model: {MODEL_PATH}")

def load_rb(fname, x_dir):
    path = os.path.join(x_dir, fname)
    with Image.open(path) as im:
        arr = np.array(im.convert("RGB"))
    R = arr[:, :, 0].astype(np.float32)
    B = arr[:, :, 2].astype(np.float32)
    return R, B

files = [f for f in os.listdir(X_TEST_DIR) if f.lower().endswith(".png")]
if not files:
    raise RuntimeError(f"No .png files found in {X_TEST_DIR}")

for i, fname in enumerate(sorted(files)):
    R, B = load_rb(fname, X_TEST_DIR)
    X = np.stack([R.reshape(-1), B.reshape(-1)], axis=1)
    Xp = poly.transform(scaler.transform(X))
    G_hat = reg.predict(Xp).reshape(32, 32)
    G_u8 = np.clip(G_hat, 0, 255).astype(np.uint8)

    out_path = os.path.join(OUTPUT_DIR, fname)
    Image.fromarray(G_u8, mode="L").save(out_path)
    if (i + 1) % 100 == 0 or i == len(files) - 1:
        print(f"Saved {i+1}/{len(files)} predicted images")


# %%
