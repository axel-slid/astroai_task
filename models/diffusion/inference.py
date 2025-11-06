# %%

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler
from src.data.rbg_dataset import RBGDataset
import yaml

# ============================
# CONFIGURE PATHS
# ============================
ckpt_path = "/home/dils/astroai/task/code/models/diffusion/outputs/checkpoints/last.pt"
x_test_dir = "/home/dils/astroai/task/data/split/x_test"
save_dir = "/home/dils/astroai/task/outputs/diffusion_predicted_y_test"
os.makedirs(save_dir, exist_ok=True)

# load YAML config to ensure consistency
config_path = "/home/dils/astroai/task/code/models/diffusion/configs/rbg.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================
# LOAD DATASET
# ============================
test_set = RBGDataset(x_dir=x_test_dir, y_dir=x_test_dir, norm=cfg["norm"])  # y_dir dummy since we only need x
print(f"Loaded {len(test_set)} test images")

# ============================
# REBUILD MODEL + LOAD CHECKPOINT
# ============================
from src.models.build_unet import build_unet_and_scheduler
unet, scheduler = build_unet_and_scheduler(cfg["model"], cfg["scheduler"])
state = torch.load(ckpt_path, map_location="cpu")
unet.load_state_dict(state["unet"])
unet.to(device)
unet.eval()

# ============================
# SAMPLING FUNCTION
# ============================
@torch.no_grad()
def sample_green_from_rb(rb_batch):
    B = rb_batch.size(0)
    g = torch.randn(B, 1, cfg["image_size"], cfg["image_size"], device=device)
    for t in reversed(range(scheduler.config.num_train_timesteps)):
        ts = torch.full((B,), t, device=device, dtype=torch.long)
        model_in = torch.cat([g, rb_batch.to(device)], dim=1)
        noise_pred = unet(model_in, ts).sample
        out = scheduler.step(model_output=noise_pred, timestep=ts[0], sample=g)
        g = out.prev_sample
    return torch.clamp(g, 0.0, 1.0)

# ============================
# LOOP THROUGH TEST IMAGES
# ============================
batch_size = 32
loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

for rb, _, fnames in tqdm(loader, desc="Predicting G channels"):
    rb = rb.to(device)
    g_hat = sample_green_from_rb(rb)  # (B,1,32,32)

    g_hat = (g_hat * 255.0).byte().cpu().numpy()  # back to 0–255
    for i, fname in enumerate(fnames):
        out_path = os.path.join(save_dir, fname)
        Image.fromarray(g_hat[i, 0]).save(out_path)

print(f"✅ Saved all predicted G-channel images to: {save_dir}")

# %%
