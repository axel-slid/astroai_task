import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class RBGDataset(Dataset):
    """Dataset returning (rb, g, fname)
    rb: (2,H,W) with channels [R,B], g: (1,H,W),
    normalized per config.
    """
    def __init__(self, x_dir, y_dir, norm='zero_one'):
        xs = {f for f in os.listdir(x_dir) if f.lower().endswith('.png')}
        ys = {f for f in os.listdir(y_dir) if f.lower().endswith('.png')}
        self.files = sorted(xs & ys)
        if not self.files:
            raise RuntimeError(f'No matching .png in {x_dir} and {y_dir}')
        self.x_dir, self.y_dir, self.norm = x_dir, y_dir, norm

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        x = np.array(Image.open(os.path.join(self.x_dir, fname)).convert('RGB'), dtype=np.float32) # (H,W,3)
        g = np.array(Image.open(os.path.join(self.y_dir, fname)).convert('L'), dtype=np.float32)    # (H,W)
        R, B = x[:,:,0], x[:,:,2]
        if self.norm == 'zero_one':
            R, B, g = R/255.0, B/255.0, g/255.0
        elif self.norm == 'neg_one_one':
            R, B, g = (R/127.5-1.0), (B/127.5-1.0), (g/127.5-1.0)
        rb = np.stack([R,B], axis=0)
        g = g[None,...]
        return torch.from_numpy(rb), torch.from_numpy(g), fname
