import os, torch
import numpy as np
from torch.utils.data import DataLoader
from ..models.build_unet import build_unet_and_scheduler
from ..utils.viz import to_grid
import matplotlib.pyplot as plt

class Sampler:
    def __init__(self, cfg, test_ds, out_dir, ckpt_path):
        self.cfg = cfg
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.sample_dir = os.path.join(out_dir, 'samples')
        os.makedirs(self.sample_dir, exist_ok=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.unet, self.scheduler = build_unet_and_scheduler(cfg['model'], cfg['scheduler'])
        state = torch.load(ckpt_path, map_location='cpu')
        self.unet.load_state_dict(state['unet'])
        self.unet.to(self.device)
        self.unet.eval()

        self.test_ds = test_ds

    @torch.no_grad()
    def sample_batch(self, rb):
        B = rb.size(0)
        g = torch.randn(B, 1, self.cfg['image_size'], self.cfg['image_size'], device=self.device)
        for t in reversed(range(self.scheduler.config.num_train_timesteps)):
            ts = torch.full((B,), t, device=self.device, dtype=torch.long)
            model_in = torch.cat([g, rb.to(self.device)], dim=1)
            noise_pred = self.unet(model_in, ts).sample
            out = self.scheduler.step(model_output=noise_pred, timestep=ts[0], sample=g)
            g = out.prev_sample
        return torch.clamp(g, 0.0, 1.0)

    def run(self, n_images=9, nrow=9):
        idxs = list(range(len(self.test_ds)))[:n_images]
        rbs, gs = [], []
        for i in idxs:
            rb, g, _ = self.test_ds[i]
            rbs.append(rb); gs.append(g)
        rb = torch.stack(rbs).to(self.device)
        g  = torch.stack(gs).to(self.device)

        g_hat = self.sample_batch(rb)

        R = rb[:,0:1]; B = rb[:,1:2]
        pred_rgb = torch.cat([R, g_hat, B], dim=1)
        rb_rgb   = torch.cat([R, torch.zeros_like(R), B], dim=1)
        diff     = (g - g_hat).abs()

        # Save grids
        grids = {
            'pred_rgb.png': to_grid(pred_rgb, nrow=nrow),
            'rb_rgb.png': to_grid(rb_rgb, nrow=nrow),
            'g_actual.png': to_grid(g.repeat(1,3,1,1), nrow=nrow),
            'g_pred.png': to_grid(g_hat.repeat(1,3,1,1), nrow=nrow),
            'g_diff.png': to_grid(diff.repeat(1,3,1,1), nrow=nrow),
        }
        for name, grid in grids.items():
            plt.figure(figsize=(10,10))
            plt.imshow(grid)
            plt.axis('off')
            out = os.path.join(self.sample_dir, name)
            plt.tight_layout()
            plt.savefig(out, bbox_inches='tight')
            plt.close()
            print(f'[sample] saved -> {out}')
