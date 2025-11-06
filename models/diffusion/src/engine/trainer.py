import os, math
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..utils.seed import set_seed
from ..models.build_unet import build_unet_and_scheduler
from ..utils.viz import to_grid
import matplotlib.pyplot as plt

@dataclass
class TrainConfig:
    epochs: int
    lr: float
    weight_decay: float
    log_every: int
    save_every_epochs: int

class Trainer:
    def __init__(self, cfg, train_ds, out_dir):
        self.cfg = cfg
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.check_dir = os.path.join(out_dir, 'checkpoints')
        self.sample_dir = os.path.join(out_dir, 'samples')
        os.makedirs(self.check_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.unet, self.scheduler = build_unet_and_scheduler(cfg['model'], cfg['scheduler'])
        self.unet.to(self.device)

        self.opt = torch.optim.AdamW(self.unet.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
        self.train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                                       num_workers=cfg['num_workers'], pin_memory=cfg.get('pin_memory', True))

    def train(self):
        set_seed(42)
        global_step = 0
        for epoch in range(1, self.cfg['train']['epochs']+1):
            running = 0.0
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch:02d}')
            for rb, g, _ in pbar:
                rb = rb.to(self.device)  # (B,2,32,32)
                g  = g.to(self.device)   # (B,1,32,32)

                noise = torch.randn_like(g)
                timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (g.shape[0],), device=self.device).long()
                g_noisy = self.scheduler.add_noise(g, noise, timesteps)

                model_in = torch.cat([g_noisy, rb], dim=1)  # (B,3,32,32)
                pred = self.unet(model_in, timesteps).sample  # (B,1,32,32)

                loss = F.mse_loss(pred, noise)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                running += loss.item() * g.size(0)
                global_step += 1
                if global_step % self.cfg['train']['log_every'] == 0:
                    pbar.set_postfix({'loss': f'{loss.item():.6f}'})

            epoch_loss = running / len(self.train_loader.dataset)
            print(f'Epoch {epoch:02d} | Train MSE(noise): {epoch_loss:.6f}')

            if epoch % self.cfg['train']['save_every_epochs'] == 0:
                self.save_checkpoint(os.path.join(self.check_dir, 'last.pt'))

    def save_checkpoint(self, path):
        torch.save({'unet': self.unet.state_dict()}, path)
        print(f'[ckpt] saved -> {path}')
