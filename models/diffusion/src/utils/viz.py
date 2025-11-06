import numpy as np
import torch
from torchvision.utils import make_grid

def to_grid(img: torch.Tensor, nrow: int = 9):
    """img: (N,C,H,W) in [0,1]"""
    grid = make_grid(img.cpu(), nrow=nrow, padding=2)
    grid = grid.permute(1,2,0).numpy()
    grid = np.clip(grid, 0, 1)
    return grid
