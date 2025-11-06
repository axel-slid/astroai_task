import argparse, os, yaml
from src.data.rbg_dataset import RBGDataset
from src.engine.sampler import Sampler

def load_cfg(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    def r(p): return p.replace('${data_root}', cfg['data_root'])
    for k in ['raw_dir','x_train','y_train','x_test','y_test','out_dir']:
        cfg[k] = r(cfg[k])
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True, help='Path to checkpoint .pt from training')
    ap.add_argument('--n', type=int, default=None, help='How many test images to sample')
    ap.add_argument('--nrow', type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    test_ds = RBGDataset(cfg['x_test'], cfg['y_test'], norm=cfg['norm'])
    sampler = Sampler(cfg, test_ds, cfg['out_dir'], args.ckpt)

    n = args.n if args.n is not None else cfg['sample']['n_images']
    nrow = args.nrow if args.nrow is not None else cfg['sample']['nrow']
    sampler.run(n_images=n, nrow=nrow)

if __name__ == '__main__':
    main()
