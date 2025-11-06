import argparse, os, yaml
from src.data.rbg_dataset import RBGDataset
from src.engine.trainer import Trainer

def load_cfg(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    # Resolve ${data_root}
    def r(p): return p.replace('${data_root}', cfg['data_root'])
    for k in ['raw_dir','x_train','y_train','x_test','y_test','out_dir']:
        cfg[k] = r(cfg[k])
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to YAML config')
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    os.makedirs(cfg['out_dir'], exist_ok=True)

    train_ds = RBGDataset(cfg['x_train'], cfg['y_train'], norm=cfg['norm'])
    trainer = Trainer(cfg, train_ds, cfg['out_dir'])
    trainer.train()

if __name__ == '__main__':
    main()
