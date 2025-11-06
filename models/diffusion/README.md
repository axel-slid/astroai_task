# astroai-rbg
Conditional diffusion (RB → G) on 32×32 images.

## Setup
```bash
# (optional) create env, then:
pip install -r requirements.txt
```

## Train
```bash
python train.py --config configs/rbg.yaml
```
Checkpoints/samples are saved under `outputs/`.

## Sample from a checkpoint
```bash
python sample.py --config configs/rbg.yaml --ckpt outputs/checkpoints/last.pt --n 9
```
Outputs go to `outputs/samples/`.

## What it does
- Trains a UNet with diffusion to denoise **G** given fixed **R,B**.
- Input = concat([noisy_G, R, B]) → predict noise on G.
