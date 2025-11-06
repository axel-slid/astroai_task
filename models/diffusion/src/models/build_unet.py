from diffusers import UNet2DModel, DDPMScheduler

def build_unet_and_scheduler(cfg_model: dict, cfg_sched: dict):
    unet = UNet2DModel(
        sample_size=cfg_model.get('sample_size', 32),
        in_channels=cfg_model.get('in_channels', 3),
        out_channels=cfg_model.get('out_channels', 1),
        layers_per_block=cfg_model.get('layers_per_block', 2),
        block_out_channels=tuple(cfg_model.get('block_out_channels', [64,128,128])),
        down_block_types=tuple(cfg_model.get('down_block_types', ["DownBlock2D","DownBlock2D","AttnDownBlock2D"])),
        up_block_types=tuple(cfg_model.get('up_block_types', ["AttnUpBlock2D","UpBlock2D","UpBlock2D"])),
    )
    scheduler = DDPMScheduler(
        num_train_timesteps=cfg_sched.get('num_train_timesteps', 1000),
        beta_start=cfg_sched.get('beta_start', 1e-4),
        beta_end=cfg_sched.get('beta_end', 2e-2),
    )
    return unet, scheduler
