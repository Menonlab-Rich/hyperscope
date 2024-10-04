from typing import Tuple
from pytorch_lightning import LightningModule
from pytorch_lightning import LightningDataModule
from config import Config, CONFIG_FILE_PATH

def load_model(ckpt_path: str, config: Config) -> Tuple[LightningModule, LightningDataModule]:
    from model import UNetLightning
    from dataset import UNetDataModule, InputLoader, TargetLoader
    prefix = config.get('prefix', None)
    input_loader = InputLoader(config.data_dir)
    target_loader = TargetLoader(config.mask_dir)
    model = UNetLightning.load_from_checkpoint(ckpt_path, strict=False)
    data_module = UNetDataModule.load_from_checkpoint(ckpt_path, test_loaders='validation', n_workers=4, prefix=prefix)
    #data_module = UNetDataModule(
    #    input_loader=input_loader,
    #    target_loader=target_loader,
    #    batch_size=config.batch_size,
    #    transforms=config.transform,
    #    test_loaders=(input_loader, target_loader),
    #    n_workers=4
    #)
    return model, data_module

def main(config: Config):
    import os
    from pytorch_lightning.loggers import NeptuneLogger
    from pytorch_lightning import Trainer
    
    logger = NeptuneLogger(
        api_key=os.environ.get("NEPTUNE_API_TOKEN"),
        project="richbai90/UNet2",
        tags=["testing", "unet", "hires", prefix or 'all'],
    )
    
    trainer_args = {
        "logger": logger,
        "max_epochs": config.epochs,
        "precision": config.precision,
        "accelerator": config.accelerator,
        "accumulate_grad_batches": 10,
    }
    
    model, data_module = load_model(config.ckpt_path, config)
    
    trainer = Trainer(**trainer_args)
    trainer.test(model, datamodule=data_module)
    
if __name__ == '__main__':
    cfg = Config(CONFIG_FILE_PATH)
    for prefix in ['605', '625', 'composite', None]:
        cfg.set('prefix', prefix)
        main(cfg)
