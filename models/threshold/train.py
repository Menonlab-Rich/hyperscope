import os
import sys
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
from config import Config
from dataset import NPYDataModule
from lightning_model import Threshold
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (ModelCheckpoint,
                                         StochasticWeightAveraging)
from pytorch_lightning.loggers import NeptuneLogger


class TestOnEpochEnd(Callback):
    """
    Custom callback to run a test loop at the end of each training epoch.
    """
    def __init__(self, every_n) -> None:
        super().__init__()
        self.every_n = every_n
        self.last_test_epoch = 0
    def on_train_epoch_end(self, trainer, pl_module):
        """
        This hook is called after the training epoch and validation loops are finished.
        """
        if trainer.is_global_zero: # Ensure this only runs on the main process
            print(f"\n--- Running test set at the end of epoch {trainer.current_epoch} ---")
            # The trainer object has access to the model and datamodule
            if trainer.current_epoch == self.last_test_epoch + self.every_n:
                self.last_test_epoch = trainer.current_epoch
                trainer.test(model=pl_module, datamodule=trainer.datamodule)

if __name__ == "__main__":
    # 1. Load configuration using the new custom Config class
    base_cfg = OmegaConf.load("config.yml")
    if len(sys.argv) > 1:
        cli_cfg = OmegaConf.from_cli()
        base_cfg = OmegaConf.merge(base_cfg, cli_cfg)
    
    cfg = Config(base_cfg)

    # 2. Set up Neptune Logger
    logger = NeptuneLogger(
        api_key=os.environ.get("NEPTUNE_API_TOKEN"),
        project=cfg.neptune.project,
        tags=list(cfg.neptune.tags), # Convert ListConfig to list
    )
    run_id = logger.experiment["sys/id"].fetch()

    # 3. Prepare callbacks
    callbacks = []

    # Setup Model Checkpointing using the class method
    if "checkpoint.enable" in cfg and cfg.checkpoint.enable:
        checkpoint_mappings = {
            "monitor": "checkpoint.monitor",
            "dirpath": "paths.output",
            "filename": "checkpoint.filename",
            "save_top_k": "checkpoint.save_top_k",
            "mode": "checkpoint.mode",
        }
        checkpoint_args = cfg.select_as_kwargs(checkpoint_mappings)
        checkpoint_args["filename"] = f"{checkpoint_args.get('filename', 'model')}-{run_id}"
        callbacks.append(ModelCheckpoint(**checkpoint_args))

    # Setup SWA using the class method
    if "training.swa.enable" in cfg and cfg.training.swa.enable:
        swa_mappings = {"swa_lrs": "training.swa.lrs", "swa_epoch_start": "training.swa.epoch_start"}
        swa_args = cfg.select_as_kwargs(swa_mappings)
        callbacks.append(StochasticWeightAveraging(**swa_args))

    if "training.test_after_epoch" in cfg and cfg.training.test_after_epoch.enable:
        test_after_epoch_mappings = {
                "every_n": "training.test_after_epoch.every_n"
        }

        test_after_epoch_args = cfg.select_as_kwargs(test_after_epoch_mappings)
        callbacks.append(TestOnEpochEnd(**test_after_epoch_args))

    # 4. Prepare Trainer arguments using the class method
    trainer_mappings = {
        "max_epochs": "training.epochs",
        "precision": "training.precision",
        "accelerator": "training.accelerator",
        "devices": "training.n_devices",
        "gradient_clip_val": "training.gradient_clip_val",
        "accumulate_grad_batches": "training.grad_batches",
        "limit_val_batches": "training.limit_val_batches",
        "limit_train_batches": "training.limit_train_batches",
        "log_every_n_steps": "training.log_every_n_steps",
    }
    trainer_args = cfg.select_as_kwargs(trainer_mappings)
    trainer_args["logger"] = logger
    trainer_args["callbacks"] = callbacks
    trainer_args["enable_checkpointing"] = "checkpoint.enable" in cfg and cfg.checkpoint.enable

    # 5. Instantiate DataModule
    dm = NPYDataModule(
        data_dir=cfg.data_module.data_dir,
        batch_size=cfg.data_module.batch_size,
        num_workers=cfg.data_module.num_workers,
        image_size=cfg.data_module.image_size,
        processor_model_name=cfg.data_module.processor_model_name,
        train_val_test_split=tuple(cfg.data_module.train_val_test_split),
        seed=cfg.data_module.seed,
    )

    # 6. Instantiate LightningModule using the class method
    model_mappings = {
        "lambda_entropy": "model.loss_params.lambda_entropy",
        "lambda_contrastive": "model.loss_params.lambda_contrastive",
        "learning_rate": "model.learning_rate",
        "n_classes": "model.loss_params.n_classes",
        "temp": "model.loss_params.temp",
        "out_shape": "data_module.image_size",
        "optimizer_configs": "model.loss_params.optimizer",
        "scheduler_configs": "model.loss_params.scheduler"
    }

    model_args = cfg.select_as_kwargs(model_mappings)
    lightning_model = Threshold(**model_args)

    # 7. Instantiate and run the Trainer
    trainer = pl.Trainer(**trainer_args)
    print(f"\n--- Starting Training for {trainer_args.get('max_epochs', 'N/A')} epochs ---")
    trainer.fit(lightning_model, dm)
    print("\n--- Training Complete ---")
