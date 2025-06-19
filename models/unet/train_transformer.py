# train_unet_h5.py

import os
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import NeptuneLogger
from model import UNetLightning
from datset.h5_dataset import H5UNetDataModule # Adjust import path if needed
from config import Config, CONFIG_FILE_PATH
import warnings

def manual_validation_h5(input_tensor, target_tensor, model, _loss, rest_data):
    """
    Performs manual validation checks specifically for HDF5 data.
    Focuses on loss calculation consistency.

    Args:
        input_tensor: Input tensor from the dataloader.
        target_tensor: Target tensor from the dataloader.
        model: The lightning model.
        _loss: The loss calculated during the training step.
        rest_data: Additional data from the dataloader (should contain HDF5 index tuple).
    """
    import torch
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss # Keep loss functions

    # Determine loss function
    n_classes = model.model.n_classes
    if n_classes == 1:
        loss_fn = BCEWithLogitsLoss()
    else:
        loss_fn = CrossEntropyLoss()

    # Extract HDF5 index for logging (optional)
    hdf5_index = rest_data[0] if rest_data and isinstance(rest_data, (tuple, list)) and len(rest_data) > 0 else 'unknown'

    # Run model prediction again on the same input
    # Ensure model is on correct device if necessary, though Lightning should handle it
    # input_tensor = input_tensor.to(model.device)
    # target_tensor = target_tensor.to(model.device)
    pred = model(input_tensor)

    # Calculate loss again
    loss_check = loss_fn(pred, target_tensor)

    # Assert that the loss calculated here matches the one from the training step
    # Use torch.isclose for floating point comparison
    if not torch.isclose(_loss.detach(), loss_check.detach(), atol=1e-6):
         warnings.warn(f"[Manual Validation HDF5 idx {hdf5_index}] Loss mismatch: {_loss.item():.6f} (train step) != {loss_check.item():.6f} (recalculated)")
    else:
         print(f"[Manual Validation HDF5 idx {hdf5_index}] Loss check OK: {_loss.item():.6f}")

    # No return needed, or return True/False for success/failure if preferred


def main(config: Config, debug: bool = False, manual: bool = False):
    torch.cuda.empty_cache()

    if not config.h5_file_path:
         raise ValueError("Config error: 'h5_file_path' must be set for HDF5 training.")
    print(f"Using HDF5 DataModule with file: {config.h5_file_path}")

    data_module = H5UNetDataModule(
        h5_file_path=config.h5_file_path,
        image_dataset_name=config.h5_image_dset, # Make sure these are in your Config
        mask_dataset_name=config.h5_mask_dset,   # Make sure these are in your Config
        batch_size=1 if manual else config.batch_size,
        transforms=config.transform, # Assumes the same transform object works
        n_workers=8, # Adjust if needed
        split_ratio=0.5, # Or get from config
        seed=config.seed if hasattr(config, 'seed') else 42 # Use seed if available
    )

    # --- Logger setup ---
    logger = NeptuneLogger(
        api_key=os.environ.get("NEPTUNE_API_TOKEN"),
        project="richbai90/unet2", # Consider a different project or use tags clearly
        name="unet-h5-training", # Clearly identify HDF5 runs
        tags=["training", "segmentation", "unet", "32x32", "hdf5"] # Add HDF5 tag
    )
    run_id = logger.experiment["sys/id"].fetch()
    print(f"Neptune Run ID (HDF5): {run_id}")

    # --- Checkpoint callback ---
    checkpoint_cb = ModelCheckpoint(
        monitor='val_dice', # Ensure your model logs this metric
        dirpath=config.out_dir,
        filename=f'unet-h5-{run_id}-{{epoch:02d}}-{{val_dice:.4f}}', # Identify HDF5 checkpoints
        save_top_k=3,
        mode='max', # For dice coefficient
        verbose=True
    )

    # --- SWA callback ---
    swa = StochasticWeightAveraging(swa_lrs=1e-2)

    # --- Model instantiation (ensure config provides n_classes) ---
    model = UNetLightning(
        n_channels=config.input_channels,
        n_classes=len(config.classes), # Or config.n_classes
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
    )

    # --- Trainer arguments ---
    trainer_args = {
        "logger": logger,
        "max_epochs": config.epochs,
        "precision": config.precision,
        "accelerator": config.accelerator,
        "callbacks": [checkpoint_cb, swa],
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 5,
        "limit_train_batches": 0.5, # Adjust as needed
        "limit_val_batches": 0.1,   # Adjust as needed
    }

    if debug:
        trainer_args.update({
            "fast_dev_run": True,
            "limit_train_batches": 0.1,
            "limit_val_batches": 0.1,
        })
        # Disable SWA in fast_dev_run
        if "callbacks" in trainer_args:
            trainer_args["callbacks"] = [cb for cb in trainer_args["callbacks"] if not isinstance(cb, StochasticWeightAveraging)]


    trainer = Trainer(**trainer_args)

    # --- Manual Training Loop (Adapted for HDF5) ---
    if manual:
        print("--- Starting Manual Training Loop (HDF5) ---")
        if not hasattr(model, 'loss_fn'):
             raise AttributeError("Model needs a 'loss_fn' attribute for manual loop.")

        optimizer_dict = model.configure_optimizers()
        optimizer = optimizer_dict['optimizer'] if isinstance(optimizer_dict, dict) else optimizer_dict

        data_module.setup('fit') # Setup datasets
        scaler = torch.cuda.amp.GradScaler(enabled=config.precision == "16-mixed")

        if not data_module.train_dataloader() or not data_module.val_dataloader():
             raise RuntimeError("HDF5 Dataloaders could not be created.")

        for epoch in range(config.epochs):
            print(f"\nEpoch {epoch+1}/{config.epochs}")
            model.train()
            train_loss_accum = 0.0
            # <<< CHANGE >>> Capture 'rest' which contains HDF5 index tuple
            for i, batch in enumerate(data_module.train_dataloader()):
                if len(batch) != 3: raise ValueError("Train Dataloader (H5) format error")
                img, target, rest = batch # rest = (hdf5_index,)

                img = img.to(model.device)
                target = target.to(model.device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=config.precision == "16-mixed"):
                    output = model(img)
                    loss = model.loss_fn(output, target)

                if torch.isnan(loss) or torch.isinf(loss):
                    warnings.warn(f"NaN/Inf loss at train step {i}, epoch {epoch}. Skipping backward.")
                    continue

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss_accum += loss.item()
                if (i + 1) % 10 == 0:
                    print(f"  Train Batch {i+1}, Loss: {loss.item():.4f}")
                    logger.log_metrics({"train_loss_step": loss.item()}, step=trainer.global_step)

                # <<< CHANGE >>> Call the HDF5-specific manual validation
                manual_validation_h5(
                    input_tensor=img, target_tensor=target, model=model,
                    _loss=loss, rest_data=rest
                )

            avg_train_loss = train_loss_accum / (i + 1) if i is not None else 0
            print(f"Epoch {epoch+1} Avg Train Loss: {avg_train_loss:.4f}")
            logger.log_metrics({"train_loss_epoch": avg_train_loss, "epoch": epoch}, step=trainer.global_step)

            # Manual Validation Loop
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for i, batch in enumerate(data_module.val_dataloader()):
                    if len(batch) != 3: raise ValueError("Val Dataloader (H5) format error")
                    img, target, rest = batch
                    img = img.to(model.device)
                    target = target.to(model.device)

                    with torch.cuda.amp.autocast(enabled=config.precision == "16-mixed"):
                        output = model(img)
                        loss = model.loss_fn(output, target)

                    if torch.isnan(loss) or torch.isinf(loss):
                        warnings.warn(f"NaN/Inf loss at val step {i}, epoch {epoch}.")
                        continue

                    val_loss_accum += loss.item()
                    if (i + 1) % 10 == 0:
                        print(f"  Val Batch {i+1}, Loss: {loss.item():.4f}")
                        logger.log_metrics({"val_loss_step": loss.item()}, step=trainer.global_step)

            avg_val_loss = val_loss_accum / (i + 1) if i is not None else 0
            print(f"Epoch {epoch+1} Avg Val Loss: {avg_val_loss:.4f}")
            logger.log_metrics({"val_loss_epoch": avg_val_loss}, step=trainer.global_step)

    # --- Standard Training (using Trainer) ---
    else:
        print("--- Starting Standard HDF5 Training with Pytorch Lightning Trainer ---")
        trainer.fit(model, datamodule=data_module)
        print("--- HDF5 Training Finished ---")

        # Optional Testing Phase
        # data_module.setup('test') # Ensure test setup is called if needed
        # if data_module.test_dataloader():
        #     print("--- Starting HDF5 Testing ---")
        #     trainer.test(model, datamodule=data_module)
        #     print("--- HDF5 Testing Finished ---")
        # else:
        #      print("--- No HDF5 test dataloader configured, skipping testing ---")


if __name__ == '__main__':
    parser = ArgumentParser()
    # Keep debug and manual flags, remove --use-hdf5
    parser.add_argument("--debug", action="store_true", help="Run in fast_dev_run mode.")
    parser.add_argument("--manual", action="store_true", help="Run manual training loop instead of Trainer.fit.")
    args = parser.parse_args()

    # Load configuration (ensure it has HDF5 paths/settings)
    config = Config(CONFIG_FILE_PATH)

    # Call main - it now only uses HDF5 settings from config
    main(config, debug=args.debug, manual=args.manual)
