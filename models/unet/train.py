import os
import torch


from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import NeptuneLogger
from model import UNetLightning
from dataset import UNetDataModule, InputLoader, TargetLoader
from config import Config, CONFIG_FILE_PATH


def main(config: Config, debug: bool = False, manual: bool = False):

    torch.cuda.empty_cache()

    input_loader = InputLoader(config.data_dir)

    target_loader = TargetLoader(config.mask_dir)

    data_module = UNetDataModule(
        input_loader=input_loader,
        target_loader=target_loader,
        batch_size=1 if manual else config.batch_size,
        transforms=config.transform,
        n_workers=8,
        split_ratio=config.split_ratio,
    )

    logger = NeptuneLogger(
        api_key=os.environ.get("NEPTUNE_API_TOKEN"),
        project="richbai90/unet2",
        tags=["training", "segmentation", "unet", "64x64", "worms"],
    )

    run_id = logger.experiment["sys/id"].fetch()

    checkpoint_cb = ModelCheckpoint(
        monitor="val_dice",
        dirpath=config.out_dir,
        filename=f"unet-{run_id}-{{epoch:02d}}-{{val_dice:.2f}}",
        save_top_k=3,
        mode="max",
        verbose=True,
    )

    swa = StochasticWeightAveraging(swa_lrs=1e-2)

    model = UNetLightning(
        n_channels=config.input_channels,
        n_classes=len(config.classes),
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
    )

    trainer_args = {
        "logger": logger,
        "max_epochs": config.epochs,
        "precision": config.precision,
        "accelerator": config.accelerator,
        "callbacks": [swa],
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 3,
        "limit_train_batches": 0.05,
        "limit_val_batches": 0.05,
        "devices": 1,
    }

    if debug:

        trainer_args.update(
            {
                "fast_dev_run": True,
                "limit_train_batches": 0.1,
                "limit_val_batches": 0.01,
            }
        )

    trainer = Trainer(**trainer_args)

    if manual:

        optimizer = model.configure_optimizers()["optimizer"]

        data_module.setup("fit")

        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(config.epochs):
            model.train()

            for i, (img, target) in enumerate(data_module.train_dataloader()):

                with torch.cuda.amp.autocast():
                    output = model(img)
                    loss = model.loss_fn(output, target)

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()

                logger.log_metrics({"train_loss": loss.item()}, step=i)

            model.eval()

            for i, (img, target) in enumerate(data_module.val_dataloader()):

                with torch.cuda.amp.autocast():
                    output = model(img)
                    loss = model.loss_fn(output, target)

                logger.log_metrics({"val_loss": loss.item()}, step=i)

            logger.log_metrics({"epoch": epoch}, step=epoch)

    else:
        if config.split_ratio > 1:
            k = int(config.split_ratio)
            for i in range(k):
                model = UNetLightning(
                    n_channels=config.input_channels,
                    n_classes=len(config.classes),
                    learning_rate=config.learning_rate,
                    batch_size=config.batch_size,
                )

                trainer = Trainer(**trainer_args)
                data_module.setup_fold(i)
                trainer.fit(model, data_module)
                trainer.test(model, datamodule=data_module)
        else:
            trainer_args['callbacks'].append(checkpoint_cb)
            trainer = Trainer(**trainer_args)
            trainer.fit(model, data_module)
            trainer.test(model, datamodule=data_module)



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    config = Config(CONFIG_FILE_PATH)

    main(config, debug=args.debug, manual=False)
