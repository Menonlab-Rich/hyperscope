"""
A pytorch-lightning module for the UNet model.
Training, validation, and testing steps are copied from the original UNet implementation.
See: https://github.com/milesial/Pytorch-UNet/blob/master/train.py
"""

import pytorch_lightning as pl
import torch.nn.functional as F
from torch import optim, nn
import torch
from unet.pytorch_model import UNet
from base.loss import DiceLoss
from base.metrics import GeneralizedDiceScore
from torchmetrics import MeanMetric
from neptune.types import File
from warnings import warn
from torchmetrics.classification import MulticlassConfusionMatrix


class UNetLightning(pl.LightningModule):
    def __init__(self, n_channels=1, n_classes=3, bilinear=False,
                 learning_rate=1e-5, weight_decay=1e-9, momentum=0.8,
                 amp=False, batch_size=None):
        super(UNetLightning, self).__init__()
        self.batch_size=batch_size
        self.model = UNet(n_channels, n_classes, bilinear)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.amp = amp
        self.n_classes = n_classes
        self.criterion = nn.CrossEntropyLoss() if n_classes > 1 else nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(ignore_index=0)  # 0 is the background class
        self.val_accuracy = GeneralizedDiceScore(
            self.n_classes, include_background=False)
        self.train_loss_metric = MeanMetric()
        self.val_loss_metric = MeanMetric()
        self.val_outputs = []
        self.batch_count = 0
        self.mccm = MulticlassConfusionMatrix(num_classes=self.n_classes, ignore_index=0, normalize='true').to(self.device)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.RMSprop(self.parameters(),
                                  lr=self.learning_rate,
                                  weight_decay=self.weight_decay,
                                  momentum=self.momentum,
                                  foreach=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'max', patience=2)
        return {'optimizer': optimizer, 'lr_scheduler': {"scheduler": scheduler, 'monitor': 'val_dice'}}

    def calc_loss(self, masks_pred, masks_true):
        if self.n_classes == 1:
            masks_pred = masks_pred.squeeze(1)
        criterion_loss = self.criterion(masks_pred, masks_true)
        dloss = self.dice_loss(masks_pred, masks_true)
        return criterion_loss + dloss

    def training_step(self, batch, batch_idx):
        images, true_masks, _ = batch
        masks_pred = self(images)
        loss = self.calc_loss(masks_pred, true_masks)

        # Update and log the training loss
        self.train_loss_metric.update(loss)
        self.log('train_loss', self.train_loss_metric.compute(),
                 on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        images, true_masks, _ = batch
        masks_pred = self(images)
        loss = self.calc_loss(masks_pred, true_masks)
        if torch.isnan(masks_pred).any():
            warn("mask predictions have nan values")
        if torch.isnan(true_masks).any():
            warn("true masks have nan values")
        if torch.isinf(masks_pred).any():
            warn("mask predictions have inf values")
        if torch.isinf(true_masks).any():
            warn("true masks have inf values")
        # Update and log the custom accuracy
        self.val_accuracy.update(masks_pred, true_masks)
        self.val_loss_metric.update(loss)
        self.log(
            'val_loss', self.val_loss_metric.compute(),
            on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log(
            'val_dice', self.val_accuracy.compute(),
            on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return {
            'loss': loss,
            'accuracy': self.val_accuracy,
            'img': images,
            'mask': true_masks,
            'pred': masks_pred,
        }

    def on_train_epoch_start(self) -> None:
        # See https://github.com/Lightning-AI/pytorch-lightning/issues/17245
        if self.current_epoch == self.trainer.max_epochs - 1:
            # Workaround to always save the last epoch until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/4539)
            self.trainer.check_val_every_n_epoch = 1

            # Disable backward pass for SWA until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/17245)
            self.automatic_optimization = False

    def on_validation_batch_end(self, outputs, *args, **kwargs) -> None:
        self.batch_count += 1
        if self.batch_count % 1000 != 0:
            return # only save a few batches
        pred = F.softmax(outputs['pred'], dim=1)
        pred = torch.argmax(pred, dim=1)
        self.val_outputs.append(
            (outputs['img'],
             outputs['mask'],
             pred))

    def on_validation_epoch_end(self, *args, **kwargs):
        from random import sample

        # Reset the metrics after each validation epoch
        self.val_accuracy.reset()
        self.val_loss_metric.reset()

        if len(self.val_outputs) == 0:
            return  # No images to plot. Happens during dry run

        # Separate the validation outputs based on the ground truth mask values
        mask_value_1 = [output for output in self.val_outputs
                        if (output[1] == 1).any()]
        mask_value_2 = [output for output in self.val_outputs
                        if (output[1] == 2).any()]
        # Select 1 random image from each category
        selected_outputs = []
        if mask_value_1:
            selected_outputs.append(sample(mask_value_1, 1)[0])
        if mask_value_2:
            selected_outputs.append(sample(mask_value_2, 1)[0])

        # Plot the selected images
        for img, mask, pred in selected_outputs:
            _img = img[0].unsqueeze(0)
            _mask = mask[0].unsqueeze(0)
            _pred = pred[0].unsqueeze(0)
            self.plot_segmentation_map(_img, _mask, _pred)

        # Reset the outputs
        self.val_outputs = []

    def test_step(self, batch, batch_idx):
        images, true_masks, _ = batch
        masks_pred = self(images)
        if torch.isnan(masks_pred).any():
            warn("mask predictions have nan values")
        if torch.isnan(true_masks).any():
            warn("true masks have nan values")
        if torch.isinf(masks_pred).any():
            warn("mask predictions have inf values")
        if torch.isinf(true_masks).any():
            warn("true masks have inf values")

        # Update and log the custom accuracy
        self.val_accuracy.update(masks_pred, true_masks)
        self._update_confusion_matrix(masks_pred, true_masks)

        return {
            'img': images,
            'mask': true_masks,
            'pred': masks_pred,
            'accuracy': self.val_accuracy
        }

    def on_test_batch_end(self, outputs, batch, batch_idx):
        # Log the incremental batch dice score
        self.log(
            'batch_test_dice', outputs['accuracy'].compute(),
            prog_bar=True)

        # Select a random sample from the batch
        images, true_masks, masks_pred = outputs['img'], outputs['mask'], outputs['pred']
        idx = torch.randint(0, images.size(0), (1,)).item()
        img, mask, pred = images[idx], true_masks[idx], masks_pred[idx]
        
        pred = F.softmax(pred, dim=0)
        pred = torch.argmax(pred, dim=0)
        self.plot_segmentation_map(img, mask, pred)

    def on_test_epoch_end(self):
        # Log the total dice score for the entire test set
        self.log('total_test_dice', self.val_accuracy.compute(), prog_bar=True)
        self.val_accuracy.reset()  # Reset after logging
        plt = self.mccm.plot()
        self.logger.experiment['Confusion Matrix'].log(File.as_image(plt))
        self.mccm.reset()

    def use_checkpointing(self):
        self.model.use_checkpointing()

    def predict_step(self, batch, batch_idx):
        images, true_masks, _ = batch
        masks_pred = self(images)
        masks_pred = F.softmax(masks_pred, dim=1)
        masks_pred = torch.argmax(masks_pred, dim=1)
        return self._classify_tensors(masks_pred).cpu().numpy()
        
    def _update_confusion_matrix(self, pred, target):
        # Squeeze the tensors to remove the channel dimension
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred_classified = self._classify_tensors(pred)
        target_classified = self._classify_tensors(target)
        
        # Update the confusion matrix
        self.mccm.update(pred_classified, target_classified)
        
    def _classify_tensors(self, tensor, patch_size=8):
        tensor = tensor.squeeze(1)
        B, H, W = tensor.shape  # Keep batch dimension

        # Divide the tensor into 8x8 patches
        patches = tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)

        # Flatten the patches to create a list of 8x8 patches
        patches = patches.contiguous().view(B, -1, patch_size, patch_size)

        # Initialize the classified patches tensor
        classified_patches = torch.zeros(B, patches.shape[1], dtype=torch.int64)

        # Classify each patch
        for i in range(patches.shape[1]):
            patch = patches[:, i, :, :]
            count_1 = (patch == 1).sum(dim=[1, 2])
            count_2 = (patch == 2).sum(dim=[1, 2])

            # Determine classification based on the count
            classified_patches[:, i] = torch.where(count_1 > count_2, 1,
                                                torch.where(count_2 > count_1, 2, 0))

        # Flatten the classified patches to be compatible with MulticlassConfusionMatrix
        classified_patches = classified_patches.view(-1)

        return classified_patches
        

    def _mask_to_rgb(self, mask):
        """
        Convert a segmentation mask to an RGB image.

        Args:
            mask: A tensor of shape [H, W] with class indices.

        Returns:
            An RGB image of shape [H, W, 3].
        """
        import numpy as np

        # Create an RGB image with the same height and width as the mask
        mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        # Check the number of classes in the mask
        if mask.max() == 0:
            return mask_rgb
        # Assign colors to each class (you can modify the colors as needed)
        mask_rgb[mask == 1, :] = [255, 0, 0]  # Red for class 1
        mask_rgb[mask == 2, :] = [0, 255, 0]  # Green for class 2

        return mask_rgb

    def plot_segmentation_map(self, image, mask, pred):
        import numpy as np
        import matplotlib.pyplot as plt
        image = image.squeeze().cpu().numpy()
        mask = mask.squeeze().cpu().numpy()
        pred = pred.squeeze().cpu().numpy()
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(image.squeeze(), cmap='gray')
        ax[0].set_title('Image')
        ax[1].imshow(self._mask_to_rgb(mask))
        ax[1].set_title('Ground Truth')
        ax[2].imshow(self._mask_to_rgb(pred))
        ax[2].set_title('Prediction')

        # Hide axes
        for a in ax:
            a.axis('off')
        # Add a label to the figure
        fig.suptitle('Segmentation Map')

        # Add legend to the figure
        legend = [plt.Line2D([0], [0], color='red', lw=4, label='625'),
                  plt.Line2D([0], [0], color='green', lw=4, label='605')]

        fig.legend(handles=legend, loc='center', ncol=2)
        plt.tight_layout()

        self.logger.experiment['Segmentation Map'].log(File.as_image(fig))
        # Save the raw tensors to Neptune for debugging
        self.logger.experiment['Raw Input'].log(File.as_image(image))
        self.logger.experiment['Raw Mask'].log(File.as_image(mask))
        self.logger.experiment['Raw Prediction'].log(File.as_image(pred))
        plt.close()
