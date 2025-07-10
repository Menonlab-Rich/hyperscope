import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchmetrics.functional.segmentation import generalized_dice_score
from typing import *

"""Common image segmentation losses.
"""

import torch
from torch.nn import functional as F


def bce_loss(true, logits, pos_weight=None):
    """Computes the weighted binary cross-entropy loss.

    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, 1, H, W]. Corresponds to
            the raw output or logits of the model.
        pos_weight: a scalar representing the weight attributed
            to the positive class. This is especially useful for
            an imbalanced dataset.

    Returns:
        bce_loss: the weighted binary cross-entropy loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(
        logits.float(),
        true.float(),
        pos_weight=pos_weight,
    )
    return bce_loss


def ce_loss(true, logits, weights, ignore=255):
    """Computes the weighted multi-class cross-entropy loss.

    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        weight: a tensor of shape [C,]. The weights attributed
            to each class.
        ignore: the class index to ignore.

    Returns:
        ce_loss: the weighted multi-class cross-entropy loss.
    """
    ce_loss = F.cross_entropy(
        logits.float(),
        true.long(),
        ignore_index=ignore,
        weight=weights,
    )
    return ce_loss


def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.

    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.

    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return 1 - jacc_loss


def tversky_loss(true, logits, alpha, beta, eps=1e-7):
    """Computes the Tversky loss [1].

    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.

    Returns:
        tversky_loss: the Tversky loss.

    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff

    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()
    return 1 - tversky_loss


def ce_dice(true, pred, log=False, w1=1, w2=1):
    pass


def ce_jaccard(true, pred, log=False, w1=1, w2=1):
    pass


def focal_loss(true, pred):
    pass


class JML1(nn.Module):
    """
    A soft Jaccard loss that uses the L1 norm to calculate the error
    Based on the paper Jaccard Metric Losses: Optimizing the Jaccard Index with Soft Labels
    https://arxiv.org/pdf/2302.05666
    """

    def __init__(self, num_classes=2, weights=None, smoothing=0.1, k=3):
        super(JML1, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        if weights is not None:
            # assume is weights is not none then it is a tensor
            self.register_buffer("weights", weights)
        self.k = k

    def forward(self, y_pred, y_true):
        # For boundary detection
        maxpool = torch.nn.MaxPool2d(kernel_size=self.k, stride=1, padding=self.k // 2)
        # Ensure y_pred is in probability form
        y_pred = F.softmax(y_pred, dim=1)
        # One-hot encode y_true, expected y_true to be [batch_size, height, width] for image data
        y_true = F.one_hot(y_true, num_classes=self.num_classes).float()

        # OneHot moves the channel dimension to the last dimension
        # SoftMax does not change the order of the dimensions
        # So we change the one-hot encoded, y_true tensor to have the same order as the softmaxed y_pred
        y_true = y_true.permute(0, 3, 2, 1)
        y_true_pooled = maxpool(y_true)
        boundaries = (y_true_pooled > y_true).any(dim=1, keepdim=True)
        smoothed_boundaries = y_true * (1 - self.smoothing) + (self.smoothing / self.num_classes)
        labels = torch.where(boundaries, smoothed_boundaries, y_true)

        # Normalize over the spatial dimensions
        labels_norm = torch.norm(labels, p=1, dim=[2, 3])
        pred_norm = torch.norm(y_pred, p=1, dim=[2, 3])
        err_norm = torch.norm(labels - y_pred, p=1, dim=[2, 3])

        jaccard_index = (labels_norm + pred_norm - err_norm) / (labels_norm + pred_norm + err_norm)

        if getattr(self, "weights", None) is not None:
            jaccard_index = self.weights.unsqueeze(0) * jaccard_index
        # The loss is 1 - the average Jaccard index over the batch and classes
        # Averaging over both batch and classes
        return 1 - torch.mean(jaccard_index)


class WeightedComboLoss(nn.Module):
    def __init__(self, loss_a: nn.Module, loss_b: nn.Module, alpha=0.5, beta=0.5) -> None:
        super(WeightedComboLoss, self).__init__()
        self.loss_a = loss_a
        self.loss_b = loss_b
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        return self.loss_a(y_pred, y_true) * self.alpha + self.loss_b(y_pred, y_true) * self.beta


class PowerJaccardLoss(nn.Module):
    def __init__(self, num_classes=2, weights=None, smoothing=1e-6, power=1.0):
        super(PowerJaccardLoss, self).__init__()
        self.num_classes = num_classes
        # If weights are not provided, use equal weighting
        if weights is not None:
            self.register_buffer("weights", weights)
        self.smoothing = smoothing
        self.power = power

    def forward(self, y_pred, y_true):
        # Ensure y_pred is in probability form
        y_pred = F.softmax(y_pred, dim=1)
        # One-hot encode y_true, expected y_true to be [batch_size, height, width] for image data
        y_true = F.one_hot(y_true, num_classes=self.num_classes).float()

        # Change shape to [batch, classes, other dimensions...]
        y_pred = y_pred.permute(0, 2, 3, 1)
        y_true = y_true.permute(0, 2, 3, 1)

        # Flatten the last dimensions to simplify the sum operations
        y_pred = y_pred.contiguous().view(y_pred.shape[0], y_pred.shape[1], -1)
        y_true = y_true.contiguous().view(y_true.shape[0], y_true.shape[1], -1)

        # Calculate dot product and L1 norm across the spatial dimensions for each example in the batch
        dot_product = torch.sum(y_true * y_pred, dim=2)
        l1_norm = torch.sum(torch.abs(y_pred - y_true), dim=2)

        # Calculate the modified Jaccard index per example in the batch
        jaccard_index = dot_product / (dot_product + l1_norm + self.smoothing)

        if getattr(self, "weights", None) is not None:
            jaccard_index = self.weights * jaccard_index
        # The loss is 1 - the average Jaccard index over the batch and classes
        # Averaging over both batch and classes
        return 1 - torch.mean(jaccard_index)


class WeightedMSELoss(nn.Module):
    def __init__(self, weights=None, scale=1.0):
        """
        Create a weighted MSE loss function

        Parameters:
        ----------
        weights: torch.Tensor
            The weights to use for the loss
        scale: float
            The scale factor to apply to the loss
            Default: 1.0 (no scaling)

        Throws:
        -------
        AssertionError: If the input and target are not on the same device
        AssertionError: If the weights are provided and not on the same device as the input
        """
        super(WeightedMSELoss, self).__init__()
        self.register_buffer("weight", weights)
        self.register_buffer("scale", torch.tensor(scale))  # Register the scale factor

    def forward(self, input, target, classes=None):
        assert input.device == target.device, "Input and target must be on the same device"
        if self.weight is not None:
            assert (
                self.weight.device == input.device
            ), "Weights must be on the same device as input"
            # Normalize the weights to sum to 1
            # map the classes to the weights
            if classes is not None:
                device = self.weight.device
                weights = torch.tensor(
                    [self.weight[cls] for cls in classes], dtype=torch.float
                ).to(
                    device
                )  # Get the weights for the classes

            normalized_weight = weights / weights.sum()
            # Expand the weights to the same shape as the input for broadcasting
            normalized_weight = normalized_weight.view(-1, 1, 1, 1).expand_as(input)
            # Compute the weighted MSE
            loss = (normalized_weight * (input - target) ** 2).mean()
        else:
            # Fallback to regular MSE if no weights are provided
            loss = ((input - target) ** 2).mean()
        return loss * self.scale  # Scale the loss by the scale factor


def with_loss_fn(
    loss_fn: Union[str, Callable, nn.Module], **kwargs
) -> Callable[[Type[pl.LightningModule]], Type[pl.LightningModule]]:
    """
    A decorator to add a loss function to a LightningModule
    """

    def decorator(cls: Type[pl.LightningDataModule]) -> Type[pl.LightningDataModule]:
        if isinstance(loss_fn, str):
            loss_fn_instance = globals()[loss_fn](**kwargs)
        else:
            loss_fn_instance = loss_fn(**kwargs) if callable(loss_fn) else loss_fn

        class WrappedClass(cls, pl.LightningModule):
            def __init__(self, *args, **init_kwargs):
                super(WrappedClass, self).__init__(*args, **init_kwargs)

            def loss_fn(self, *args, **kwargs):
                return loss_fn_instance(*args, **kwargs)

        return WrappedClass

    return decorator


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction="mean"):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = torch.sum(torch.mul(predict, target), dim=1)

        # Denominator terms
        sum_predict_p = torch.sum(predict.pow(self.p), dim=1)
        # For binary 0/1 targets, target.pow(self.p) == target for p >= 1.
        # If target can have values other than 0/1 (not typical for one-hot slices), .pow(self.p) is important.
        sum_target_p = torch.sum(target.pow(self.p), dim=1)

        # Standard Dice coefficient
        dice_coeff = (2.0 * intersection + self.smooth) / (
            sum_predict_p + sum_target_p + self.smooth
        )

        loss = 1.0 - dice_coeff

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise Exception("Unexpected reduction {}".format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss for multiclass segmentation
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, H, W]
        target: A tensor of shape [N, H, W]
        other args pass to BinaryDiceLoss
    Returns:
        Loss tensor according to BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size do not match"
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)  # Now target is [N, H, W]
        num_classes = predict.shape[1]

        # One-hot encode the target tensor if necessary
        if target.dim() == 3:  # [N, H, W]
            target = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(num_classes):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert (
                        self.weight.shape[0] == num_classes
                    ), "Expect weight shape [{}], got [{}]".format(
                        num_classes, self.weight.shape[0]
                    )
                    dice_loss *= self.weight[i]
                total_loss += dice_loss

        num_contributing_classes = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                if (
                    self.weight is not None and self.weight[i] == 0
                ):  # Also consider zero-weighted classes
                    continue
                num_contributing_classes += 1

        if num_contributing_classes > 0:
            return total_loss / num_contributing_classes
        else:
            return torch.tensor(
                0.0, device=predict.device, dtype=predict.dtype
            )  # Or handle as appropriate

# Author: Arash Khoeini
# Email: arashkhoeini[at]gmail[dot]com

class InfoNCE(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        Computes the InfoNCE loss.
        
        Args:
            features (torch.Tensor): The feature matrix of shape [2 * batch_size, feature_dim], 
                                     where features[:batch_size] are the representations of 
                                     the first set of augmented images, and features[batch_size:] 
                                     are the representations of the second set.
        
        Returns:
            torch.Tensor: The computed InfoNCE loss.
        """
        # Normalize features to have unit norm
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Get batch size
        batch_size = features.shape[0] // 2
        
        # Construct labels where each sample's positive pair is in the other view
        labels = torch.arange(batch_size, device=features.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # Mask out self-similarities by setting the diagonal elements to -inf
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

class PixelWiseInfoNCE(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(PixelWiseInfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, features_v1, features_v2): # Expects two separate feature tensors
        """
        Computes the InfoNCE loss for pixel-level features.
        
        Args:
            features_v1 (torch.Tensor): Feature matrix from the first augmented view. Shape: [B, C, H_feat, W_feat]
            features_v2 (torch.Tensor): Feature matrix from the second augmented view. Shape: [B, C, H_feat, W_feat]
        
        Returns:
            torch.Tensor: The computed InfoNCE loss.
        """
        # Flatten features from (B, C, H_feat, W_feat) to (B*H_feat*W_feat, C)
        # and normalize.
        features_v1_flat = F.normalize(features_v1.permute(0, 2, 3, 1).reshape(-1, features_v1.shape[1]), dim=1)
        features_v2_flat = F.normalize(features_v2.permute(0, 2, 3, 1).reshape(-1, features_v2.shape[1]), dim=1)
        
        # Now the 'batch_size' for InfoNCE is the total number of pixels/patches per original image batch
        pixel_batch_size = features_v1_flat.shape[0] # B * H_feat * W_feat

        # Concatenate features: features_v1_flat stacked on top of features_v2_flat
        # Shape: [2 * pixel_batch_size, C]
        features_combined = torch.cat([features_v1_flat, features_v2_flat], dim=0)

        # Compute similarity matrix: [2 * pixel_batch_size, 2 * pixel_batch_size]
        similarity_matrix = torch.matmul(features_combined, features_combined.T) / self.temperature

        # Construct labels:
        # For a pixel `i` from `features_v1_flat` (at index `i`), its positive is `features_v2_flat[i]` (at index `i + pixel_batch_size`)
        # For a pixel `j` from `features_v2_flat` (at index `j + pixel_batch_size`), its positive is `features_v1_flat[j]` (at index `j`)
        
        # Labels for the first half (queries from features_v1_flat)
        labels_first_half = torch.arange(pixel_batch_size, device=features_combined.device) + pixel_batch_size
        
        # Labels for the second half (queries from features_v2_flat)
        labels_second_half = torch.arange(pixel_batch_size, device=features_combined.device)

        # Concatenate labels: [positives for first half, positives for second half]
        labels = torch.cat([labels_first_half, labels_second_half], dim=0)

        # Mask out self-similarities
        mask = torch.eye(2 * pixel_batch_size, dtype=torch.bool, device=features_combined.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # InfoNCE loss (cross-entropy)
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

# Example usage
if __name__ == "__main__":
    # Example prediction and target tensors
    preds = torch.randn(2, 3, 256, 256)  # [N, C, H, W]
    targets = torch.randint(0, 3, (2, 256, 256))  # [N, H, W]

    # Instantiate the loss function
    loss_fn = DiceLoss(weight=torch.tensor([1.0, 1.0, 1.0]), ignore_index=None)

    # Compute the loss
    loss = loss_fn(preds, targets)
    print(f"Loss: {loss.item()}")
