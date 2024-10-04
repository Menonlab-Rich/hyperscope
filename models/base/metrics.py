from torchmetrics import Metric
from torchmetrics.segmentation import GeneralizedDiceScore as TorchGeneralizedDiceScore
import torch.nn.functional as F
import torch

class ClassSpecificAccuracy(Metric):
    def __init__(self, ignore_indices=[0], dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_indices = ignore_indices
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds and target should be long tensors
        preds = preds.argmax(dim=1)
        mask = target not in self.ignore_indices
        preds = preds[mask]
        target = target[mask]
        self.correct += (preds == target).sum()
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total

from torchmetrics.segmentation import GeneralizedDiceScore as TorchGeneralizedDiceScore
import torch.nn.functional as F
import torch

class GeneralizedDiceScore(TorchGeneralizedDiceScore):
    def __init__(self, num_classes, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        assert preds.shape[0] == target.shape[0], "predict & target batch size don't match"
        
        # Squeeze target to remove singleton dimension if present
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
        
        # One-hot encode the target tensor if necessary
        if target.dim() == 3:  # [N, H, W]
            target = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        
        # Apply softmax to predictions to get probabilities
        preds = F.softmax(preds, dim=1)
        
        # Convert probabilities to class indices
        preds = torch.argmax(preds, dim=1)  # [N, C, H, W] -> [N, H, W]
        
        # One-hot encode the predictions
        preds = F.one_hot(preds, self.num_classes).permute(0, 3, 1, 2)  # [N, H, W] -> [N, C, H, W]
        
        # Call the parent class update method
        super().update(preds, target)

if __name__ == "__main__":
    # Example usage
    gds = GeneralizedDiceScore(num_classes=3, include_background=False)
    preds = torch.randn(4, 3, 128, 128)
    target = torch.randint(0, 3, (4, 128, 128))
    gds.update(preds, target)
    print(gds.compute())

