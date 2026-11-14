import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice_score)

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)

class MetricTracker:
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix.fill(0)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds_np = preds.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()

        mask = (targets_np >= 0) & (targets_np < self.num_classes)
        hist = np.bincount(
            self.num_classes * targets_np[mask] + preds_np[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += hist

    def get_metrics(self):
        hist = self.confusion_matrix
        tp = np.diag(hist)
        fp = hist.sum(axis=0) - tp
        fn = hist.sum(axis=1) - tp

        union = tp + fp + fn
        ious = np.where(union > 0, tp / union, 0.0)
        miou = float(np.mean(ious))

        dice_den = 2 * tp + fp + fn
        dice_scores = np.where(dice_den > 0, (2 * tp) / dice_den, 0.0)
        mean_dice = float(np.mean(dice_scores))

        total_correct = tp.sum()
        total_pixels = hist.sum()
        pixel_accuracy = float(total_correct / total_pixels) if total_pixels > 0 else 0.0

        return {
            "per_class_iou": {f"class_{i}": float(ious[i]) for i in range(self.num_classes)},
            "miou": miou,
            "per_class_dice": {f"class_{i}": float(dice_scores[i]) for i in range(self.num_classes)},
            "mean_dice": mean_dice,
            "pixel_accuracy": pixel_accuracy,
            "confusion_matrix": hist.tolist()
        }

def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}
