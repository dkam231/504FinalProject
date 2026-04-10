import torch
import torch.nn as nn
import torch.nn.functional as F


def pixel_accuracy(preds, targets):
    correct = (preds == targets).float().sum()
    total = torch.numel(targets)
    return (correct / total).item()


def binary_iou(preds, targets, eps=1e-6):
    preds = preds.bool()
    targets = targets.bool()

    intersection = (preds & targets).float().sum()
    union = (preds | targets).float().sum()

    if union == 0:
        return 1.0

    return ((intersection + eps) / (union + eps)).item()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal = alpha_t * ((1 - pt) ** self.gamma) * bce
        return focal.mean()


class BCEDiceFocalLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, focal_weight=1.0, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, logits, targets):
        targets = targets.float()
        return (
            self.bce_weight * self.bce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
            + self.focal_weight * self.focal(logits, targets)
        )
