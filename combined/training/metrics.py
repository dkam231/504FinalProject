import torch

def _threshold_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs >= threshold).float()

def binary_pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = _threshold_logits(logits, threshold=threshold)
    correct = (preds == targets).float().sum()
    total = torch.numel(targets)
    return (correct / total).item()

def binary_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    preds = _threshold_logits(logits, threshold=threshold).bool()
    targets = targets.bool()
    intersection = (preds & targets).float().sum()
    union = (preds | targets).float().sum()
    return ((intersection + eps) / (union + eps)).item()

def binary_dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    preds = _threshold_logits(logits, threshold=threshold)
    intersection = (preds * targets).sum()
    denominator = preds.sum() + targets.sum()
    return ((2 * intersection + eps) / (denominator + eps)).item()
