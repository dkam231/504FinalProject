import torch

def pixel_accuracy(preds, targets):
    correct = (preds == targets).float().sum()
    total = torch.numel(targets)
    return (correct / total).item()

def mean_iou(preds, targets, num_classes, eps=1e-6):
    ious = []

    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)

        intersection = (pred_cls & target_cls).float().sum()
        union = (pred_cls | target_cls).float().sum()

        if union == 0:
            continue

        iou = (intersection + eps) / (union + eps)
        ious.append(iou.item())

    if len(ious) == 0:
        return 0.0
    return sum(ious) / len(ious)