import torch
from tqdm import tqdm

from combined.training.metrics import binary_dice, binary_iou, binary_pixel_accuracy

def train_one_epoch(model, loader, optimizer, criterion, device, threshold=0.5):
    model.train()
    running = {"loss": 0.0, "acc": 0.0, "iou": 0.0, "dice": 0.0}

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running["loss"] += loss.item()
        running["acc"] += binary_pixel_accuracy(logits.detach(), masks, threshold)
        running["iou"] += binary_iou(logits.detach(), masks, threshold)
        running["dice"] += binary_dice(logits.detach(), masks, threshold)

    n = max(1, len(loader))
    return {key: value / n for key, value in running.items()}

@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running = {"loss": 0.0, "acc": 0.0, "iou": 0.0, "dice": 0.0}

    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        running["loss"] += loss.item()
        running["acc"] += binary_pixel_accuracy(logits, masks, threshold)
        running["iou"] += binary_iou(logits, masks, threshold)
        running["dice"] += binary_dice(logits, masks, threshold)

    n = max(1, len(loader))
    return {key: value / n for key, value in running.items()}
