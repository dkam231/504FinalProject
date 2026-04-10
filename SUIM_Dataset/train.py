import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from dataloader import SUIM_CLASSES, create_suim_dataloaders
from model import UNet
from utils import pixel_accuracy, mean_iou

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def train_one_epoch(model, loader, optimizer, criterion, device, num_classes):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_miou = 0.0

    for batch in tqdm(loader, desc="Training"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)  # [B, C, H, W]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)

        running_loss += loss.item()
        running_acc += pixel_accuracy(preds, masks)
        running_miou += mean_iou(preds, masks, num_classes=num_classes)

    n = len(loader)
    return running_loss / n, running_acc / n, running_miou / n


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    running_miou = 0.0

    for batch in tqdm(loader, desc="Validation"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        preds = torch.argmax(outputs, dim=1)

        running_loss += loss.item()
        running_acc += pixel_accuracy(preds, masks)
        running_miou += mean_iou(preds, masks, num_classes=num_classes)

    n = len(loader)
    return running_loss / n, running_acc / n, running_miou / n


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net on SUIM dataset.")
    parser.add_argument("--data-root", default="", help="Path to SUIM root directory.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=-1, help="-1 uses OS-based default.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = Path(os.environ.get("SUIM_ROOT", project_dir / "SUIM"))
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    num_classes = len(SUIM_CLASSES)
    img_size = args.img_size
    num_workers = (0 if os.name == "nt" else 4) if args.num_workers < 0 else args.num_workers
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not data_root.exists():
        raise FileNotFoundError(
            f"SUIM dataset root not found: {data_root}\n"
            "Expected structure:\n"
            "  SUIM/train_val/images\n"
            "  SUIM/train_val/masks\n"
            "  SUIM/TEST/images\n"
            "  SUIM/TEST/masks"
        )

    train_loader, val_loader, _ = create_suim_dataloaders(
        root=data_root,
        batch_size=batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        train_transform=get_train_transform(img_size),
        val_transform=get_val_transform(img_size),
    )

    sample_batch = next(iter(train_loader))
    print(f"Dataset root: {data_root}")
    print(f"Classes ({num_classes}): {SUIM_CLASSES}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"Image batch shape: {tuple(sample_batch['image'].shape)}")
    print(f"Mask batch shape: {tuple(sample_batch['mask'].shape)}")
    print(f"Unique labels in first mask: {torch.unique(sample_batch['mask'][0]).tolist()}")

    model = UNet(n_channels=3, n_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_miou = 0.0
    checkpoints_dir = project_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        train_loss, train_acc, train_miou = train_one_epoch(
            model, train_loader, optimizer, criterion, device, num_classes
        )
        val_loss, val_acc, val_miou = validate(
            model, val_loader, criterion, device, num_classes
        )

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train mIoU: {train_miou:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val   mIoU: {val_miou:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), checkpoints_dir / "best_unet_suim.pth")
            print("Saved best model.")

if __name__ == "__main__":
    main()
