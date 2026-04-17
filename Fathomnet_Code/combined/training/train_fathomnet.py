import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from combined.datasets.fathomnet_binary_dataset import create_fathomnet_binary_dataloaders
from combined.datasets.transforms import get_train_transform, get_val_transform
from combined.models.unet import UNet
from combined.training.engine import evaluate, train_one_epoch
from combined.training.losses import BCEDiceLoss
from combined.utils.checkpointing import load_checkpoint, save_checkpoint
from combined.utils.seed import seed_everything

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune binary U-Net on FathomNet.")
    parser.add_argument("--data-root", required=True, help="Path to FathomNet root directory.")
    parser.add_argument("--output-dir", default="combined/outputs/fathomnet")
    parser.add_argument("--pretrained-checkpoint", required=True, help="Path to SUIM-pretrained checkpoint.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--loss", choices=["bce", "bce_dice"], default="bce")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu"
    )
    pin_memory = torch.cuda.is_available() and device.type == "cuda"

    train_loader, val_loader = create_fathomnet_binary_dataloaders(
        root=Path(args.data_root),
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        train_transform=get_train_transform(args.img_size),
        val_transform=get_val_transform(args.img_size),
    )

    model = UNet(n_channels=3, n_classes=1).to(device)
    load_checkpoint(args.pretrained_checkpoint, model=model, device=device, strict=True)

    criterion = nn.BCEWithLogitsLoss() if args.loss == "bce" else BCEDiceLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, args.threshold)
        val_metrics = evaluate(model, val_loader, criterion, device, args.threshold)

        print(f"Epoch [{epoch}/{args.epochs}]")
        print("Train | " + " ".join(f"{k}={v:.4f}" for k, v in train_metrics.items()))
        print("Val   | " + " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

        save_checkpoint(ckpt_dir / "fathomnet_fg_bg_last.pth", model=model, optimizer=optimizer, epoch=epoch, metrics=val_metrics)

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint(ckpt_dir / "fathomnet_fg_bg_best.pth", model=model, optimizer=optimizer, epoch=epoch, metrics=val_metrics)
            print("Saved best FathomNet checkpoint.")

if __name__ == "__main__":
    main()
