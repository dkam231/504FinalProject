import csv
import os
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import torch
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from tqdm import tqdm

from dataloader import SUIM_BINARY_CLASSES, create_suim_dataloaders
from model import UNet
from utils import BCEDiceFocalLoss, binary_iou, pixel_accuracy


def get_train_transform(img_size=(320, 240)):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(p=0.3),
        A.Resize(img_size[1], img_size[0]),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])


def get_val_transform(img_size=(320, 240)):
    return A.Compose([
        A.Resize(img_size[1], img_size[0]),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])


def logits_to_binary_predictions(logits):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    return preds


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, epochs):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_iou = 0.0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)
    for step, batch in enumerate(progress_bar, start=1):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).unsqueeze(1).float()

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        preds = logits_to_binary_predictions(logits)
        batch_acc = pixel_accuracy(preds.squeeze(1), masks.squeeze(1).long())
        batch_iou = binary_iou(preds.squeeze(1), masks.squeeze(1).long())

        running_loss += loss.item()
        running_acc += batch_acc
        running_iou += batch_iou

        progress_bar.set_postfix(
            loss=f"{running_loss / step:.4f}",
            acc=f"{running_acc / step:.4f}",
            iou=f"{running_iou / step:.4f}",
        )

    n = len(loader)
    return running_loss / n, running_acc / n, running_iou / n


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, epochs):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    running_iou = 0.0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False)
    for step, batch in enumerate(progress_bar, start=1):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).unsqueeze(1).float()

        logits = model(images)
        loss = criterion(logits, masks)

        preds = logits_to_binary_predictions(logits)
        batch_acc = pixel_accuracy(preds.squeeze(1), masks.squeeze(1).long())
        batch_iou = binary_iou(preds.squeeze(1), masks.squeeze(1).long())

        running_loss += loss.item()
        running_acc += batch_acc
        running_iou += batch_iou

        progress_bar.set_postfix(
            loss=f"{running_loss / step:.4f}",
            acc=f"{running_acc / step:.4f}",
            iou=f"{running_iou / step:.4f}",
        )

    n = len(loader)
    return running_loss / n, running_acc / n, running_iou / n


def save_history_csv(history, output_path):
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "train_iou",
        "val_loss",
        "val_acc",
        "val_iou",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_training_curves(history, output_path):
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_iou = [entry["train_iou"] for entry in history]
    val_iou = [entry["val_iou"] for entry in history]
    train_acc = [entry["train_acc"] for entry in history]
    val_acc = [entry["val_acc"] for entry in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Train Acc")
    axes[1].plot(epochs, val_acc, label="Val Acc")
    axes[1].set_title("Pixel Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, train_iou, label="Train IoU")
    axes[2].plot(epochs, val_iou, label="Val IoU")
    axes[2].set_title("Foreground/Background IoU")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    project_dir = Path(__file__).resolve().parent
    data_root = Path(os.environ.get("SUIM_ROOT", project_dir / "data"))
    batch_size = int(os.environ.get("SUIM_BATCH_SIZE", 8))
    lr = float(os.environ.get("SUIM_LR", 1e-4))
    epochs = int(os.environ.get("SUIM_EPOCHS", 30))
    img_width = int(os.environ.get("SUIM_IMG_WIDTH", 320))
    img_height = int(os.environ.get("SUIM_IMG_HEIGHT", 240))
    img_size = (img_width, img_height)
    val_ratio = float(os.environ.get("SUIM_VAL_RATIO", 0.2))
    seed = int(os.environ.get("SUIM_SEED", 42))
    num_workers = 0 if os.name == "nt" else int(os.environ.get("SUIM_NUM_WORKERS", 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not data_root.exists():
        raise FileNotFoundError(
            f"SUIM dataset root not found: {data_root}\n"
            "Expected structure:\n"
            "  data/train_val/images\n"
            "  data/train_val/masks\n"
            "  data/TEST/images\n"
            "  data/TEST/masks"
        )

    train_loader, val_loader, _ = create_suim_dataloaders(
        root=data_root,
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        train_transform=get_train_transform(img_size),
        val_transform=get_val_transform(img_size),
    )

    sample_batch = next(iter(train_loader))
    print(f"Dataset root: {data_root}")
    print(f"Device: {device}")
    print(f"Task: Binary foreground/background segmentation")
    print(f"Classes: {SUIM_BINARY_CLASSES}")
    print(f"Background IDs: [0, 7] | Foreground IDs: [1, 2, 3, 4, 5, 6]")
    print(f"Resize: {img_width}x{img_height}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"Image batch shape: {tuple(sample_batch['image'].shape)}")
    print(f"Mask batch shape: {tuple(sample_batch['mask'].shape)}")
    print(f"Unique labels in first mask: {torch.unique(sample_batch['mask'][0]).tolist()}")

    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = BCEDiceFocalLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    checkpoints_dir = project_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    history = []
    best_iou = 0.0
    best_model_path = checkpoints_dir / "best_unet_suim_binary.pth"
    last_model_path = checkpoints_dir / "last_unet_suim_binary.pth"
    history_csv_path = checkpoints_dir / "training_history_binary.csv"
    curves_png_path = checkpoints_dir / "training_curves_binary.png"

    print("\nStarting training...")
    for epoch in range(epochs):
        train_loss, train_acc, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, epochs
        )
        val_loss, val_acc, val_iou = validate(
            model, val_loader, criterion, device, epoch, epochs
        )

        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_iou": val_iou,
        }
        history.append(epoch_summary)

        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val   IoU: {val_iou:.4f}")

        torch.save(model.state_dict(), last_model_path)
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to: {best_model_path}")

        save_history_csv(history, history_csv_path)
        plot_training_curves(history, curves_png_path)
        print(f"Saved training history to: {history_csv_path}")
        print(f"Saved training curves to: {curves_png_path}")

    print("\nTraining complete.")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Best checkpoint: {best_model_path}")
    print(f"Last checkpoint: {last_model_path}")


if __name__ == "__main__":
    main()
