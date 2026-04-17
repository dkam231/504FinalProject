import os
import cv2
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

import albumentations as A
from albumentations.pytorch import ToTensorV2


NUM_CLASSES = 2
CLASS_NAMES = ["background", "foreground"]


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def pixel_accuracy(preds, targets):
    preds = preds.view(-1)
    targets = targets.view(-1)
    return (preds == targets).float().mean().item()


def mean_iou_binary(preds, targets):
    preds = preds.view(-1)
    targets = targets.view(-1)

    pred_fg = preds == 1
    target_fg = targets == 1

    intersection = (pred_fg & target_fg).sum().item()
    union = (pred_fg | target_fg).sum().item()

    if union == 0:
        return 1.0
    return intersection / union


class SUIMBinaryDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, filter_empty=False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.filter_empty = filter_empty
        self.size_warning_count = 0

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask dir not found: {self.mask_dir}")

        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        mask_exts = {".png", ".bmp", ".tif", ".tiff"}

        image_files = sorted([
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in img_exts
        ])

        mask_map = {}
        duplicate_stems = []

        for root, _, files in os.walk(self.mask_dir):
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() not in mask_exts:
                    continue
                stem = p.stem
                if stem in mask_map:
                    duplicate_stems.append(stem)
                else:
                    mask_map[stem] = p

        if duplicate_stems:
            print(f"Warning: found duplicate mask stems: {len(duplicate_stems)}")
            print(f"First 10 duplicates: {duplicate_stems[:10]}")

        self.samples = []
        missing_masks = 0
        filtered_empty = 0

        print("\n" + "=" * 80)
        print("Building dataset")
        print(f"Images dir: {self.image_dir}")
        print(f"Masks dir:  {self.mask_dir}")
        print(f"Found images: {len(image_files)}")
        print(f"Found masks:  {len(mask_map)}")
        print(f"Filter empty masks: {filter_empty}")
        print("=" * 80)

        for img_path in tqdm(image_files, desc="Scanning images", unit="img"):
            stem = img_path.stem
            mask_path = mask_map.get(stem)

            if mask_path is None:
                missing_masks += 1
                if missing_masks <= 10:
                    print(f"[MISSING MASK] {img_path.name}")
                continue

            if filter_empty:
                raw_mask = self._read_mask(mask_path)
                bin_mask = self._convert_rgb_code_mask_to_binary(raw_mask)
                if np.max(bin_mask) == 0:
                    filtered_empty += 1
                    continue

            self.samples.append((img_path, mask_path))

        print("\nDataset summary")
        print(f"Valid samples:   {len(self.samples)}")
        print(f"Missing masks:   {missing_masks}")
        print(f"Filtered empty:  {filtered_empty}")
        print("=" * 80 + "\n")

        if len(self.samples) == 0:
            raise ValueError("No valid samples found.")

        self._debug_first_masks()

    def _read_mask(self, mask_path: Path):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")
        return mask

    def _convert_rgb_code_mask_to_binary(self, mask):
        """
        SUIM mask coding:
        000 -> background
        111 -> background
        others -> foreground

        OpenCV reads BGR, so convert to RGB first.
        """
        if len(mask.shape) == 2:
            return (mask > 0).astype(np.uint8)

        if mask.shape[2] < 3:
            return (mask[:, :, 0] > 0).astype(np.uint8)

        rgb = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2RGB)

        r = (rgb[:, :, 0] > 127).astype(np.uint8)
        g = (rgb[:, :, 1] > 127).astype(np.uint8)
        b = (rgb[:, :, 2] > 127).astype(np.uint8)

        code = (r << 2) | (g << 1) | b

        # 000 and 111 are background
        bin_mask = np.where((code == 0) | (code == 7), 0, 1).astype(np.uint8)
        return bin_mask

    def _debug_first_masks(self, n=5):
        print("Inspecting first masks...")
        for img_path, mask_path in self.samples[:min(n, len(self.samples))]:
            raw_mask = self._read_mask(mask_path)
            bin_mask = self._convert_rgb_code_mask_to_binary(raw_mask)

            if len(raw_mask.shape) == 3 and raw_mask.shape[2] >= 3:
                rgb = cv2.cvtColor(raw_mask[:, :, :3], cv2.COLOR_BGR2RGB)
                r = (rgb[:, :, 0] > 127).astype(np.uint8)
                g = (rgb[:, :, 1] > 127).astype(np.uint8)
                b = (rgb[:, :, 2] > 127).astype(np.uint8)
                code = (r << 2) | (g << 1) | b
                raw_unique = np.unique(code).tolist()
            else:
                raw_unique = np.unique(raw_mask).tolist()

            print(
                f"[MASK DEBUG] img={img_path.name} mask={mask_path.name} "
                f"raw_codes={raw_unique[:20]} "
                f"bin_unique={np.unique(bin_mask).tolist()} "
                f"fg_ratio={(bin_mask == 1).mean():.6f}"
            )
        print()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        raw_mask = self._read_mask(mask_path)
        mask = self._convert_rgb_code_mask_to_binary(raw_mask)

        if image.shape[:2] != mask.shape[:2]:
            if self.size_warning_count < 20:
                print(
                    f"[SIZE WARNING] image/mask size mismatch, resizing mask to image size:\n"
                    f"image: {img_path} -> {image.shape}\n"
                    f"mask:  {mask_path} -> {mask.shape}"
                )
            self.size_warning_count += 1
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"].long()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return {
            "image": image,
            "mask": mask,
            "name": img_path.name,
        }


def create_dataloaders(
    train_img_dir,
    train_mask_dir,
    val_img_dir,
    val_mask_dir,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    train_transform=None,
    val_transform=None,
):
    train_dataset = SUIMBinaryDataset(
        train_img_dir,
        train_mask_dir,
        transform=train_transform,
        filter_empty=True,
    )

    val_dataset = SUIMBinaryDataset(
        val_img_dir,
        val_mask_dir,
        transform=val_transform,
        filter_empty=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def build_deeplab_model():
    model = models.segmentation.deeplabv3_resnet50(
        weights=DeepLabV3_ResNet50_Weights.DEFAULT
    )
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier = None
    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_miou = 0.0

    for batch in tqdm(loader, desc="Training"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)["out"]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)

        running_loss += loss.item()
        running_acc += pixel_accuracy(preds, masks)
        running_miou += mean_iou_binary(preds, masks)

    n = len(loader)
    return running_loss / n, running_acc / n, running_miou / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    running_miou = 0.0

    for batch in tqdm(loader, desc="Validation"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)["out"]
        loss = criterion(outputs, masks)

        preds = torch.argmax(outputs, dim=1)

        running_loss += loss.item()
        running_acc += pixel_accuracy(preds, masks)
        running_miou += mean_iou_binary(preds, masks)

    n = len(loader)
    return running_loss / n, running_acc / n, running_miou / n


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLabV3 on SUIM_Dataset.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_base = os.path.abspath(os.path.join(script_dir, "../../data"))

    train_img_dir = os.path.join(dataset_base, "train_val", "images")
    test_img_dir = os.path.join(dataset_base, "TEST", "images")

    train_mask_dir = os.path.join(dataset_base, "train_val", "masks")
    test_mask_dir = os.path.join(dataset_base, "TEST", "masks")

    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    img_size = args.img_size
    num_workers = (0 if os.name == "nt" else 4) if args.num_workers < 0 else args.num_workers

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Train imgs:   {train_img_dir}")
    print(f"Test imgs:    {test_img_dir}")
    print(f"Train masks:  {train_mask_dir}")
    print(f"Test masks:   {test_mask_dir}")
    print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")

    train_loader, val_loader = create_dataloaders(
        train_img_dir=train_img_dir,
        train_mask_dir=train_mask_dir,
        val_img_dir=test_img_dir,
        val_mask_dir=test_mask_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        train_transform=get_train_transform(img_size),
        val_transform=get_val_transform(img_size),
    )

    sample_batch = next(iter(train_loader))
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"Image batch shape: {tuple(sample_batch['image'].shape)}")
    print(f"Mask batch shape: {tuple(sample_batch['mask'].shape)}")
    print(f"Unique labels in first mask: {torch.unique(sample_batch['mask'][0]).tolist()}")

    model = build_deeplab_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_miou = 0.0
    checkpoints_dir = Path(script_dir) / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        train_loss, train_acc, train_miou = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_miou = validate(
            model, val_loader, criterion, device
        )

        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train mIoU: {train_miou:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val   mIoU: {val_miou:.4f}")

        latest_path = checkpoints_dir / "latest_deeplab_suim_binary.pth"
        torch.save(model.state_dict(), latest_path)

        if val_miou > best_miou:
            best_miou = val_miou
            best_path = checkpoints_dir / "best_deeplab_suim_binary.pth"
            torch.save(model.state_dict(), best_path)
            print("Saved best model.")


if __name__ == "__main__":
    main()