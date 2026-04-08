from __future__ import annotations

import os
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, random_split


# ============================================================
# SUIM label definitions
# ============================================================
SUIM_CLASSES: List[str] = [
    "background",
    "human_diver",
    "aquatic_plants",
    "wrecks_ruins",
    "robots_instruments",
    "reefs_invertebrates",
    "fish_vertebrates",
    "seafloor_rocks",
]

SUIM_COLOR_MAP: Dict[Tuple[int, int, int], int] = {
    (0, 0, 0): 0,
    (0, 0, 255): 1,
    (0, 255, 0): 2,
    (0, 255, 255): 3,
    (255, 0, 0): 4,
    (255, 0, 255): 5,
    (255, 255, 0): 6,
    (255, 255, 255): 7,
}

SUIM_INDEX_TO_COLOR: Dict[int, Tuple[int, int, int]] = {
    v: k for k, v in SUIM_COLOR_MAP.items()
}


@dataclass
class SUIMSample:
    image_path: Path
    mask_path: Path


# ============================================================
# Download helpers
# ============================================================
def _extract_google_drive_id(url_or_id: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", url_or_id):
        return url_or_id

    match = re.search(r"[?&]id=([A-Za-z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)

    match = re.search(r"/d/([A-Za-z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)

    raise ValueError(f"Could not parse Google Drive file ID from: {url_or_id}")


def extract_archive(archive_path: Union[str, Path], destination_dir: Union[str, Path]) -> None:
    archive_path = Path(archive_path)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(destination_dir)
        return

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(destination_dir)
        return

    raise ValueError(f"Unsupported archive format: {archive_path}")


def download_suim(
    destination_dir: Union[str, Path],
    url_or_id: str,
    output_name: Optional[str] = None,
    extract: bool = True,
) -> Path:
    """
    Download the SUIM dataset from a Google Drive link or file ID.

    Requires:
        pip install gdown
    """
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "gdown is required for download_suim(). Install it with: pip install gdown"
        ) from exc

    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    file_id = _extract_google_drive_id(url_or_id)
    archive_path = destination_dir / (output_name or "suim_dataset.zip")

    gdown.download(id=file_id, output=str(archive_path), quiet=False, fuzzy=True)

    if extract:
        extract_archive(archive_path, destination_dir)

    return archive_path


# ============================================================
# Pairing and mask conversion helpers
# ============================================================
def _list_image_files(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _stem_without_extra_mask_tokens(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"(_mask|_gt|_label|_labels)$", "", stem)
    return stem


def _pair_images_and_masks(images_dir: Path, masks_dir: Path) -> List[SUIMSample]:
    image_files = _list_image_files(images_dir)
    mask_files = _list_image_files(masks_dir)

    if not image_files:
        raise FileNotFoundError(f"No images found in: {images_dir}")
    if not mask_files:
        raise FileNotFoundError(f"No masks found in: {masks_dir}")

    mask_by_name = {m.name.lower(): m for m in mask_files}
    mask_by_stem: Dict[str, List[Path]] = {}
    for mask_file in mask_files:
        key = _stem_without_extra_mask_tokens(mask_file)
        mask_by_stem.setdefault(key, []).append(mask_file)

    samples: List[SUIMSample] = []
    unmatched_images: List[Path] = []

    for image_file in image_files:
        candidate = None

        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            candidate = mask_by_name.get((image_file.stem + ext).lower())
            if candidate is not None:
                break

        if candidate is None:
            candidates = mask_by_stem.get(_stem_without_extra_mask_tokens(image_file), [])
            if len(candidates) == 1:
                candidate = candidates[0]

        if candidate is None:
            img_key = _stem_without_extra_mask_tokens(image_file)
            fuzzy = [
                m for key, values in mask_by_stem.items()
                if key.startswith(img_key) or img_key.startswith(key)
                for m in values
            ]
            if len(fuzzy) == 1:
                candidate = fuzzy[0]

        if candidate is None:
            unmatched_images.append(image_file)
            continue

        samples.append(SUIMSample(image_path=image_file, mask_path=candidate))

    if unmatched_images:
        preview = ", ".join(p.name for p in unmatched_images[:10])
        raise RuntimeError(
            f"Could not match {len(unmatched_images)} image(s) to masks in {images_dir} / {masks_dir}. "
            f"Examples: {preview}"
        )

    if not samples:
        raise RuntimeError(f"No image-mask pairs found in: {images_dir} and {masks_dir}")

    return samples


def rgb_mask_to_class(
    mask: np.ndarray,
    color_map: Dict[Tuple[int, int, int], int] = SUIM_COLOR_MAP,
) -> np.ndarray:
    if mask.ndim != 3 or mask.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB mask, got shape: {mask.shape}")

    label_mask = np.full(mask.shape[:2], fill_value=255, dtype=np.uint8)

    for color, class_idx in color_map.items():
        matches = np.all(mask == np.array(color, dtype=np.uint8), axis=-1)
        label_mask[matches] = class_idx

    if np.any(label_mask == 255):
        unknown = np.unique(mask[label_mask == 255].reshape(-1, 3), axis=0)
        raise ValueError(
            "Found unknown RGB values in segmentation mask. "
            f"Examples: {unknown[:10].tolist()}. "
            "If your dataset uses a different palette, override color_map."
        )

    return label_mask


def class_to_rgb_mask(class_mask: np.ndarray) -> np.ndarray:
    if class_mask.ndim != 2:
        raise ValueError(f"Expected HxW mask, got shape: {class_mask.shape}")

    rgb = np.zeros((*class_mask.shape, 3), dtype=np.uint8)
    for idx, color in SUIM_INDEX_TO_COLOR.items():
        rgb[class_mask == idx] = color
    return rgb


# ============================================================
# Albumentations transforms
# ============================================================
def get_train_transform(img_size: int = 256):
    """
    Returns an albumentations transform for training.

    Requires:
        pip install albumentations opencv-python
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError as exc:
        raise ImportError(
            "Albumentations is required for get_train_transform(). "
            "Install it with: pip install albumentations opencv-python"
        ) from exc

    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform(img_size: int = 256):
    """
    Returns an albumentations transform for validation/test.

    Requires:
        pip install albumentations opencv-python
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError as exc:
        raise ImportError(
            "Albumentations is required for get_val_transform(). "
            "Install it with: pip install albumentations opencv-python"
        ) from exc

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ============================================================
# Dataset
# ============================================================
class SUIMDataset(Dataset):
    """
    PyTorch dataset for the SUIM semantic segmentation dataset.

    Expected folder structure:
        root/
            train_val/
                images/
                masks/
            TEST/
                images/
                masks/
            Benchmark_Evaluation/

    Parameters
    ----------
    root:
        Dataset root path.
    split:
        One of: 'train_val', 'train', 'val', 'test'.
    val_ratio:
        Validation split ratio used when split is 'train' or 'val'.
    seed:
        Random seed for deterministic train/val split.
    image_transform:
        Optional transform applied only to the image.
    mask_transform:
        Optional transform applied only to the mask.
    joint_transform:
        Transform applied jointly to image and mask. Best choice for segmentation.
        Expected signature for albumentations: transform(image=image_np, mask=mask_np)
    color_map:
        RGB tuple to class-index mapping.
    return_paths:
        If True, also returns image_path and mask_path.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train_val",
        val_ratio: float = 0.2,
        seed: int = 42,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        joint_transform: Optional[Callable] = None,
        color_map: Optional[Dict[Tuple[int, int, int], int]] = None,
        return_paths: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split.lower()
        self.val_ratio = val_ratio
        self.seed = seed
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.joint_transform = joint_transform
        self.color_map = color_map or SUIM_COLOR_MAP
        self.return_paths = return_paths

        if self.split not in {"train_val", "train", "val", "test"}:
            raise ValueError("split must be one of: 'train_val', 'train', 'val', 'test'")

        if self.split in {"train_val", "train", "val"}:
            images_dir = self.root / "train_val" / "images"
            masks_dir = self.root / "train_val" / "masks"
            all_samples = _pair_images_and_masks(images_dir, masks_dir)

            if self.split == "train_val":
                self.samples = all_samples
            else:
                train_samples, val_samples = self._make_train_val_split(all_samples)
                self.samples = train_samples if self.split == "train" else val_samples
        else:
            images_dir = self.root / "TEST" / "images"
            masks_dir = self.root / "TEST" / "masks"
            self.samples = _pair_images_and_masks(images_dir, masks_dir)

    def _make_train_val_split(
        self, samples: Sequence[SUIMSample]
    ) -> Tuple[List[SUIMSample], List[SUIMSample]]:
        n_total = len(samples)
        n_val = int(round(n_total * self.val_ratio))
        n_val = max(1, n_val)
        n_train = n_total - n_val

        if n_train <= 0:
            raise ValueError("val_ratio too large: training split would be empty")

        generator = torch.Generator().manual_seed(self.seed)
        train_subset, val_subset = random_split(samples, [n_train, n_val], generator=generator)

        train_samples = [samples[i] for i in train_subset.indices]
        val_samples = [samples[i] for i in val_subset.indices]
        return train_samples, val_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image = Image.open(sample.image_path).convert("RGB")
        rgb_mask = Image.open(sample.mask_path).convert("RGB")

        image_np = np.array(image, dtype=np.uint8)
        rgb_mask_np = np.array(rgb_mask, dtype=np.uint8)
        class_mask = rgb_mask_to_class(rgb_mask_np, self.color_map)

        if self.joint_transform is not None:
            transformed = self.joint_transform(image=image_np, mask=class_mask)
            image_out = transformed["image"] if isinstance(transformed, dict) else transformed[0]
            mask_out = transformed["mask"] if isinstance(transformed, dict) else transformed[1]
        else:
            image_out = image_np
            mask_out = class_mask

            if self.image_transform is not None:
                image_out = self.image_transform(image_out)
            if self.mask_transform is not None:
                mask_out = self.mask_transform(mask_out)

        if isinstance(image_out, np.ndarray):
            if image_out.ndim == 3 and image_out.shape[-1] == 3:
                image_out = torch.from_numpy(image_out).permute(2, 0, 1).float() / 255.0
            else:
                image_out = torch.from_numpy(image_out)

        if isinstance(mask_out, np.ndarray):
            mask_out = torch.from_numpy(mask_out).long()

        image_out = image_out.float()
        mask_out = mask_out.long()

        output = {
            "image": image_out,
            "mask": mask_out,
        }

        if self.return_paths:
            output["image_path"] = str(sample.image_path)
            output["mask_path"] = str(sample.mask_path)

        return output


# ============================================================
# Dataloader builders
# ============================================================
def create_suim_dataloaders(
    root: Union[str, Path],
    batch_size: int = 8,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
    drop_last: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if train_transform is None:
        train_transform = get_train_transform()
    if val_transform is None:
        val_transform = get_val_transform()
    if test_transform is None:
        test_transform = get_val_transform()

    train_dataset = SUIMDataset(
        root=root,
        split="train",
        val_ratio=val_ratio,
        seed=seed,
        joint_transform=train_transform,
    )
    val_dataset = SUIMDataset(
        root=root,
        split="val",
        val_ratio=val_ratio,
        seed=seed,
        joint_transform=val_transform,
    )
    test_dataset = SUIMDataset(
        root=root,
        split="test",
        joint_transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    root = os.environ.get("SUIM_ROOT", "./SUIM")
    print(f"Trying to read SUIM dataset from: {root}")

    if Path(root).exists():
        train_loader, val_loader, test_loader = create_suim_dataloaders(
            root=root,
            batch_size=4,
            val_ratio=0.2,
            num_workers=0,
        )

        first_batch = next(iter(train_loader))
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        print("Image batch shape:", tuple(first_batch["image"].shape))
        print("Mask batch shape:", tuple(first_batch["mask"].shape))
        print("Mask dtype:", first_batch["mask"].dtype)
        print("Unique labels in first mask:", torch.unique(first_batch["mask"][0]).tolist())
    else:
        print("Dataset root does not exist. Set SUIM_ROOT or call download_suim() first.")
