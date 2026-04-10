from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from combined.datasets.transforms import get_train_transform, get_val_transform
from combined.utils.io import pair_image_mask_files

@dataclass
class FathomNetSample:
    image_path: Path
    mask_path: Path

def normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert a mask into {0, 1}.

    Supported inputs:
    - grayscale 0/1
    - grayscale 0/255
    - multi-valued grayscale where any non-zero pixel is foreground
    - RGB masks where any non-black pixel is foreground
    """
    if mask.ndim == 3:
        if mask.shape[2] == 1:
            mask = mask[..., 0]
        else:
            return np.any(mask > 0, axis=-1).astype(np.uint8)
    return (mask > 0).astype(np.uint8)

def _to_mask_tensor(mask_out):
    if isinstance(mask_out, np.ndarray):
        mask_out = torch.from_numpy(mask_out)
    if mask_out.ndim == 2:
        mask_out = mask_out.unsqueeze(0)
    return mask_out.float()

class FathomNetBinaryDataset(Dataset):
    def __init__(self, root, split="train", val_ratio=0.2, seed=42, joint_transform=None, return_paths=False, recursive=True):
        super().__init__()
        self.root = Path(root)
        self.split = split.lower()
        self.val_ratio = val_ratio
        self.seed = seed
        self.joint_transform = joint_transform
        self.return_paths = return_paths
        self.recursive = recursive

        if self.split not in {"train", "val", "test", "all"}:
            raise ValueError("split must be one of {'train', 'val', 'test', 'all'}")
        self.samples = self._build_samples()

    def _build_samples(self):
        split_images = self.root / self.split / "images"
        split_masks = self.root / self.split / "masks"

        if self.split in {"train", "val", "test"} and split_images.exists() and split_masks.exists():
            pairs = pair_image_mask_files(split_images, split_masks, recursive=self.recursive)
            return [FathomNetSample(img, mask) for img, mask in pairs]

        flat_images = self.root / "images"
        flat_masks = self.root / "masks"
        if not (flat_images.exists() and flat_masks.exists()):
            raise FileNotFoundError(
                f"Could not find supported FathomNet layout under {self.root}. "
                "Expected either split directories or flat images/masks directories."
            )

        pairs = pair_image_mask_files(flat_images, flat_masks, recursive=self.recursive)
        samples = [FathomNetSample(img, mask) for img, mask in pairs]
        if self.split == "all":
            return samples

        train_samples, val_samples = self._split_samples(samples)
        if self.split == "train":
            return train_samples
        if self.split == "val":
            return val_samples
        return val_samples

    def _split_samples(self, samples):
        n_total = len(samples)
        n_val = max(1, int(round(n_total * self.val_ratio)))
        n_train = n_total - n_val
        if n_train <= 0:
            raise ValueError("val_ratio is too large; training split would be empty")
        generator = torch.Generator().manual_seed(self.seed)
        train_subset, val_subset = random_split(samples, [n_train, n_val], generator=generator)
        train_samples = [samples[i] for i in train_subset.indices]
        val_samples = [samples[i] for i in val_subset.indices]
        return train_samples, val_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        raw_mask = Image.open(sample.mask_path)

        image_np = np.array(image, dtype=np.uint8)
        binary_mask = normalize_binary_mask(np.array(raw_mask))

        if self.joint_transform is not None:
            transformed = self.joint_transform(image=image_np, mask=binary_mask)
            image_out = transformed["image"]
            mask_out = transformed["mask"]
        else:
            image_out = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            mask_out = torch.from_numpy(binary_mask)

        if isinstance(image_out, np.ndarray):
            image_out = torch.from_numpy(image_out).permute(2, 0, 1).float() / 255.0

        output = {"image": image_out.float(), "mask": _to_mask_tensor(mask_out)}
        if self.return_paths:
            output["image_path"] = str(sample.image_path)
            output["mask_path"] = str(sample.mask_path)
        return output

def create_fathomnet_binary_dataloaders(root, batch_size=8, val_ratio=0.2, seed=42, num_workers=4, pin_memory=True, train_transform=None, val_transform=None):
    train_transform = train_transform or get_train_transform()
    val_transform = val_transform or get_val_transform()

    train_dataset = FathomNetBinaryDataset(root=root, split="train", val_ratio=val_ratio, seed=seed, joint_transform=train_transform)
    val_dataset = FathomNetBinaryDataset(root=root, split="val", val_ratio=val_ratio, seed=seed, joint_transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader
