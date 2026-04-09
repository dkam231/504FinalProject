from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from combined.datasets.transforms import get_train_transform, get_val_transform
from combined.utils.io import pair_image_mask_files

SUIM_COLOR_MAP = {
    (0, 0, 0): 0,         # BW
    (0, 0, 255): 1,       # HD
    (0, 255, 0): 2,       # PF
    (0, 255, 255): 3,     # WR
    (255, 0, 0): 4,       # RO
    (255, 0, 255): 5,     # RI
    (255, 255, 0): 6,     # FV
    (255, 255, 255): 7,   # SR
}
SUIM_FOREGROUND_CLASS_IDS = {1, 2, 3, 4, 5, 6}

@dataclass
class SUIMSample:
    image_path: Path
    mask_path: Path

def rgb_mask_to_class(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 3 or mask.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB mask, got shape {mask.shape}")

    label_mask = np.full(mask.shape[:2], fill_value=255, dtype=np.uint8)

    for color, class_idx in SUIM_COLOR_MAP.items():
        matches = np.all(mask == np.array(color, dtype=np.uint8), axis=-1)
        label_mask[matches] = class_idx

    if np.any(label_mask == 255):
        unknown_mask = label_mask == 255
        palette_colors = np.array(list(SUIM_COLOR_MAP.keys()), dtype=np.int16)
        palette_labels = np.array(list(SUIM_COLOR_MAP.values()), dtype=np.uint8)

        unknown_pixels = mask[unknown_mask].astype(np.int16)
        diff = unknown_pixels[:, None, :] - palette_colors[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        nearest_idx = np.argmin(dist2, axis=1)
        label_mask[unknown_mask] = palette_labels[nearest_idx]

    return label_mask

def suim_rgb_mask_to_binary(mask_rgb: np.ndarray) -> np.ndarray:
    multiclass = rgb_mask_to_class(mask_rgb)
    return np.isin(multiclass, list(SUIM_FOREGROUND_CLASS_IDS)).astype(np.uint8)

def _to_mask_tensor(mask_out):
    if isinstance(mask_out, np.ndarray):
        mask_out = torch.from_numpy(mask_out)
    if mask_out.ndim == 2:
        mask_out = mask_out.unsqueeze(0)
    return mask_out.float()

class SUIMBinaryDataset(Dataset):
    def __init__(self, root, split="train", val_ratio=0.2, seed=42, joint_transform=None, return_paths=False):
        super().__init__()
        self.root = Path(root)
        self.split = split.lower()
        self.val_ratio = val_ratio
        self.seed = seed
        self.joint_transform = joint_transform
        self.return_paths = return_paths

        if self.split not in {"train", "val", "train_val", "test"}:
            raise ValueError("split must be one of {'train', 'val', 'train_val', 'test'}")

        if self.split in {"train", "val", "train_val"}:
            pairs = pair_image_mask_files(self.root / "train_val" / "images", self.root / "train_val" / "masks")
            samples = [SUIMSample(img, mask) for img, mask in pairs]
            if self.split == "train_val":
                self.samples = samples
            else:
                self.samples = self._split_samples(samples)[0 if self.split == "train" else 1]
        else:
            pairs = pair_image_mask_files(self.root / "TEST" / "images", self.root / "TEST" / "masks")
            self.samples = [SUIMSample(img, mask) for img, mask in pairs]

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
        mask_rgb = Image.open(sample.mask_path).convert("RGB")

        image_np = np.array(image, dtype=np.uint8)
        binary_mask = suim_rgb_mask_to_binary(np.array(mask_rgb, dtype=np.uint8))

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

def create_suim_binary_dataloaders(root, batch_size=8, val_ratio=0.2, seed=42, num_workers=4, pin_memory=True, train_transform=None, val_transform=None):
    train_transform = train_transform or get_train_transform()
    val_transform = val_transform or get_val_transform()

    train_dataset = SUIMBinaryDataset(root=root, split="train", val_ratio=val_ratio, seed=seed, joint_transform=train_transform)
    val_dataset = SUIMBinaryDataset(root=root, split="val", val_ratio=val_ratio, seed=seed, joint_transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader
