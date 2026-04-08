from pathlib import Path

import torch

from dataloader import SUIM_CLASSES, create_suim_dataloaders


def main():
    project_dir = Path(__file__).resolve().parent
    data_root = project_dir / "SUIM"

    if not data_root.exists():
        raise FileNotFoundError(
            f"SUIM dataset root not found: {data_root}\n"
            "Create this folder and place the dataset inside it first."
        )

    train_loader, val_loader, test_loader = create_suim_dataloaders(
        root=data_root,
        batch_size=4,
        val_ratio=0.2,
        seed=42,
        num_workers=0,
        pin_memory=False,
    )

    first_batch = next(iter(train_loader))
    print(f"Dataset root: {data_root}")
    print(f"Classes: {SUIM_CLASSES}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Image batch shape: {tuple(first_batch['image'].shape)}")
    print(f"Mask batch shape: {tuple(first_batch['mask'].shape)}")
    print(f"Image dtype: {first_batch['image'].dtype}")
    print(f"Mask dtype: {first_batch['mask'].dtype}")
    print(f"Unique labels in first batch: {torch.unique(first_batch['mask']).tolist()}")


if __name__ == "__main__":
    main()
