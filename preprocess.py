import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from coco_lib import COCO


class FathomNetDataset(Dataset):
    """
    Dataset class for FathomNet COCO-format data.

    Loads images and creates FG/BG masks and category masks from annotations.
    """

    def __init__(self, json_path, img_dir, transform=None, num_categories=2):
        """
        Args:
            json_path: Path to COCO JSON annotation file
            img_dir: Directory containing images
            transform: Optional transform to apply to images
            num_categories: Number of categories (including background)
        """
        self.coco = COCO(json_path)
        self.img_dir = img_dir
        self.transform = transform
        self.num_categories = num_categories

        # Get image IDs that have annotations
        self.img_ids = []
        for img_id in self.coco.imgs.keys():
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                self.img_ids.append(img_id)

        print(f"Loaded {len(self.img_ids)} images with annotations")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Create masks
        height, width = img_info['height'], img_info['width']
        fg_mask = np.zeros((height, width), dtype=np.uint8)
        cat_mask = np.zeros((height, width), dtype=np.int64)

        for ann in anns:
            if 'segmentation' in ann and ann['segmentation']:
                # Convert segmentation to mask
                mask = self.coco.annToMask(ann)
                fg_mask = np.maximum(fg_mask, mask)  # Binary FG mask
                cat_id = ann['category_id']
                cat_mask[mask > 0] = cat_id

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        fg_mask = torch.from_numpy(fg_mask).unsqueeze(0).float()
        cat_mask = torch.from_numpy(cat_mask).long()

        if self.transform:
            image = self.transform(image)
            # Note: transforms for masks might need separate handling

        return {
            'image': image,
            'fg_mask': fg_mask,
            'cat_mask': cat_mask,
            'img_id': img_id
        }


def create_data_loaders(train_json, test_json, img_dir, batch_size=4, num_workers=2):
    """
    Create train and test data loaders.

    Args:
        train_json: Path to train JSON
        test_json: Path to test JSON
        img_dir: Image directory
        batch_size: Batch size
        num_workers: Number of workers for data loading

    Returns:
        train_loader, test_loader
    """
    from torch.utils.data import DataLoader

    train_dataset = FathomNetDataset(train_json, img_dir)
    test_dataset = FathomNetDataset(test_json, img_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    train_json = "train_dataset.json"
    test_json = "test_dataset.json"
    img_dir = "images"  # Assuming images are downloaded here

    if os.path.exists(train_json):
        dataset = FathomNetDataset(train_json, img_dir)
        print(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print("Sample keys:", sample.keys())
            print("Image shape:", sample['image'].shape)
            print("FG mask shape:", sample['fg_mask'].shape)
            print("Cat mask shape:", sample['cat_mask'].shape)
    else:
        print("Train JSON not found. Please ensure data is downloaded.")
