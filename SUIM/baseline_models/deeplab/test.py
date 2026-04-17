import os
import cv2
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

import albumentations as A
from albumentations.pytorch import ToTensorV2


NUM_CLASSES = 2


def get_test_transform(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class TestImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.image_files = sorted([
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ])

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {img_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        if self.transform is not None:
            aug = self.transform(image=image_rgb)
            image = aug["image"]
        else:
            image = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0

        return {
            "image": image,
            "image_bgr": image_bgr,
            "name": img_path.name,
            "orig_h": orig_h,
            "orig_w": orig_w,
        }


def test_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    image_bgrs = [item["image_bgr"] for item in batch]
    names = [item["name"] for item in batch]
    orig_hs = [item["orig_h"] for item in batch]
    orig_ws = [item["orig_w"] for item in batch]

    return {
        "image": images,
        "image_bgr": image_bgrs,
        "name": names,
        "orig_h": orig_hs,
        "orig_w": orig_ws,
    }


def build_deeplab_model():
    model = models.segmentation.deeplabv3_resnet50(
        weights=DeepLabV3_ResNet50_Weights.DEFAULT
    )
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier = None
    return model


def decode_binary_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_mask[mask == 1] = [255, 255, 255]
    return color_mask


@torch.no_grad()
def run_inference(model, loader, device, output_dir):
    model.eval()

    color_dir = output_dir / "color_masks"
    raw_dir = output_dir / "raw_masks"
    overlay_dir = output_dir / "overlays"

    color_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="Testing"):
        images = batch["image"].to(device)
        outputs = model(images)["out"]
        preds = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)

        names = batch["name"]
        image_bgrs = batch["image_bgr"]
        orig_hs = batch["orig_h"]
        orig_ws = batch["orig_w"]

        for i in range(len(names)):
            pred = preds[i]
            name = names[i]
            stem = Path(name).stem

            image_bgr = image_bgrs[i]
            orig_h = int(orig_hs[i])
            orig_w = int(orig_ws[i])

            pred_resized = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            color_mask = decode_binary_mask(pred_resized)
            raw_mask = (pred_resized * 255).astype(np.uint8)
            overlay = cv2.addWeighted(image_bgr, 0.6, color_mask, 0.4, 0)

            cv2.imwrite(str(raw_dir / f"{stem}_raw.png"), raw_mask)
            cv2.imwrite(str(color_dir / f"{stem}_color.png"), color_mask)
            cv2.imwrite(str(overlay_dir / f"{stem}_overlay.png"), overlay)

            unique_pred = np.unique(pred_resized).tolist()
            fg_ratio = float((pred_resized == 1).mean())
            print(f"{name}: unique_pred={unique_pred}, fg_ratio={fg_ratio:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test DeepLabV3 on SUIM_Dataset.")
    parser.add_argument("--weights", default="checkpoints/best_deeplab_suim_binary.pth", help="Path to trained model weights.")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_base = os.path.abspath(os.path.join(script_dir, "../../data"))
    test_img_dir = os.path.join(dataset_base, "TEST", "images")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    num_workers = (0 if os.name == "nt" else 4) if args.num_workers < 0 else args.num_workers

    dataset = TestImageDataset(
        image_dir=test_img_dir,
        transform=get_test_transform(args.img_size),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=test_collate_fn,
    )

    model = build_deeplab_model().to(device)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)

    output_dir = Path(script_dir) / "deeplab_test_results"
    run_inference(model, loader, device, output_dir)
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()