import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class FathomNetSegmentationDataset(Dataset):
    """
    PyTorch Dataset for FathomNet COCO-style segmentation.

    Returns image/mask pairs where:
    - image: float32 tensor in [0, 1], shape (3, H, W)
    - mask: uint8 tensor of 0/1, shape (H, W)
    """

    def __init__(
        self,
        json_path,
        img_dir,
        transform=None,
        mask_transform=None,
        mask_mode="auto",
        only_downloaded=True,
    ):
        """
        Args:
            json_path: Path to FathomNet COCO JSON (e.g., train.json, eval.json)
            img_dir: Directory containing downloaded images
            transform: Optional callable applied to image tensor
            mask_transform: Optional callable applied to mask tensor
            mask_mode: "auto", "segmentation", or "bbox"
            only_downloaded: if True, only keep samples where image file exists
        """
        if not os.path.isfile(json_path):
            raise FileNotFoundError(
                f"Annotation JSON not found: {json_path}\n"
                "Use a real path like ./train_dataset.json or ./test_dataset.json."
            )
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"Image directory not found: {img_dir}\n"
                "Download images first (for example with download.py) and pass that folder."
            )

        self.coco = COCO(json_path)
        self.mask_mode = mask_mode
        if self.mask_mode not in {"auto", "segmentation", "bbox"}:
            raise ValueError('mask_mode must be one of: "auto", "segmentation", "bbox"')

        self.img_dir = img_dir
        self.transform = transform
        self.mask_transform = mask_transform
        all_img_ids = list(self.coco.imgs.keys())

        if only_downloaded:
            filtered = []
            for img_id in all_img_ids:
                file_name = self.coco.loadImgs(img_id)[0]["file_name"]
                image_path = os.path.join(self.img_dir, file_name)
                if os.path.isfile(image_path):
                    filtered.append(img_id)
            self.img_ids = filtered
        else:
            self.img_ids = all_img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_metadata = self.coco.loadImgs(img_id)[0]
        file_name = img_metadata["file_name"]
        image_path = os.path.join(self.img_dir, file_name)

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros(
            (img_metadata["height"], img_metadata["width"]), dtype=np.uint8
        )
        seg_count = 0
        bbox_count = 0
        for ann in anns:
            pixel_mask, source = self._ann_to_mask(
                ann, img_metadata["height"], img_metadata["width"]
            )
            mask = np.maximum(mask, pixel_mask)
            if source == "segmentation":
                seg_count += 1
            elif source == "bbox":
                bbox_count += 1

        mask = torch.from_numpy(mask)

        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return {
            "file_name": file_name,
            "image": image,
            "mask": mask,
            "image_id": img_id,
            "seg_ann_count": seg_count,
            "bbox_ann_count": bbox_count,
        }

    def _ann_to_mask(self, ann, height, width):
        use_segmentation = self.mask_mode in {"auto", "segmentation"}
        use_bbox = self.mask_mode in {"auto", "bbox"}

        if use_segmentation:
            seg = ann.get("segmentation", None)
            if seg is not None and seg != []:
                try:
                    return self.coco.annToMask(ann).astype(np.uint8), "segmentation"
                except Exception:
                    if self.mask_mode == "segmentation":
                        raise

        if use_bbox and "bbox" in ann:
            x, y, w, h = ann["bbox"]
            x1 = max(0, int(np.floor(x)))
            y1 = max(0, int(np.floor(y)))
            x2 = min(width, int(np.ceil(x + w)))
            y2 = min(height, int(np.ceil(y + h)))
            out = np.zeros((height, width), dtype=np.uint8)
            if x2 > x1 and y2 > y1:
                out[y1:y2, x1:x2] = 1
            return out, "bbox"

        return np.zeros((height, width), dtype=np.uint8), "none"


def _run_preview(args):
    dataset = FathomNetSegmentationDataset(
        args.json_path,
        args.img_dir,
        mask_mode=args.mask_mode,
        only_downloaded=not args.include_missing,
    )
    print(f"Loaded {len(dataset)} images from {args.json_path}")

    if len(dataset) == 0:
        print("No images found in annotation JSON.")
        return

    sample = dataset[args.index]
    print(f"Preview index: {args.index}")
    print(f"File: {sample['file_name']}")
    print(f"Image tensor shape: {tuple(sample['image'].shape)}")
    print(f"Mask tensor shape: {tuple(sample['mask'].shape)}")
    print(f"Mask unique values: {torch.unique(sample['mask']).tolist()}")
    print(f"Annotations from segmentation: {sample['seg_ann_count']}")
    print(f"Annotations from bbox fallback: {sample['bbox_ann_count']}")

    if args.save_mask_path:
        mask_np = (sample["mask"].numpy() * 255).astype(np.uint8)
        cv2.imwrite(args.save_mask_path, mask_np)
        print(f"Saved preview mask to: {args.save_mask_path}")

    if args.save_comparison_path:
        _save_side_by_side(sample["image"], sample["mask"], args.save_comparison_path)
        print(f"Saved side-by-side comparison to: {args.save_comparison_path}")


def _export_all_masks(args):
    dataset = FathomNetSegmentationDataset(
        args.json_path,
        args.img_dir,
        mask_mode=args.mask_mode,
        only_downloaded=not args.include_missing,
    )

    out_dir = Path(args.export_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {len(dataset)} masks to {out_dir}")
    for i in range(len(dataset)):
        sample = dataset[i]
        mask_np = (sample["mask"].numpy() * 255).astype(np.uint8)
        out_path = out_dir / f"{Path(sample['file_name']).stem}.png"
        cv2.imwrite(str(out_path), mask_np)
    print(f"Done. Saved masks to {out_dir}")


def _save_side_by_side(image_tensor, mask_tensor, output_path):
    image = (image_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    mask = (mask_tensor.numpy() * 255).astype(np.uint8)

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = image.copy()
    overlay[mask > 0] = (0.6 * overlay[mask > 0] + 0.4 * np.array([255, 0, 0])).astype(
        np.uint8
    )

    divider = np.full((image.shape[0], 8, 3), 255, dtype=np.uint8)
    canvas = np.concatenate([image, divider, mask_rgb, divider, overlay], axis=1)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Create pixel-level masks from FathomNet COCO annotations."
    )
    parser.add_argument(
        "--json-path",
        required=True,
        help="Path to COCO JSON (e.g., train.json or eval.json).",
    )
    parser.add_argument(
        "--img-dir",
        required=True,
        help="Directory containing the downloaded FathomNet images.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Dataset index to preview.",
    )
    parser.add_argument(
        "--save-mask-path",
        default="",
        help="Optional path to save one preview mask PNG.",
    )
    parser.add_argument(
        "--save-comparison-path",
        default="",
        help="Optional path to save side-by-side image|mask|overlay visualization.",
    )
    parser.add_argument(
        "--mask-mode",
        default="auto",
        choices=["auto", "segmentation", "bbox"],
        help='Mask creation mode: "auto" uses segmentation first, then bbox fallback.',
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include entries even if image file is missing from img-dir.",
    )
    parser.add_argument(
        "--export-dir",
        default="",
        help="If set, export masks for all available images to this directory.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.export_dir:
        _export_all_masks(args)
    else:
        _run_preview(args)
