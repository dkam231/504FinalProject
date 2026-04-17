import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO


def build_split(json_path: Path, img_dir: Path, out_dir: Path, link_images: bool = False):
    coco = COCO(str(json_path))
    image_ids = list(coco.imgs.keys())

    images_out = out_dir / "images"
    masks_out = out_dir / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    paired = 0
    missing = 0
    manifest = []

    for img_id in image_ids:
        info = coco.loadImgs([img_id])[0]
        file_name = info["file_name"]
        src_img = img_dir / file_name

        if not src_img.exists():
            missing += 1
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
        for ann in anns:
            seg = ann.get("segmentation")
            if seg:
                mask = np.maximum(mask, coco.annToMask(ann).astype(np.uint8))

        mask_name = f"{Path(file_name).stem}.png"
        dst_img = images_out / file_name
        dst_mask = masks_out / mask_name

        if link_images:
            if dst_img.exists():
                dst_img.unlink()
            dst_img.symlink_to(src_img.resolve())
        else:
            shutil.copy2(src_img, dst_img)

        cv2.imwrite(str(dst_mask), mask * 255)

        manifest.append(
            {
                "image_id": img_id,
                "image": str(dst_img.relative_to(out_dir)),
                "mask": str(dst_mask.relative_to(out_dir)),
            }
        )
        paired += 1

    with open(out_dir / "pairs_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "json_source": str(json_path),
                "image_dir_source": str(img_dir),
                "paired_count": paired,
                "missing_images_count": missing,
                "pairs": manifest,
            },
            f,
            indent=2,
        )

    return paired, missing


def main():
    parser = argparse.ArgumentParser(
        description="Build image-mask pair repo from FathomNet segmentation COCO JSON."
    )
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--link-images", action="store_true")
    args = parser.parse_args()

    paired, missing = build_split(
        json_path=Path(args.json_path),
        img_dir=Path(args.img_dir),
        out_dir=Path(args.out_dir),
        link_images=args.link_images,
    )
    print(f"Paired images: {paired}")
    print(f"Missing images: {missing}")
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
