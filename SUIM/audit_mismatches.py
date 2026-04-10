from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dataloader import SUIM_COLOR_MAP, _pair_images_and_masks, class_to_rgb_mask, rgb_mask_to_class


def build_overlay(image_np, mask_rgb, alpha=0.45):
    image_float = image_np.astype(np.float32) / 255.0
    mask_float = mask_rgb.astype(np.float32) / 255.0
    overlay = (1.0 - alpha) * image_float + alpha * mask_float
    return np.clip(overlay, 0.0, 1.0)


def crop_mask_to_image(image, raw_mask):
    if image.size == raw_mask.size:
        return raw_mask

    image_w, image_h = image.size
    mask_w, mask_h = raw_mask.size

    if mask_w != image_w:
        raise ValueError(
            f"Width mismatch is not handled by this audit script: image={image.size}, mask={raw_mask.size}"
        )

    if mask_h < image_h:
        raise ValueError(
            f"Mask height is smaller than image height, so bottom-cropping cannot fix it: "
            f"image={image.size}, mask={raw_mask.size}"
        )

    # Keep the top aligned and crop away extra rows from the bottom.
    return raw_mask.crop((0, 0, image_w, image_h))


def save_visualization(image_path, mask_path, output_dir):
    image = Image.open(image_path).convert("RGB")
    raw_mask = Image.open(mask_path).convert("RGB")

    image_np = np.array(image, dtype=np.uint8)
    raw_mask_np = np.array(raw_mask, dtype=np.uint8)

    cropped_mask = crop_mask_to_image(image, raw_mask)

    cropped_mask_np = np.array(cropped_mask, dtype=np.uint8)
    class_mask = rgb_mask_to_class(cropped_mask_np, SUIM_COLOR_MAP)
    clean_mask_rgb = class_to_rgb_mask(class_mask)
    overlay = build_overlay(image_np, clean_mask_rgb)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image_np)
    axes[0].set_title(f"Image\n{image_np.shape[1]}x{image_np.shape[0]}")
    axes[1].imshow(raw_mask_np)
    axes[1].set_title(f"Raw Mask\n{raw_mask_np.shape[1]}x{raw_mask_np.shape[0]}")
    axes[2].imshow(clean_mask_rgb)
    axes[2].set_title(f"Cropped/Cleaned Mask\n{clean_mask_rgb.shape[1]}x{clean_mask_rgb.shape[0]}")
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    save_path = output_dir / f"{image_path.stem}_mismatch_audit.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main():
    project_dir = Path(__file__).resolve().parent
    data_root = project_dir / "data"
    images_dir = data_root / "train_val" / "images"
    masks_dir = data_root / "train_val" / "masks"
    output_dir = project_dir / "mismatch_audit"
    output_dir.mkdir(exist_ok=True)

    samples = _pair_images_and_masks(images_dir, masks_dir)

    mismatch_count = 0
    saved_paths = []

    for sample in samples:
        image = Image.open(sample.image_path)
        mask = Image.open(sample.mask_path)
        if image.size != mask.size:
            mismatch_count += 1
            print(
                f"Mismatch: {sample.image_path.name} | "
                f"image={image.size} | mask={mask.size}"
            )
            save_path = save_visualization(sample.image_path, sample.mask_path, output_dir)
            saved_paths.append(save_path)

    print(f"\nTotal mismatched pairs: {mismatch_count}")
    if saved_paths:
        print("Saved visualizations:")
        for path in saved_paths[:20]:
            print(path)
        if len(saved_paths) > 20:
            print(f"... and {len(saved_paths) - 20} more")
    else:
        print("No size-mismatched pairs found.")


if __name__ == "__main__":
    main()
