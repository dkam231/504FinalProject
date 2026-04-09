from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dataloader import SUIM_COLOR_MAP, class_to_rgb_mask, rgb_mask_to_class


VALID_PALETTE = set(SUIM_COLOR_MAP.keys())


def build_overlay(image_np, mask_rgb, alpha=0.45):
    image_float = image_np.astype(np.float32) / 255.0
    mask_float = mask_rgb.astype(np.float32) / 255.0
    overlay = (1.0 - alpha) * image_float + alpha * mask_float
    return np.clip(overlay, 0.0, 1.0)


def find_bad_colors(mask_rgb_np):
    colors = np.unique(mask_rgb_np.reshape(-1, 3), axis=0)
    bad_colors = [
        tuple(map(int, color))
        for color in colors
        if tuple(map(int, color)) not in VALID_PALETTE
    ]
    return colors, bad_colors


def save_visualization(image_path, mask_path, output_dir):
    image = Image.open(image_path).convert("RGB")
    raw_mask = Image.open(mask_path).convert("RGB")

    image_np = np.array(image, dtype=np.uint8)
    raw_mask_np = np.array(raw_mask, dtype=np.uint8)

    if image.size != raw_mask.size:
        raw_mask = raw_mask.resize(image.size, resample=Image.NEAREST)
        raw_mask_np = np.array(raw_mask, dtype=np.uint8)

    class_mask = rgb_mask_to_class(raw_mask_np, SUIM_COLOR_MAP)
    clean_mask_rgb = class_to_rgb_mask(class_mask)
    overlay = build_overlay(image_np, clean_mask_rgb)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[1].imshow(raw_mask_np)
    axes[1].set_title("Raw Mask")
    axes[2].imshow(clean_mask_rgb)
    axes[2].set_title("Cleaned Mask")
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    save_path = output_dir / f"{image_path.stem}_non_palette_audit.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main():
    project_dir = Path(__file__).resolve().parent
    data_root = project_dir / "data"
    images_dir = data_root / "train_val" / "images"
    masks_dir = data_root / "train_val" / "masks"
    output_dir = project_dir / "non_palette_audit"
    output_dir.mkdir(exist_ok=True)

    problematic_count = 0
    saved_paths = []

    for mask_path in sorted(masks_dir.glob("*.bmp")):
        image_path = images_dir / f"{mask_path.stem}.jpg"
        if not image_path.exists():
            continue
        raw_mask_np = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
        print(np.array(Image.open(mask_path)) == raw_mask_np)
        colors, bad_colors = find_bad_colors(raw_mask_np)

        if bad_colors:
            problematic_count += 1
            print(
                f"Non-palette mask: {mask_path.name} | "
                f"unique_colors={len(colors)} | bad_colors={len(bad_colors)} | "
                f"examples={bad_colors[:10]}"  f" | first line: {colors[0]}"
            )
            save_path = save_visualization(image_path, mask_path, output_dir)
            saved_paths.append(save_path)

    print(f"\nTotal masks with non-palette colors: {problematic_count}")
    if saved_paths:
        print("Saved visualizations:")
        for path in saved_paths[:20]:
            print(path)
        if len(saved_paths) > 20:
            print(f"... and {len(saved_paths) - 20} more")
    else:
        print("No non-palette masks found.")


if __name__ == "__main__":
    main()
