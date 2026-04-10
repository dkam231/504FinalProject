import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from combined.datasets.fathomnet_binary_dataset import normalize_binary_mask
from combined.utils.io import list_image_files

def _convert_directory(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_files = list_image_files(input_dir, recursive=True)

    for mask_path in tqdm(mask_files, desc=f"Converting {input_dir.name}", leave=False):
        mask_np = np.array(Image.open(mask_path))
        binary = normalize_binary_mask(mask_np) * 255
        relative = mask_path.relative_to(input_dir)
        (output_dir / relative.parent).mkdir(parents=True, exist_ok=True)
        Image.fromarray(binary.astype(np.uint8)).save(output_dir / relative)

def convert_fathomnet_masks(input_root, output_root) -> None:
    input_root = Path(input_root)
    output_root = Path(output_root)

    handled = False
    for split in ["train", "val", "test"]:
        src = input_root / split / "masks"
        dst = output_root / split / "masks"
        if src.exists():
            _convert_directory(src, dst)
            handled = True

    flat_masks = input_root / "masks"
    if flat_masks.exists():
        _convert_directory(flat_masks, output_root / "masks")
        handled = True

    if not handled:
        raise FileNotFoundError(f"No mask directories found under {input_root}")

def parse_args():
    parser = argparse.ArgumentParser(description="Normalize FathomNet masks into binary FG/BG masks.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    convert_fathomnet_masks(args.input_root, args.output_root)

if __name__ == "__main__":
    main()
