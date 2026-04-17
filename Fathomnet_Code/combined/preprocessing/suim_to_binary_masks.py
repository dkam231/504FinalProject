import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from combined.datasets.suim_binary_dataset import suim_rgb_mask_to_binary
from combined.utils.io import list_image_files

def _convert_directory(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_files = list_image_files(input_dir)

    for mask_path in tqdm(mask_files, desc=f"Converting {input_dir.name}", leave=False):
        rgb_mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
        binary = suim_rgb_mask_to_binary(rgb_mask) * 255
        Image.fromarray(binary.astype(np.uint8)).save(output_dir / mask_path.name)

def convert_suim_masks(input_root, output_root) -> None:
    input_root = Path(input_root)
    output_root = Path(output_root)

    mappings = [(input_root / "train_val" / "masks", output_root / "train_val" / "masks")]
    if (input_root / "TEST" / "masks").exists():
        mappings.append((input_root / "TEST" / "masks", output_root / "TEST" / "masks"))

    for src, dst in mappings:
        if src.exists():
            _convert_directory(src, dst)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert SUIM RGB masks to binary FG/BG masks.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    convert_suim_masks(args.input_root, args.output_root)

if __name__ == "__main__":
    main()
