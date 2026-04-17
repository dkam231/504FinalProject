import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from combined.utils.io import list_image_files

def summarize_masks(mask_dir, recursive=True, limit=20):
    mask_dir = Path(mask_dir)
    mask_files = list_image_files(mask_dir, recursive=recursive)
    if not mask_files:
        raise FileNotFoundError(f"No masks found in {mask_dir}")

    global_counter = Counter()

    for mask_path in mask_files[:limit]:
        mask = np.array(Image.open(mask_path))
        unique_values = np.unique(mask)
        global_counter.update(unique_values.tolist())
        print(f"{mask_path.name}: unique values = {unique_values.tolist()}")

    print("\nAggregate unique values seen in preview:")
    print(dict(sorted(global_counter.items())))

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect mask values for sanity checking.")
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()

def main():
    args = parse_args()
    summarize_masks(args.mask_dir, limit=args.limit)

if __name__ == "__main__":
    main()
