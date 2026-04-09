import re
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_image_files(folder: Path, recursive: bool = False):
    folder = Path(folder)
    if not folder.exists():
        return []
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        [p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )

def _normalize_stem(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"(_mask|_masks|_label|_labels|_gt|_seg|_annotation)$", "", stem)
    return stem

def pair_image_mask_files(images_dir: Path, masks_dir: Path, recursive: bool = False):
    image_files = list_image_files(images_dir, recursive=recursive)
    mask_files = list_image_files(masks_dir, recursive=recursive)

    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")
    if not mask_files:
        raise FileNotFoundError(f"No masks found in {masks_dir}")

    mask_by_name = {m.name.lower(): m for m in mask_files}
    mask_by_stem = {}
    for mask_file in mask_files:
        mask_by_stem.setdefault(_normalize_stem(mask_file), []).append(mask_file)

    pairs = []
    unmatched = []

    for image_file in image_files:
        candidate = None

        for ext in IMAGE_EXTENSIONS:
            mask_name = (image_file.stem + ext).lower()
            if mask_name in mask_by_name:
                candidate = mask_by_name[mask_name]
                break

        if candidate is None:
            candidates = mask_by_stem.get(_normalize_stem(image_file), [])
            if len(candidates) == 1:
                candidate = candidates[0]

        if candidate is None:
            img_key = _normalize_stem(image_file)
            fuzzy = [
                m
                for key, values in mask_by_stem.items()
                if key.startswith(img_key) or img_key.startswith(key)
                for m in values
            ]
            if len(fuzzy) == 1:
                candidate = fuzzy[0]

        if candidate is None:
            unmatched.append(image_file)
            continue

        pairs.append((image_file, candidate))

    if unmatched:
        preview = ", ".join(p.name for p in unmatched[:10])
        raise RuntimeError(
            f"Could not match {len(unmatched)} image(s) to masks. Examples: {preview}"
        )

    return pairs
