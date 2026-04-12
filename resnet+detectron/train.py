import os
import argparse
import cv2
import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2.structures import BoxMode

from pycocotools.coco import COCO

setup_logger()

def get_fathom_coco_dicts(json_file, img_dir, mask_dir=None, limit=None, require_nonempty_mask=False):
    print("\n" + "=" * 60)
    print(f"Loading COCO annotations from: {json_file}")
    print(f"Image directory: {img_dir}")
    print(f"Mask directory:  {mask_dir if mask_dir else 'None'}")
    print(f"Requested limit: {limit if limit else 'all'}")
    print("=" * 60)

    coco_api = COCO(json_file)

    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    thing_classes = [c["name"] for c in cats]
    cat_id_map = {cat_id: i for i, cat_id in enumerate(cat_ids)}

    img_ids = list(coco_api.getImgIds())

    dataset_dicts = []
    missing_file_count = 0
    missing_mask_count = 0
    empty_mask_count = 0
    unreadable_mask_count = 0
    no_ann_count = 0
    bad_bbox_count = 0

    for img_id in img_ids:
        try:
            img_info = coco_api.imgs[img_id]
        except KeyError:
            try:
                img_info = coco_api.loadImgs([img_id])[0]
            except Exception:
                print(f"[WARN] Failed to load image info for image id: {img_id}")
                continue

        file_name = img_info.get("file_name", "")
        full_img_path = os.path.join(img_dir, file_name)

        if not os.path.exists(full_img_path):
            missing_file_count += 1
            if missing_file_count <= 20:
                print(f"[WARN] Missing image file: {full_img_path}")
            continue

        if mask_dir is not None:
            stem = os.path.splitext(file_name)[0]
            full_mask_path = os.path.join(mask_dir, stem + ".png")

            if not os.path.exists(full_mask_path):
                missing_mask_count += 1
                if missing_mask_count <= 20:
                    print(f"[WARN] Missing mask file: {full_mask_path}")
                continue

            if require_nonempty_mask:
                mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    unreadable_mask_count += 1
                    if unreadable_mask_count <= 20:
                        print(f"[WARN] Unreadable mask file: {full_mask_path}")
                    continue

                if np.max(mask) == 0:
                    empty_mask_count += 1
                    if empty_mask_count <= 20:
                        print(f"[WARN] Empty mask file: {full_mask_path}")
                    continue

        ann_ids = coco_api.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco_api.loadAnns(ann_ids)

        if len(anns) == 0:
            no_ann_count += 1
            continue

        record = {
            "file_name": full_img_path,
            "image_id": img_id,
            "height": img_info["height"],
            "width": img_info["width"],
        }

        objs = []
        for ann in anns:
            if ann.get("ignore", 0) == 1:
                continue

            bbox = ann.get("bbox", None)
            if bbox is None or len(bbox) != 4:
                bad_bbox_count += 1
                continue

            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                bad_bbox_count += 1
                continue

            seg = ann.get("segmentation", None)
            if seg is None:
                continue

            cat_id = ann["category_id"]
            if cat_id not in cat_id_map:
                continue

            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": cat_id_map[cat_id],
                "iscrowd": ann.get("iscrowd", 0),
                "segmentation": seg,
            }

            if "area" in ann:
                obj["area"] = ann["area"]

            objs.append(obj)

        if len(objs) == 0:
            continue

        record["annotations"] = objs
        dataset_dicts.append(record)

        if limit is not None and limit > 0 and len(dataset_dicts) >= limit:
            break

    print("\n" + "=" * 60)
    print("Dataset summary")
    print(f"Total valid loaded records: {len(dataset_dicts)}")
    print(f"Missing image files:        {missing_file_count}")
    print(f"Missing mask files:         {missing_mask_count}")
    print(f"Unreadable mask files:      {unreadable_mask_count}")
    print(f"Empty mask files:           {empty_mask_count}")
    print(f"Images with no anns:        {no_ann_count}")
    print(f"Bad/invalid bboxes:         {bad_bbox_count}")
    print(f"Num categories:             {len(cat_ids)}")
    print("=" * 60 + "\n")

    return dataset_dicts, thing_classes


def train_fathomnet(num_train=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_base = os.path.abspath(os.path.join(script_dir, ".."))

    train_json = os.path.join(dataset_base, "fathomnet_segmentations", "train.json")
    test_json = os.path.join(dataset_base, "fathomnet_segmentations", "test.json")

    train_img_dir = os.path.join(dataset_base, "images_seg", "train")
    test_img_dir = os.path.join(dataset_base, "images_seg", "test")

    train_mask_dir = os.path.join(dataset_base, "masks_seg", "train")
    test_mask_dir = os.path.join(dataset_base, "masks_seg", "test")

    print(f"Script dir:   {script_dir}")
    print(f"Dataset base: {dataset_base}")
    print(f"Train json:   {train_json}")
    print(f"Test json:    {test_json}")
    print(f"Train imgs:   {train_img_dir}")
    print(f"Test imgs:    {test_img_dir}")
    print(f"Train masks:  {train_mask_dir}")
    print(f"Test masks:   {test_mask_dir}")

    train_dicts, thing_classes = get_fathom_coco_dicts(
        train_json,
        train_img_dir,
        mask_dir=train_mask_dir,
        limit=num_train,
        require_nonempty_mask=False,
    )

    test_dicts, _ = get_fathom_coco_dicts(
        test_json,
        test_img_dir,
        mask_dir=test_mask_dir,
        limit=None,
        require_nonempty_mask=True,
    )

    if len(train_dicts) == 0:
        raise ValueError("No valid training records found after skipping missing images/masks.")

    if "fathom_train" in DatasetCatalog.list():
        DatasetCatalog.remove("fathom_train")
    if "fathom_test" in DatasetCatalog.list():
        DatasetCatalog.remove("fathom_test")

    DatasetCatalog.register("fathom_train", lambda: train_dicts)
    MetadataCatalog.get("fathom_train").set(thing_classes=thing_classes)

    DatasetCatalog.register("fathom_test", lambda: test_dicts)
    MetadataCatalog.get("fathom_test").set(thing_classes=thing_classes)

    print(f"[CHECK] fathom_train loaded records: {len(train_dicts)}")
    print(f"[CHECK] fathom_test loaded records:  {len(test_dicts)}")
    print(f"[CHECK] NUM_CLASSES = {len(thing_classes)}")

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    )

    cfg.DATASETS.TRAIN = ("fathom_train",)
    cfg.DATASETS.TEST = ("fathom_test",)

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    )

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = (6000, 8000)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.DEVICE = "cuda"
    cfg.INPUT.MASK_FORMAT = "bitmask"

    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print(f"Start training with {len(train_dicts)} valid images...")
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Detectron2 on FathomNet COCO JSON")
    parser.add_argument(
        "--num_train", "-n",
        type=int,
        default=None,
        help="number of valid training images to use; missing files will be skipped until this many valid images are collected"
    )
    args = parser.parse_args()

    train_fathomnet(num_train=args.num_train)