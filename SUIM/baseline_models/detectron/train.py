import os
import cv2
import csv
import argparse
import numpy as np
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, hooks
from detectron2.utils.logger import setup_logger
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

setup_logger()

SUIM_COLOR_MAP = {
    (0, 0, 0): 0,         # BW
    (0, 0, 255): 1,       # HD
    (0, 255, 0): 2,       # PF
    (255, 0, 0): 3,       # WR
    (255, 255, 0): 4,     # RO
    (255, 0, 255): 5,     # RI
    (0, 255, 255): 6,     # FV
    (255, 255, 255): 7,   # SR
}

SUIM_CLASSES = ["BW", "HD", "PF", "WR", "RO", "RI", "FV", "SR"]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_mask_as_class_ids(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")

    if len(mask.shape) == 2:
        return mask.astype(np.uint8)

    if len(mask.shape) == 3 and mask.shape[2] == 4:
        mask = mask[:, :, :3]

    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        class_mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)

        matched = np.zeros(mask_rgb.shape[:2], dtype=bool)
        for color, class_id in SUIM_COLOR_MAP.items():
            color_arr = np.array(color, dtype=np.uint8)
            region = np.all(mask_rgb == color_arr, axis=-1)
            class_mask[region] = class_id
            matched |= region

        if not np.all(matched):
            unknown = np.size(matched) - np.count_nonzero(matched)
            print(f"[WARN] {mask_path}: {unknown} pixels unmatched, assigned to class 0")

        return class_mask

    raise ValueError(f"Unsupported mask shape: {mask.shape}")


def find_mask_for_image(mask_dir, image_name):
    stem = os.path.splitext(image_name)[0]
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        candidate = os.path.join(mask_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def binary_mask_to_polygons(binary_mask):
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        contour = contour.flatten().tolist()
        if len(contour) >= 6:
            polygons.append(contour)
    return polygons


def get_suim_detectron_dicts(img_dir, mask_dir, limit=None, min_area=20):
    print("\n" + "=" * 60)
    print(f"Image dir: {img_dir}")
    print(f"Mask dir:  {mask_dir}")
    print(f"Limit:     {limit if limit else 'all'}")
    print("=" * 60)

    img_files = sorted([
        f for f in os.listdir(img_dir)
        if os.path.isfile(os.path.join(img_dir, f))
        and os.path.splitext(f)[1].lower() in VALID_EXTS
    ])

    dataset_dicts = []
    missing_masks = 0
    empty_annotations = 0

    for idx, img_file in enumerate(img_files):
        if limit is not None and len(dataset_dicts) >= limit:
            break

        img_path = os.path.join(img_dir, img_file)
        mask_path = find_mask_for_image(mask_dir, img_file)

        if mask_path is None:
            missing_masks += 1
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] failed to read image: {img_path}")
            continue

        h, w = image.shape[:2]
        class_mask = load_mask_as_class_ids(mask_path)

        record = {
            "file_name": img_path,
            "image_id": idx,
            "height": h,
            "width": w,
        }

        objs = []

        for class_id in range(len(SUIM_CLASSES)):
            binary = (class_mask == class_id).astype(np.uint8)
            if binary.sum() == 0:
                continue

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

            for comp_id in range(1, num_labels):
                x, y, bw, bh, area = stats[comp_id]
                if area < min_area:
                    continue

                comp_mask = (labels == comp_id).astype(np.uint8)
                polygons = binary_mask_to_polygons(comp_mask)
                if len(polygons) == 0:
                    continue

                objs.append({
                    "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": class_id,
                    "segmentation": polygons,
                    "iscrowd": 0,
                    "area": float(area),
                })

        if len(objs) == 0:
            empty_annotations += 1
            continue

        record["annotations"] = objs
        dataset_dicts.append(record)

    print("\nDataset summary")
    print(f"Valid records:        {len(dataset_dicts)}")
    print(f"Missing masks:        {missing_masks}")
    print(f"Empty annotations:    {empty_annotations}")
    print(f"Num classes:          {len(SUIM_CLASSES)}")
    print("=" * 60 + "\n")

    return dataset_dicts


def instances_to_semantic_mask(instances, height, width):
    pred_mask = np.zeros((height, width), dtype=np.uint8)

    if len(instances) == 0:
        return pred_mask

    instances = instances.to("cpu")
    masks = instances.pred_masks.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()

    order = np.argsort(scores)  # low -> high, high score overwrite later
    for idx in order:
        mask = masks[idx].astype(bool)
        cls_id = int(classes[idx])
        pred_mask[mask] = cls_id

    return pred_mask


def update_confusion_matrix(conf_mat, gt, pred, num_classes):
    valid = (gt >= 0) & (gt < num_classes)
    gt = gt[valid]
    pred = pred[valid]
    inds = num_classes * gt.astype(np.int64) + pred.astype(np.int64)
    conf_mat += np.bincount(inds, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return conf_mat


def compute_miou_from_confmat(conf_mat):
    ious = []
    per_class = {}

    for cls_id, cls_name in enumerate(SUIM_CLASSES):
        tp = conf_mat[cls_id, cls_id]
        fp = conf_mat[:, cls_id].sum() - tp
        fn = conf_mat[cls_id, :].sum() - tp
        denom = tp + fp + fn
        iou = float("nan") if denom == 0 else tp / denom
        per_class[cls_name] = iou
        if not np.isnan(iou):
            ious.append(iou)

    miou = float(np.mean(ious)) if ious else float("nan")
    return miou, per_class


class MIoUHook(hooks.HookBase):
    def __init__(self, eval_period, model, dataset_dicts, output_dir, num_classes):
        self.eval_period = eval_period
        self.model = model
        self.dataset_dicts = dataset_dicts
        self.output_dir = output_dir
        self.num_classes = num_classes
        self.csv_path = os.path.join(output_dir, "miou_results.csv")

        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["iter", "mIoU"] + [f"IoU_{c}" for c in SUIM_CLASSES])

    def _do_eval(self):
        self.model.eval()
        conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        with torch.no_grad():
            for record in self.dataset_dicts:
                img = cv2.imread(record["file_name"])
                if img is None:
                    continue

                gt_mask = load_mask_as_class_ids(find_mask_for_image(
                    os.path.dirname(record["file_name"]).replace("images", "masks"),
                    os.path.basename(record["file_name"])
                ))

                h, w = gt_mask.shape[:2]
                if img.shape[:2] != gt_mask.shape[:2]:
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

                inputs = [{"image": torch.as_tensor(img.transpose(2, 0, 1).astype("float32")).to(self.model.device),
                           "height": h,
                           "width": w}]
                outputs = self.model(inputs)[0]

                if "instances" in outputs:
                    pred_mask = instances_to_semantic_mask(outputs["instances"], h, w)
                else:
                    pred_mask = np.zeros((h, w), dtype=np.uint8)

                conf_mat = update_confusion_matrix(conf_mat, gt_mask, pred_mask, self.num_classes)

        miou, per_class = compute_miou_from_confmat(conf_mat)
        cur_iter = self.trainer.iter + 1

        print("\n" + "=" * 60)
        print(f"[mIoU EVAL] iter={cur_iter}  mIoU={miou:.4f}")
        for cls_name, iou in per_class.items():
            if np.isnan(iou):
                print(f"{cls_name}: nan")
            else:
                print(f"{cls_name}: {iou:.4f}")
        print("=" * 60)

        self.trainer.storage.put_scalar("test_mIoU", miou, smoothing_hint=False)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            row = [cur_iter, miou] + [per_class[c] for c in SUIM_CLASSES]
            writer.writerow(row)

        self.model.train()

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if self.eval_period > 0 and (is_final or next_iter % self.eval_period == 0):
            self._do_eval()


class Trainer(DefaultTrainer):
    def __init__(self, cfg, test_dicts):
        self.test_dicts = test_dicts
        super().__init__(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks_list = super().build_hooks()

        miou_hook = MIoUHook(
            eval_period=self.cfg.TEST.EVAL_PERIOD,
            model=self.model,
            dataset_dicts=self.test_dicts,
            output_dir=self.cfg.OUTPUT_DIR,
            num_classes=len(SUIM_CLASSES),
        )

        hooks_list.insert(-1, miou_hook)
        return hooks_list


def train_suim(num_train=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_base = os.path.abspath(os.path.join(script_dir, "../../data"))

    train_img_dir = os.path.join(dataset_base, "train_val", "images")
    test_img_dir = os.path.join(dataset_base, "TEST", "images")
    train_mask_dir = os.path.join(dataset_base, "train_val", "masks")
    test_mask_dir = os.path.join(dataset_base, "TEST", "masks")

    print(f"Script dir:      {script_dir}")
    print(f"Dataset base:    {dataset_base}")
    print(f"Train img dir:   {train_img_dir}")
    print(f"Test img dir:    {test_img_dir}")
    print(f"Train mask dir:  {train_mask_dir}")
    print(f"Test mask dir:   {test_mask_dir}")

    train_dicts = get_suim_detectron_dicts(train_img_dir, train_mask_dir, limit=num_train)
    test_dicts = get_suim_detectron_dicts(test_img_dir, test_mask_dir, limit=None)

    if len(train_dicts) == 0:
        raise ValueError("No valid training records found.")
    if len(test_dicts) == 0:
        raise ValueError("No valid test records found.")

    for name in ["suim_train", "suim_test"]:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)

    DatasetCatalog.register("suim_train", lambda: train_dicts)
    MetadataCatalog.get("suim_train").set(thing_classes=SUIM_CLASSES)

    DatasetCatalog.register("suim_test", lambda: test_dicts)
    MetadataCatalog.get("suim_test").set(thing_classes=SUIM_CLASSES)

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    )

    cfg.DATASETS.TRAIN = ("suim_train",)
    cfg.DATASETS.TEST = ("suim_test",)

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    )

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = (6000, 8000)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(SUIM_CLASSES)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.DEVICE = "cuda"
    cfg.INPUT.MASK_FORMAT = "polygon"

    # every 1000 iter evaluate AP and mIoU
    cfg.TEST.EVAL_PERIOD = 1000

    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print(f"Start training with {len(train_dicts)} valid images...")
    trainer = Trainer(cfg, test_dicts=test_dicts)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Detectron2 Mask R-CNN on SUIM")
    parser.add_argument(
        "--num_train", "-n",
        type=int,
        default=None,
        help="number of valid training images to use"
    )
    args = parser.parse_args()
    train_suim(num_train=args.num_train)