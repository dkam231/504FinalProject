import os
import cv2
import torch
import argparse
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

SUIM_CLASSES = ["BW", "HD", "PF", "WR", "RO", "RI", "FV", "SR"]


def run_inference(num_test=None, use_train=False):
    cfg = get_cfg()

    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        )
    )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(SUIM_CLASSES)
    cfg.MODEL.WEIGHTS = os.path.join("./output", "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_base = os.path.abspath(os.path.join(script_dir, "../../data"))

    train_img_dir = os.path.join(dataset_base, "train_val", "images")
    test_img_dir = os.path.join(dataset_base, "TEST", "images")
    train_mask_dir = os.path.join(dataset_base, "train_val", "masks")
    test_mask_dir = os.path.join(dataset_base, "TEST", "masks")

    img_dir = train_img_dir if use_train else test_img_dir
    mask_dir = train_mask_dir if use_train else test_mask_dir

    print(f"Script dir:      {script_dir}")
    print(f"Dataset base:    {dataset_base}")
    print(f"Using img dir:   {img_dir}")
    print(f"Using mask dir:  {mask_dir}")

    metadata_name = "suim_inference"
    MetadataCatalog.get(metadata_name).set(thing_classes=SUIM_CLASSES)
    metadata = MetadataCatalog.get(metadata_name)

    predictor = DefaultPredictor(cfg)

    os.makedirs("results", exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_files = sorted(
        [
            f for f in os.listdir(img_dir)
            if os.path.isfile(os.path.join(img_dir, f))
            and os.path.splitext(f)[1].lower() in valid_exts
        ]
    )

    if len(img_files) == 0:
        print(f"No images found in {img_dir}")
        return

    if num_test is not None:
        img_files = img_files[:num_test]

    print(f"Running inference on {len(img_files)} images from {img_dir}")

    for img_file in tqdm(img_files, desc="Running inference", unit="img"):
        img_path = os.path.join(img_dir, img_file)
        img_id = os.path.splitext(img_file)[0]

        im = cv2.imread(img_path)
        if im is None:
            print(f"Warning: failed to read image {img_path}")
            continue

        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")

        pred_classes = instances.pred_classes.tolist() if instances.has("pred_classes") else []
        print(f"{img_file}: detected {len(pred_classes)} instances, classes={pred_classes}")

        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
        out = v.draw_instance_predictions(instances)

        save_path = os.path.join("results", f"result_{img_id}.png")
        cv2.imwrite(save_path, out.get_image()[:, :, ::-1])

    print("✅ All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_test", "-n",
        type=int,
        default=None,
        help="Run inference on the first num_test images only"
    )
    parser.add_argument(
        "--use_train",
        action="store_true",
        help="Use SUIM train_val/images instead of TEST/images"
    )
    args = parser.parse_args()

    run_inference(
        num_test=args.num_test,
        use_train=args.use_train
    )