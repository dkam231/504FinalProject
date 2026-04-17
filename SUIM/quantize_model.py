import json
import os
from pathlib import Path

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

from dataloader import SUIM_BINARY_CLASSES, create_suim_dataloaders
from model import UNet
from model_quantized import (
    DEFAULT_QUANTIZED_MODEL_NAME,
    quantize_unet_post_training,
    save_quantized_model,
)
from test import evaluate
from utils import BCEDiceFocalLoss


def get_eval_transform(img_size=(320, 240)):
    return A.Compose([
        A.Resize(img_size[1], img_size[0]),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])


def load_float_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return model


def file_size_mb(path):
    return Path(path).stat().st_size / (1024 * 1024)


def main():
    project_dir = Path(__file__).resolve().parent
    checkpoints_dir = project_dir / "checkpoints"
    data_root = Path(os.environ.get("SUIM_ROOT", project_dir / "data"))
    checkpoint_path = Path(
        os.environ.get("SUIM_CHECKPOINT", checkpoints_dir / "best_unet_suim_binary.pth")
    )
    quantized_output_path = Path(
        os.environ.get("SUIM_QUANTIZED_OUTPUT", checkpoints_dir / DEFAULT_QUANTIZED_MODEL_NAME)
    )
    report_path = Path(
        os.environ.get("SUIM_QUANT_REPORT", checkpoints_dir / "quantization_report.json")
    )
    batch_size = int(os.environ.get("SUIM_BATCH_SIZE", 8))
    img_width = int(os.environ.get("SUIM_IMG_WIDTH", 320))
    img_height = int(os.environ.get("SUIM_IMG_HEIGHT", 240))
    img_size = (img_width, img_height)
    val_ratio = float(os.environ.get("SUIM_VAL_RATIO", 0.2))
    seed = int(os.environ.get("SUIM_SEED", 42))
    num_workers = 0 if os.name == "nt" else int(os.environ.get("SUIM_NUM_WORKERS", 1))
    calibration_batches = int(os.environ.get("SUIM_QUANT_CALIB_BATCHES", 16))

    if not data_root.exists():
        raise FileNotFoundError(f"SUIM dataset root not found: {data_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Float checkpoint not found: {checkpoint_path}")

    eval_transform = get_eval_transform(img_size)
    train_loader, _, test_loader = create_suim_dataloaders(
        root=data_root,
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
        num_workers=num_workers,
        pin_memory=False,
        train_transform=eval_transform,
        val_transform=eval_transform,
        test_transform=eval_transform,
    )

    float_model = UNet(n_channels=3, n_classes=1).cpu().eval()
    float_model = load_float_checkpoint(float_model, checkpoint_path, "cpu")
    criterion = BCEDiceFocalLoss()

    example_input = next(iter(train_loader))["image"]
    quantized_model = quantize_unet_post_training(
        float_model,
        train_loader,
        example_input,
        num_calibration_batches=calibration_batches,
    )
    save_quantized_model(quantized_model, example_input, quantized_output_path)

    float_loss, float_acc, float_iou = evaluate(float_model, test_loader, criterion, "cpu")
    quant_loss, quant_acc, quant_iou = evaluate(quantized_model, test_loader, criterion, "cpu")

    report = {
        "task": "binary foreground/background segmentation",
        "classes": SUIM_BINARY_CLASSES,
        "float_checkpoint": str(checkpoint_path),
        "quantized_model": str(quantized_output_path),
        "calibration_batches": calibration_batches,
        "float_model_size_mb": round(file_size_mb(checkpoint_path), 4),
        "quantized_model_size_mb": round(file_size_mb(quantized_output_path), 4),
        "float_metrics": {
            "loss": float_loss,
            "pixel_accuracy": float_acc,
            "iou": float_iou,
        },
        "quantized_metrics": {
            "loss": quant_loss,
            "pixel_accuracy": quant_acc,
            "iou": quant_iou,
        },
    }

    report_path.write_text(json.dumps(report, indent=2))

    print(f"Float checkpoint: {checkpoint_path}")
    print(f"Quantized model saved to: {quantized_output_path}")
    print(f"Quantization report saved to: {report_path}")
    print(f"Float size (MB): {report['float_model_size_mb']}")
    print(f"Quantized size (MB): {report['quantized_model_size_mb']}")
    print(
        f"Float test metrics -> Loss: {float_loss:.4f}, Acc: {float_acc:.4f}, IoU: {float_iou:.4f}"
    )
    print(
        f"Quantized test metrics -> Loss: {quant_loss:.4f}, Acc: {quant_acc:.4f}, IoU: {quant_iou:.4f}"
    )


if __name__ == "__main__":
    main()
