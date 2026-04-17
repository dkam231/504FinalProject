import os
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from dataloader import SUIM_BINARY_CLASSES, binary_mask_to_rgb, create_suim_dataloaders
from model_quantized import DEFAULT_QUANTIZED_MODEL_NAME, load_quantized_model
from utils import BCEDiceFocalLoss, binary_iou, pixel_accuracy


def get_test_transform(img_size=(320, 240)):
    return A.Compose([
        A.Resize(img_size[1], img_size[0]),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])


def denormalize_image(image_tensor):
    return image_tensor.cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def logits_to_binary_predictions(logits):
    probs = torch.sigmoid(logits)
    return (probs >= 0.5).long()


def save_visualizations(model, loader, output_dir, max_images=6):
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].cpu()
            masks = batch["mask"]

            logits = model(images)
            preds = logits_to_binary_predictions(logits).cpu()

            for sample_idx in range(images.size(0)):
                if saved >= max_images:
                    return

                image_np = denormalize_image(images[sample_idx])
                gt_mask_np = masks[sample_idx].cpu().numpy().astype("uint8")
                pred_mask_np = preds[sample_idx, 0].numpy().astype("uint8")

                gt_rgb = binary_mask_to_rgb(gt_mask_np)
                pred_rgb = binary_mask_to_rgb(pred_mask_np)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(image_np)
                axes[0].set_title("Input Image")
                axes[1].imshow(gt_rgb)
                axes[1].set_title("Ground Truth")
                axes[2].imshow(pred_rgb)
                axes[2].set_title("Quantized Prediction")

                for axis in axes:
                    axis.axis("off")

                fig.tight_layout()
                save_path = output_dir / f"quantized_test_vis_{saved:03d}.png"
                fig.savefig(save_path, bbox_inches="tight")
                plt.close(fig)
                saved += 1


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    running_iou = 0.0

    progress_bar = tqdm(loader, desc="Testing Quantized Model", leave=False)
    for step, batch in enumerate(progress_bar, start=1):
        images = batch["image"].cpu()
        masks = batch["mask"].cpu().unsqueeze(1).float()

        logits = model(images)
        loss = criterion(logits, masks)

        preds = logits_to_binary_predictions(logits)
        batch_acc = pixel_accuracy(preds.squeeze(1), masks.squeeze(1).long())
        batch_iou = binary_iou(preds.squeeze(1), masks.squeeze(1).long())

        running_loss += loss.item()
        running_acc += batch_acc
        running_iou += batch_iou

        progress_bar.set_postfix(
            loss=f"{running_loss / step:.4f}",
            acc=f"{running_acc / step:.4f}",
            iou=f"{running_iou / step:.4f}",
        )

    n = len(loader)
    return running_loss / n, running_acc / n, running_iou / n


def main():
    project_dir = Path(__file__).resolve().parent
    checkpoints_dir = project_dir / "checkpoints"
    data_root = Path(os.environ.get("SUIM_ROOT", project_dir / "data"))
    quantized_model_path = Path(
        os.environ.get("SUIM_QUANTIZED_MODEL", checkpoints_dir / DEFAULT_QUANTIZED_MODEL_NAME)
    )
    vis_dir = Path(
        os.environ.get("SUIM_QUANT_VIS_DIR", project_dir / "test_visualizations_quantized")
    )
    max_vis = int(os.environ.get("SUIM_MAX_VIS", 6))
    batch_size = int(os.environ.get("SUIM_BATCH_SIZE", 8))
    img_width = int(os.environ.get("SUIM_IMG_WIDTH", 320))
    img_height = int(os.environ.get("SUIM_IMG_HEIGHT", 240))
    img_size = (img_width, img_height)
    val_ratio = float(os.environ.get("SUIM_VAL_RATIO", 0.2))
    seed = int(os.environ.get("SUIM_SEED", 42))
    num_workers = 0 if os.name == "nt" else int(os.environ.get("SUIM_NUM_WORKERS", 1))

    if not data_root.exists():
        raise FileNotFoundError(f"SUIM dataset root not found: {data_root}")
    if not quantized_model_path.exists():
        raise FileNotFoundError(f"Quantized model not found: {quantized_model_path}")

    _, _, test_loader = create_suim_dataloaders(
        root=data_root,
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
        num_workers=num_workers,
        pin_memory=False,
        test_transform=get_test_transform(img_size),
    )

    quantized_model = load_quantized_model(quantized_model_path)
    criterion = BCEDiceFocalLoss()

    print(f"Dataset root: {data_root}")
    print(f"Quantized model: {quantized_model_path}")
    print("Device: cpu")
    print(f"Task: Binary foreground/background segmentation")
    print(f"Classes: {SUIM_BINARY_CLASSES}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Resize: {img_width}x{img_height}")

    test_loss, test_acc, test_iou = evaluate(quantized_model, test_loader, criterion)

    print("\nQuantized Test Results")
    print(f"Loss: {test_loss:.4f}")
    print(f"Pixel Accuracy: {test_acc:.4f}")
    print(f"IoU: {test_iou:.4f}")

    save_visualizations(quantized_model, test_loader, vis_dir, max_images=max_vis)
    print(f"Saved {max_vis} quantized test visualizations to: {vis_dir}")


if __name__ == "__main__":
    main()
