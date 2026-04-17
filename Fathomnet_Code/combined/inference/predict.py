import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from combined.datasets.transforms import get_val_transform
from combined.models.unet import UNet
from combined.utils.checkpointing import load_checkpoint

@torch.no_grad()
def predict_mask(model, image_path, device, img_size=256, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image, dtype=np.uint8)
    transform = get_val_transform(img_size)
    transformed = transform(image=image_np, mask=np.zeros(image_np.shape[:2], dtype=np.uint8))
    tensor = transformed["image"].unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()[0, 0].cpu().numpy().astype(np.uint8)
    return pred

def parse_args():
    parser = argparse.ArgumentParser(description="Run binary segmentation inference on a single image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu"
    )

    model = UNet(n_channels=3, n_classes=1).to(device)
    load_checkpoint(args.checkpoint, model=model, device=device)
    model.eval()

    pred = predict_mask(model, args.image_path, device, args.img_size, args.threshold)
    Image.fromarray((pred * 255).astype(np.uint8)).save(args.output_path)
    print(f"Saved prediction to {args.output_path}")

if __name__ == "__main__":
    main()
