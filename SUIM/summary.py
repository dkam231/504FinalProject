import json
from pathlib import Path

import torch
from torchinfo import summary

from SUIM.model import UNet
from SUIM.model_quantized import DEFAULT_QUANTIZED_MODEL_NAME, load_quantized_model


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = PROJECT_DIR / "SUIM" / "checkpoints"
FLOAT_CHECKPOINT_PATH = CHECKPOINTS_DIR / "best_unet_suim_binary.pth"
QUANTIZED_MODEL_PATH = CHECKPOINTS_DIR / DEFAULT_QUANTIZED_MODEL_NAME
QUANT_REPORT_PATH = CHECKPOINTS_DIR / "quantization_report.json"
INPUT_SIZE = (1, 3, 240, 320)


def file_size_mb(path):
    return path.stat().st_size / (1024 * 1024)


def print_float_model_summary():
    model = UNet(n_channels=3, n_classes=1)
    print("Float Model Summary")
    print(summary(model, input_size=INPUT_SIZE, verbose=0))

    if FLOAT_CHECKPOINT_PATH.exists():
        print(f"Float checkpoint: {FLOAT_CHECKPOINT_PATH}")
        print(f"Float checkpoint size: {file_size_mb(FLOAT_CHECKPOINT_PATH):.4f} MB")
    else:
        print(f"Float checkpoint not found: {FLOAT_CHECKPOINT_PATH}")


def print_quantized_model_summary():
    print("\nQuantized Model Summary")

    if not QUANTIZED_MODEL_PATH.exists():
        print(f"Quantized model not found: {QUANTIZED_MODEL_PATH}")
        return

    quantized_model = load_quantized_model(QUANTIZED_MODEL_PATH)
    example_input = torch.randn(INPUT_SIZE)
    quantized_output = quantized_model(example_input)

    print(f"Quantized model: {QUANTIZED_MODEL_PATH}")
    print(f"Quantized model size: {file_size_mb(QUANTIZED_MODEL_PATH):.4f} MB")
    print(f"Quantized output shape: {tuple(quantized_output.shape)}")
    try:
        print(summary(quantized_model, input_size=INPUT_SIZE, verbose=0))
    except Exception as exc:
        print("torchinfo summary for the quantized TorchScript model was not available.")
        print(f"Reason: {type(exc).__name__}: {exc}")
        print(quantized_model)

    if QUANT_REPORT_PATH.exists():
        report = json.loads(QUANT_REPORT_PATH.read_text())
        print("\nQuantization Report")
        print(json.dumps(report, indent=2))
    else:
        print(f"Quantization report not found: {QUANT_REPORT_PATH}")


def main():
    print_float_model_summary()
    print_quantized_model_summary()


if __name__ == "__main__":
    main()
