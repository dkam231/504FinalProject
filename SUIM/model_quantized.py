import copy
from pathlib import Path

import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx


DEFAULT_QUANTIZED_MODEL_NAME = "best_unet_suim_binary_quantized.ts"


def get_quantization_backend():
    supported = torch.backends.quantized.supported_engines
    if "x86" in supported:
        return "x86"
    if "fbgemm" in supported:
        return "fbgemm"
    if "qnnpack" in supported:
        return "qnnpack"
    raise RuntimeError(f"No supported quantization backend found. Available engines: {supported}")


@torch.no_grad()
def calibrate_model(prepared_model, calibration_loader, num_batches):
    prepared_model.eval()
    for batch_idx, batch in enumerate(calibration_loader):
        if num_batches is not None and batch_idx >= num_batches:
            break
        images = batch["image"].cpu()
        prepared_model(images)


def quantize_unet_post_training(model, calibration_loader, example_input, num_calibration_batches=16):
    backend = get_quantization_backend()
    torch.backends.quantized.engine = backend

    model_to_quantize = copy.deepcopy(model).cpu().eval()
    qconfig_mapping = get_default_qconfig_mapping(backend)
    prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, (example_input.cpu(),))
    calibrate_model(prepared_model, calibration_loader, num_calibration_batches)
    quantized_model = convert_fx(prepared_model)
    return quantized_model


def save_quantized_model(quantized_model, example_input, output_path):
    output_path = Path(output_path)
    traced_model = torch.jit.trace(quantized_model, example_input.cpu())
    traced_model.save(str(output_path))
    return output_path


def load_quantized_model(model_path):
    return torch.jit.load(str(model_path), map_location="cpu").eval()
