# Run

## Train U-Net on SUIM

From the project root:

```bash
# From the project root
cd ./SUIM
python train.py
# After train completed
python test.py
```

The training script:

- loads SUIM data from `SUIM_Dataset/SUIM`
- builds train and validation splits from `train_val`
- trains an 8-class U-Net
- evaluates pixel accuracy and mean IoU
- saves the best checkpoint to `SUIM_Dataset/checkpoints/best_unet_suim.pth`

### Optional: use a custom dataset path

If your dataset is stored somewhere else, set `SUIM_ROOT`.

### Windows PowerShell

```powershell
$env:SUIM_ROOT="C:\path\to\SUIM"
python train.py
```

### macOS / Linux / WSL

```bash
export SUIM_ROOT="/path/to/SUIM"
python train.py
```

## Train Baseline Models on SUIM
ResNet-101 + Detectron2 could be executed by running the command below:
```bash
# From the project root
cd ./SUIM/baseline_models/detectron
python train.py
# After train completed
python test.py
```

ResNet-50 + DeepLabV3 could be executed by running the command below:
```bash
# From the project root
cd ./SUIM/baseline_models/deeplab
python train.py
# After train completed
python test.py
```

## Common issues

### Dataset root not found

Make sure this exists:

```text
SUIM/data/train_val/images
SUIM/data/train_val/masks
SUIM/data/TEST/images
SUIM/data/TEST/masks
```

### Unknown RGB values in masks

This means some segmentation masks do not exactly match the expected SUIM palette. Check whether:

- the masks are true RGB label masks
- the files were altered by image compression or editing
- the dataset variant uses a different color mapping

### Slow or unstable training on Windows

If you hit Windows dataloader issues, start with:

- `num_workers=0`
- a smaller `batch_size`

Then increase settings once the pipeline is stable.
