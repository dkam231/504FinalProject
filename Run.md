# Run

## Train the multiclass U-Net on SUIM

From the project root:

```bash
cd SUIM_Dataset
python train.py
```

The training script:

- loads SUIM data from `SUIM_Dataset/SUIM`
- builds train and validation splits from `train_val`
- trains an 8-class U-Net
- evaluates pixel accuracy and mean IoU
- saves the best checkpoint to `SUIM_Dataset/checkpoints/best_unet_suim.pth`

## Optional: use a custom dataset path

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

## Recommended first run

Before a full training run:

1. Run `python check_dataset.py`
2. Confirm the sample counts and class labels look correct
3. Then launch `python train.py`

## Common issues

### Dataset root not found

Make sure this exists:

```text
SUIM_Dataset/SUIM/train_val/images
SUIM_Dataset/SUIM/train_val/masks
SUIM_Dataset/SUIM/TEST/images
SUIM_Dataset/SUIM/TEST/masks
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
