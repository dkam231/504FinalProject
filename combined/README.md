# Combined Workflow: SUIM → FathomNet FG/BG Segmentation

This folder contains the final, self-contained binary segmentation workflow for the project.

## Objective

Train a foreground/background segmentation model in two stages:

1. **Pretrain on SUIM** after collapsing multiclass labels into a binary task.
2. **Transfer to FathomNet** by loading the SUIM checkpoint and fine-tuning on FathomNet masks.

## Binary label convention

Across every stage of the workflow:

- `0` = background
- `1` = foreground

### SUIM mapping

Original SUIM semantic classes are mapped as follows:

- `BW` / `(0, 0, 0)` → background
- `SR` / `(255, 255, 255)` → background
- all remaining SUIM classes → foreground

That means:
- background waterbody + sand / sea-floor / rocks → background
- divers, plants, wrecks, robots, reefs/invertebrates, fish/vertebrates → foreground

## Recommended training sequence

### 1) Train on SUIM
```bash
python -m combined.training.train_suim \
  --data-root /path/to/SUIM \
  --epochs 30 \
  --batch-size 8 \
  --img-size 256
```

### 2) Fine-tune on FathomNet
```bash
python -m combined.training.train_fathomnet \
  --data-root /path/to/fathomnet_binary \
  --pretrained-checkpoint /path/to/suim_fg_bg_best.pth \
  --epochs 20 \
  --batch-size 8 \
  --img-size 256
```

## Expected dataset layouts

### SUIM
```text
SUIM/
├── train_val/
│   ├── images/
│   └── masks/
└── TEST/
    ├── images/
    └── masks/   # optional for evaluation
```

### FathomNet

Supported layouts:

**Split layout**
```text
fathomnet/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

**Flat layout**
```text
fathomnet/
├── images/
└── masks/
```

For the flat layout, the loader creates a deterministic train/val split.

## Folder contents

- `datasets/` — dataset classes and transforms
- `models/` — U-Net baseline
- `preprocessing/` — mask conversion and verification scripts
- `training/` — losses, metrics, loops, and training entry points
- `inference/` — prediction utilities
- `utils/` — pairing, checkpointing, and reproducibility helpers
- `DESIGN_DECISIONS.md` — professional design document
