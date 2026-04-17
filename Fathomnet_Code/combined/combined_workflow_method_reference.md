# Combined Workflow Code Reference
**Method-by-method explanation for `combined_workflow_v1/`**

## Purpose

This document explains what each file, class, and function does inside the `combined_workflow_v1` workflow package.

The package is organized into six main parts:

- `datasets/` — loads SUIM and FathomNet image-mask pairs
- `models/` — defines the U-Net architecture
- `preprocessing/` — converts masks into clean binary foreground/background masks
- `training/` — loss functions, metrics, training loop, and training entry points
- `inference/` — single-image prediction code
- `utils/` — helper utilities for checkpointing, seeds, and file pairing

The entire package is designed around one convention:

- **0 = background**
- **1 = foreground**

---

# 1. Top-level files

## `__init__.py`
This file marks the folder as a Python package.  
It does not contain logic itself.

## `README.md`
This is the user-facing overview of the workflow:
- what the package is for
- how SUIM and FathomNet connect
- how to run training and inference

## `DESIGN_DECISIONS.md`
This is the design document:
- why the binary segmentation formulation was chosen
- why SUIM is used before FathomNet
- how the folder is structured
- what assumptions the code makes

---

# 2. `datasets/`

## `datasets/__init__.py`
This file re-exports the main dataset entry points so they can be imported more easily.

It exposes:
- `SUIMBinaryDataset`
- `create_suim_binary_dataloaders`
- `FathomNetBinaryDataset`
- `create_fathomnet_binary_dataloaders`

It does not perform any processing itself.

---

## `datasets/transforms.py`

This file defines the image+mask transforms used during training and validation.

### `get_train_transform(img_size=256)`
Creates the augmentation pipeline for training.

What it does:
- resizes image and mask to a fixed square size
- applies random horizontal flip
- applies random vertical flip
- applies random 90-degree rotation
- applies color jitter to the image
- normalizes image channels
- converts image and mask into tensors

Why it exists:
- all training images need a consistent size
- data augmentation improves generalization
- the same transform must be applied jointly to image and mask so they stay aligned

### `get_val_transform(img_size=256)`
Creates the transform pipeline for validation or inference.

What it does:
- resizes image and mask
- normalizes image channels
- converts to tensors

Why it is separate:
- validation should be deterministic
- no random augmentation should be used during evaluation

---

## `datasets/suim_binary_dataset.py`

This file is the SUIM-specific dataset loader for the binary foreground/background task.

### Constants

#### `SUIM_COLOR_MAP`
Maps each official SUIM RGB mask color to its multiclass label id.

Meaning:
- `(0,0,0)` = BW
- `(255,255,255)` = SR
- all other listed colors are the foreground classes

This is the base mapping used before reducing the problem to binary FG/BG.

#### `SUIM_FOREGROUND_CLASS_IDS`
Defines which multiclass ids count as foreground.

In this workflow:
- `{1,2,3,4,5,6}` = foreground
- class `0` and class `7` are background

This implements your rule:
- `BW` and `SR` → background
- everything else → foreground

### `SUIMSample`
A tiny dataclass with two fields:
- `image_path`
- `mask_path`

Why it exists:
- it keeps paired image-mask metadata in a clean object
- it makes the dataset code easier to read

### `rgb_mask_to_class(mask)`
Converts a SUIM RGB mask into a multiclass integer mask.

Input:
- RGB segmentation mask from SUIM

Output:
- 2D label mask with class ids from `0` to `7`

Important behavior:
- it first tries exact RGB matching
- if some pixels do not match exactly, it assigns each unmatched pixel to the nearest official SUIM color

Why that fallback matters:
- some saved masks may have slight color artifacts
- nearest-color matching makes the loader more robust

### `suim_rgb_mask_to_binary(mask_rgb)`
Converts an RGB SUIM mask directly into a binary FG/BG mask.

What it does:
1. calls `rgb_mask_to_class`
2. checks whether each class id is in `SUIM_FOREGROUND_CLASS_IDS`
3. returns a binary mask with values `{0,1}`

This is the key label-conversion function for the SUIM stage.

### `_to_mask_tensor(mask_out)`
Converts a mask into the tensor format expected by training.

What it does:
- converts numpy arrays into PyTorch tensors
- adds a channel dimension if needed
- casts the result to `float`

Why it exists:
- the model outputs shape `(B,1,H,W)`
- the binary mask should match that shape during loss computation

### `SUIMBinaryDataset.__init__(...)`
Initializes the SUIM dataset.

What it handles:
- dataset root path
- which split to load: `train`, `val`, `train_val`, or `test`
- train/validation split ratio
- random seed
- joint transform
- whether file paths should be returned

How it works:
- for `train`, `val`, or `train_val`, it reads from `train_val/images` and `train_val/masks`
- for `test`, it reads from `TEST/images` and `TEST/masks`
- it pairs images and masks using `pair_image_mask_files`

### `SUIMBinaryDataset._split_samples(samples)`
Splits a full list of paired samples into train and validation subsets.

What it does:
- computes validation size from `val_ratio`
- uses `torch.random_split`
- uses a fixed seed for reproducibility

Why it exists:
- SUIM provides a train/validation folder, but this workflow still needs a deterministic internal split

### `SUIMBinaryDataset.__len__()`
Returns the number of samples in the selected split.

### `SUIMBinaryDataset.__getitem__(index)`
Loads one training example.

What it does:
1. opens the image in RGB
2. opens the RGB segmentation mask
3. converts the SUIM mask into binary FG/BG
4. applies the joint transform if provided
5. converts image and mask into tensors
6. returns a dictionary with:
   - `image`
   - `mask`
   - optionally `image_path` and `mask_path`

This is the main data loading method used during training.

### `create_suim_binary_dataloaders(...)`
Builds the SUIM training and validation dataloaders.

What it does:
- creates default train and validation transforms if none are provided
- creates `SUIMBinaryDataset` for both train and val
- wraps them in PyTorch `DataLoader`s

Why it exists:
- it gives one clean entry point for `train_suim.py`

---

## `datasets/fathomnet_binary_dataset.py`

This file is the FathomNet dataset loader for binary foreground/background segmentation.

### `FathomNetSample`
A simple dataclass with:
- `image_path`
- `mask_path`

Same purpose as `SUIMSample`: clean paired sample storage.

### `normalize_binary_mask(mask)`
Normalizes a FathomNet mask into binary form.

Supported cases:
- grayscale mask with `0/1`
- grayscale mask with `0/255`
- grayscale mask with arbitrary nonzero values
- RGB mask where any non-black pixel is treated as foreground

Output:
- a 2D mask with values `{0,1}`

Why it matters:
- FathomNet masks may not be stored in one single format
- this function forces them into the same binary convention as SUIM

### `_to_mask_tensor(mask_out)`
Same purpose as in the SUIM file:
- converts the mask into a float tensor
- adds a channel dimension when needed

### `FathomNetBinaryDataset.__init__(...)`
Initializes the FathomNet dataset.

What it handles:
- root path
- split name
- validation ratio
- seed
- transforms
- optional path return
- recursive search toggle

Supported layouts:
1. split layout:
   - `root/train/images`, `root/train/masks`
   - `root/val/images`, `root/val/masks`
   - `root/test/images`, `root/test/masks`
2. flat layout:
   - `root/images`
   - `root/masks`

This makes the loader flexible.

### `FathomNetBinaryDataset._build_samples()`
Builds the list of `FathomNetSample` objects.

What it does:
- checks whether split-specific directories exist
- otherwise falls back to a flat images/masks layout
- pairs files using `pair_image_mask_files`
- if needed, creates train/val subsets from one flat dataset

This is the core path-resolution function for FathomNet.

### `FathomNetBinaryDataset._split_samples(samples)`
Performs deterministic train/validation splitting using `random_split`.

Same role as in the SUIM dataset file.

### `FathomNetBinaryDataset.__len__()`
Returns the number of samples.

### `FathomNetBinaryDataset.__getitem__(index)`
Loads one FathomNet example.

What it does:
1. opens the RGB image
2. opens the raw mask
3. normalizes the mask to binary `{0,1}`
4. applies transforms
5. converts outputs into tensors
6. returns the same dictionary format as the SUIM loader

Why this is important:
- it keeps FathomNet training fully compatible with the same model and same loss code used for SUIM

### `create_fathomnet_binary_dataloaders(...)`
Creates train and validation dataloaders for FathomNet.

Same role as the SUIM dataloader builder, but for the FathomNet dataset.

---

# 3. `models/`

## `models/__init__.py`
Marks the folder as a package. No logic inside.

## `models/unet.py`

This file defines the segmentation model.

### `DoubleConv`
A U-Net building block with two convolution layers.

#### `DoubleConv.__init__(in_channels, out_channels)`
Creates:
- conv
- batch norm
- ReLU
- conv
- batch norm
- ReLU

Why this block exists:
- it is the standard “feature extraction” block used repeatedly in U-Net

#### `DoubleConv.forward(x)`
Applies the two-convolution block to the input tensor.

### `Down`
A downsampling block in the encoder.

#### `Down.__init__(in_channels, out_channels)`
Creates:
- max pooling
- followed by a `DoubleConv`

#### `Down.forward(x)`
Runs one encoder downsampling step.

Purpose:
- reduce spatial size
- increase feature depth

### `Up`
An upsampling block in the decoder.

#### `Up.__init__(in_channels, out_channels, bilinear=True)`
Creates either:
- bilinear upsampling + `DoubleConv`
or
- transposed convolution + `DoubleConv`

Current default:
- bilinear upsampling

#### `Up.forward(x1, x2)`
Takes:
- `x1` = decoder feature map
- `x2` = encoder skip connection

What it does:
- upsamples `x1`
- pads it if needed so the shapes match
- concatenates with `x2`
- applies `DoubleConv`

This is the core U-Net skip-connection logic.

### `OutConv`
The final output projection layer.

#### `OutConv.__init__(in_channels, out_channels)`
Creates a `1x1` convolution.

#### `OutConv.forward(x)`
Maps features to the final prediction channels.

For this workflow:
- `out_channels = 1`

### `UNet`
The full encoder-decoder network.

#### `UNet.__init__(n_channels=3, n_classes=1, bilinear=True)`
Builds:
- input block
- four encoder down blocks
- four decoder up blocks
- output convolution

Important choice:
- `n_classes=1` for binary segmentation

#### `UNet.forward(x)`
Runs the full U-Net:
1. encoder path
2. bottleneck
3. decoder path with skip connections
4. final logits output

Output shape:
- `(B, 1, H, W)`

Those are logits, not thresholded masks.

---

# 4. `preprocessing/`

## `preprocessing/__init__.py`
Package marker only.

## `preprocessing/suim_to_binary_masks.py`

This file converts SUIM RGB segmentation masks into saved binary masks on disk.

### `_convert_directory(input_dir, output_dir)`
Converts every mask file in one directory.

What it does:
- lists mask images
- loads each mask
- applies `suim_rgb_mask_to_binary`
- converts `{0,1}` into `{0,255}` for storage
- saves the output mask with the same filename

Why it exists:
- some workflows prefer preprocessing masks once rather than converting on the fly every epoch

### `convert_suim_masks(input_root, output_root)`
Runs SUIM conversion across expected SUIM subfolders.

It handles:
- `train_val/masks`
- `TEST/masks` if present

### `parse_args()`
Parses command-line arguments:
- `--input-root`
- `--output-root`

### `main()`
Command-line entry point:
- reads args
- runs `convert_suim_masks`

---

## `preprocessing/fathomnet_to_binary_masks.py`

This file normalizes FathomNet masks and saves them as binary files.

### `_convert_directory(input_dir, output_dir)`
Converts all masks in one directory tree.

What it does:
- recursively scans masks
- loads each mask
- applies `normalize_binary_mask`
- converts to `{0,255}` for storage
- preserves relative subfolder structure in the output

Why it preserves subfolders:
- FathomNet layouts may be nested

### `convert_fathomnet_masks(input_root, output_root)`
Runs mask conversion across supported FathomNet layouts.

It checks:
- `train/masks`
- `val/masks`
- `test/masks`
- flat `masks/`

If no supported mask directory exists, it raises an error.

### `parse_args()`
Parses:
- `--input-root`
- `--output-root`

### `main()`
Command-line entry point for the conversion script.

---

## `preprocessing/verify_masks.py`

This file is a debugging/sanity-check utility for masks.

### `summarize_masks(mask_dir, recursive=True, limit=20)`
Prints the unique pixel values found in a sample of mask files.

What it does:
- scans masks
- loads up to `limit` masks
- prints each file’s unique pixel values
- prints an aggregate summary

Why it exists:
- segmentation bugs often come from incorrect mask values
- it helps confirm whether masks really are binary

### `parse_args()`
Parses:
- `--mask-dir`
- `--limit`

### `main()`
Command-line entry point for mask inspection.

---

# 5. `training/`

## `training/__init__.py`
Package marker only.

## `training/metrics.py`

This file contains binary segmentation metrics.

### `_threshold_logits(logits, threshold=0.5)`
Converts raw model logits into binary predictions.

What it does:
1. applies sigmoid
2. thresholds probabilities at `threshold`
3. returns a float mask with values `{0,1}`

This helper is used by all metric functions.

### `binary_pixel_accuracy(logits, targets, threshold=0.5)`
Computes pixelwise accuracy.

Formula:
- fraction of pixels where predicted label equals target label

Limitation:
- can be misleading when background dominates

### `binary_iou(logits, targets, threshold=0.5, eps=1e-6)`
Computes foreground IoU.

Formula:
- intersection / union

Why it matters:
- this is one of the best segmentation quality metrics for binary masks

### `binary_dice(logits, targets, threshold=0.5, eps=1e-6)`
Computes the Dice coefficient.

Formula:
- `2 * intersection / (pred + target total)`

Why it matters:
- especially useful when masks are sparse or imbalanced

---

## `training/losses.py`

This file defines the training losses.

### `DiceLoss`
Loss version of the Dice metric.

#### `DiceLoss.__init__(eps=1e-6)`
Stores the smoothing constant.

#### `DiceLoss.forward(logits, targets)`
What it does:
- applies sigmoid to logits
- flattens prediction and target per sample
- computes soft Dice overlap
- returns `1 - mean_dice`

Why it exists:
- Dice-based losses help when the foreground occupies a small area

### `BCEDiceLoss`
A combined loss.

#### `BCEDiceLoss.__init__(bce_weight=0.5, dice_weight=0.5)`
Builds:
- internal `BCEWithLogitsLoss`
- internal `DiceLoss`
- weighting coefficients for both parts

#### `BCEDiceLoss.forward(logits, targets)`
Returns:
- `bce_weight * BCE + dice_weight * Dice`

Why this is useful:
- BCE provides stable optimization
- Dice emphasizes overlap quality

---

## `training/engine.py`

This file contains the reusable training and validation loops.

### `train_one_epoch(model, loader, optimizer, criterion, device, threshold=0.5)`
Runs one full training epoch.

What it does:
1. sets model to train mode
2. loops over batches
3. moves images and masks to device
4. computes logits
5. computes loss
6. backpropagates
7. updates optimizer
8. accumulates:
   - loss
   - accuracy
   - IoU
   - Dice

Output:
- a metrics dictionary with averaged values

Why it exists:
- keeps the actual training scripts short and clean

### `evaluate(model, loader, criterion, device, threshold=0.5)`
Runs one validation pass without gradient updates.

What it does:
- sets model to eval mode
- disables gradients
- computes the same metrics as training
- returns averaged metrics

This is the reusable evaluation companion to `train_one_epoch`.

---

## `training/train_suim.py`

This is the command-line training script for the SUIM pretraining stage.

### `parse_args()`
Defines the CLI arguments:
- dataset root
- output directory
- batch size
- learning rate
- epochs
- image size
- workers
- validation ratio
- seed
- threshold
- loss type
- device

### `main()`
This is the entry point for SUIM pretraining.

What it does:
1. parses arguments
2. seeds randomness
3. chooses CPU or GPU
4. builds SUIM train/val dataloaders
5. builds the U-Net with one output channel
6. chooses the loss function
7. creates the Adam optimizer
8. creates checkpoint directories
9. runs the epoch loop
10. saves:
   - `suim_fg_bg_last.pth`
   - `suim_fg_bg_best.pth`

Best model criterion:
- highest validation IoU

This is the first stage of the overall workflow.

---

## `training/train_fathomnet.py`

This is the command-line training script for the FathomNet fine-tuning stage.

### `parse_args()`
Defines CLI arguments similar to `train_suim.py`, plus one extra:

- `--pretrained-checkpoint`

That checkpoint is expected to be the SUIM-trained model.

### `main()`
This is the entry point for FathomNet fine-tuning.

What it does:
1. parses arguments
2. seeds randomness
3. chooses device
4. builds FathomNet train/val dataloaders
5. constructs the same one-channel U-Net
6. loads the pretrained SUIM checkpoint
7. chooses the loss
8. creates the optimizer
9. runs the epoch loop
10. saves:
   - `fathomnet_fg_bg_last.pth`
   - `fathomnet_fg_bg_best.pth`

Best model criterion:
- highest validation IoU

This is the transfer-learning stage of the pipeline.

---

# 6. `inference/`

## `inference/__init__.py`
Re-exports `predict_mask`.

## `inference/predict.py`

This file handles single-image inference.

### `predict_mask(model, image_path, device, img_size=256, threshold=0.5)`
Runs segmentation on one image.

What it does:
1. loads the RGB image
2. applies the validation transform
3. adds a batch dimension
4. runs the model
5. applies sigmoid
6. thresholds the result
7. returns a binary numpy mask

This is the core reusable inference function.

### `parse_args()`
Parses CLI arguments:
- checkpoint path
- image path
- output path
- image size
- threshold
- device

### `main()`
Command-line inference entry point.

What it does:
1. loads the model
2. loads checkpoint weights
3. runs `predict_mask`
4. saves the binary prediction as an image file

Output mask format:
- `0` or `255`

---

# 7. `utils/`

## `utils/__init__.py`
Re-exports the main utility functions:
- `seed_everything`
- `save_checkpoint`
- `load_checkpoint`
- `pair_image_mask_files`
- `list_image_files`

---

## `utils/io.py`

This file handles file discovery and image-mask matching.

### `IMAGE_EXTENSIONS`
Set of supported image file extensions.

### `list_image_files(folder, recursive=False)`
Returns image files from a directory.

What it does:
- checks the folder
- optionally searches recursively
- filters by supported file extension
- returns a sorted list

Used by:
- preprocessing scripts
- dataset pairing logic

### `_normalize_stem(path)`
Normalizes a filename stem for matching.

What it removes:
- suffixes like `_mask`, `_label`, `_gt`, `_seg`, `_annotation`

Why this matters:
- datasets often store masks with slightly different filenames than images
- stem normalization makes matching more robust

### `pair_image_mask_files(images_dir, masks_dir, recursive=False)`
Pairs images and masks automatically.

Matching logic:
1. exact filename match with alternate image extensions
2. normalized stem match
3. fuzzy prefix-style stem match

If any images remain unmatched, it raises an error.

Why this function is important:
- pairing image/mask files is foundational to the whole pipeline
- this is one of the most important utility functions in the package

---

## `utils/checkpointing.py`

This file handles saving and loading model checkpoints.

### `save_checkpoint(path, model, optimizer=None, epoch=None, metrics=None)`
Saves a checkpoint file.

Stored contents:
- model state dict
- optimizer state dict if provided
- epoch
- metrics

Why it exists:
- keeps checkpoint structure consistent across SUIM and FathomNet

### `load_checkpoint(path, model, optimizer=None, device="cpu", strict=True)`
Loads a checkpoint.

What it does:
- reads the checkpoint
- restores model weights
- optionally restores optimizer state
- returns the checkpoint payload

Important use:
- FathomNet fine-tuning loads the SUIM checkpoint with this function

---

## `utils/seed.py`

This file handles reproducibility.

### `seed_everything(seed=42)`
Sets seeds for:
- Python random
- NumPy
- PyTorch CPU
- PyTorch CUDA

It also adjusts cuDNN settings for deterministic behavior.

Why it matters:
- train/validation splits should be reproducible
- experiments should be easier to compare

---

# 8. End-to-end flow

The intended flow across the package is:

1. **Optional preprocessing**
   - convert SUIM or FathomNet masks into binary files on disk

2. **Dataset loading**
   - `SUIMBinaryDataset` or `FathomNetBinaryDataset`

3. **Transforms**
   - `get_train_transform` or `get_val_transform`

4. **Model**
   - `UNet(n_channels=3, n_classes=1)`

5. **Loss**
   - `BCEWithLogitsLoss` or `BCEDiceLoss`

6. **Metrics**
   - pixel accuracy
   - IoU
   - Dice

7. **Training**
   - `train_suim.py` for pretraining
   - `train_fathomnet.py` for fine-tuning

8. **Inference**
   - `predict.py`

---

# 9. Most important functions to understand first

If you want the quickest understanding of the workflow, read these in order:

1. `suim_rgb_mask_to_binary`
2. `normalize_binary_mask`
3. `SUIMBinaryDataset.__getitem__`
4. `FathomNetBinaryDataset.__getitem__`
5. `UNet.forward`
6. `train_one_epoch`
7. `evaluate`
8. `train_suim.main`
9. `train_fathomnet.main`
10. `predict_mask`

Those ten functions cover almost the whole pipeline.

---

# 10. Short practical summary

- `datasets/` turns files into tensors
- `models/` turns tensors into logits
- `losses.py` tells the model how wrong it is
- `metrics.py` tells us how good the predictions are
- `engine.py` runs one epoch of train/eval
- `train_suim.py` learns a binary underwater prior from SUIM
- `train_fathomnet.py` transfers that model to FathomNet
- `predict.py` runs the trained model on a new image
- `utils/io.py` is the glue that keeps image-mask matching working
