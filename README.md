# Robust Underwater Segmentation with U-Net  
### EECS 504 Final Project

**Authors:** Divyam Kamboj, Shivam Udeshi, Jaemin Jeon, Zhe Jiang

## Overview

This repository contains our EECS 504 final project on **underwater image segmentation using U-Net-based pipelines**.  
The project focuses on building a lightweight and reproducible segmentation workflow for underwater scenes, with an emphasis on:

- preserving fine spatial structure in noisy underwater images,
- improving foreground/background mask quality,
- using a simpler segmentation pipeline than heavier detection-to-segmentation baselines,
- supporting both **SUIM-based training** and **FathomNet transfer experiments**.

Our core motivation is that underwater imagery is difficult: contrast is low, boundaries are weak, and objects are often partially occluded or visually blended into the background. Instead of relying only on large, multi-stage pipelines, we explore whether a clean U-Net workflow can provide strong segmentation quality while remaining easier to train, debug, and extend.

---

## Project Goals

This repo supports multiple stages of the project:

1. **Multiclass semantic segmentation on SUIM**
   - Train a U-Net on the SUIM underwater segmentation dataset.
   - Evaluate with pixel accuracy and mean IoU.

2. **Binary foreground/background segmentation**
   - Collapse underwater segmentation into a simpler FG/BG task.
   - Emphasize mask quality and boundary preservation.

3. **Transfer learning from SUIM to FathomNet**
   - Pretrain on SUIM.
   - Fine-tune on FathomNet binary masks for underwater organism segmentation.

4. **Experimentation and analysis**
   - Jupyter notebooks for supervised and unsupervised experiments.
   - Preprocessing scripts for dataset construction and mask generation.
   - Utilities for downloading, organizing, and validating underwater data.

---

## Why U-Net?

The baseline ecosystem around underwater segmentation often uses more complex pipelines, including detection backbones followed by segmentation components. In contrast, **U-Net** is a strong fit for this task because:

- it preserves spatial detail through skip connections,
- it performs well on pixel-level prediction tasks,
- it is relatively lightweight and interpretable,
- it is easier to adapt to binary segmentation and transfer learning.

For underwater scenes, this is especially useful because thin structures, curved object boundaries, and small foreground regions are easy to lose in deeper feature hierarchies.

---

## Repository Structure

```text
504FinalProject/
├── SUIM/                         # Core SUIM segmentation code
│   ├── baseline_models/
│   ├── dataloader.py
│   ├── model.py
│   ├── model_quantized.py
│   ├── quantize_model.py
│   ├── check_dataset.py
│   ├── test.py
│   └── ...
│
├── SUIM_Dataset/                 # Standalone SUIM training entry point
│   └── train.py
│
├── combined/                     # Final integrated SUIM → FathomNet FG/BG workflow
│   ├── datasets/
│   ├── inference/
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   ├── utils/
│   ├── README.md
│   └── DESIGN_DECISIONS.md
│
├── fathomnet_data_download/      # FathomNet data download helpers
│   ├── download_all_fathomnet_data.py
│   ├── submit_download.sh
│   └── where_to_look_for_data.txt
│
├── unsup_sup_notebooks/          # Experimental notebooks
│   ├── model.ipynb
│   ├── unsupervised_train_500_from_model.ipynb
│   └── fathomnet_data_visualization_starter.ipynb
│
├── build_fathomnet_seg_dataset.py
├── download.py
├── download_images.py
├── download_fathomnet_segmentations.py
├── preprocess.py
├── preprocess_pixel_level.py
├── process_test_data.py
├── SETUP.md
├── Run.md
└── requirements.txt
```

## Setup and Run
Please refer to `SETUP.md` and `RUN.md` respectively.