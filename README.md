# 🌊 Implementing U-Net Pipelines for Robust Underwater Image Segmentation

## Team Members
* Divyam Kamboj
* Shivam Udeshi
* Jaemin Jeon
* Zhe Jiang

## Project Description
FathomNet is an open-source, expert-annotated image database designed to train machine learning algorithms to understand ocean life and underwater environments. The baseline segmentation model provided by FathomNet utilizes a **ResNet backbone** with **Detectron2**. 

While ResNet excels at image classification, it tends to lose concrete spatial details as data moves through deep layers, especially in high-noise, low-contrast underwater environments. 

This project aims to replace the generic ResNet + Detectron2 architecture with a custom **U-Net** pipeline. U-Net’s symmetric encoder-decoder structure and skip connections allow it to recover fine spatial details and delicate boundaries of marine organisms far more effectively. By focusing on pixel-level Foreground/Background (FG/BG) segmentation, we aim to achieve higher precision (measured via mIoU and Pixel Accuracy) while simplifying the training pipeline.

## Key Features
* **Adaptive Asynchronous Downloader (`download.py`):** A custom, highly optimized data ingestion script using `asyncio` and `httpx` with adaptive worker scaling and exponential backoff.
* **Baseline Benchmarking:** Evaluation of the off-the-shelf FathomNet Detectron2 model on the out-of-sample dataset.
* **Custom U-Net Architecture:** A specialized segmentation model optimized for underwater imagery.

### 1. Requirements
Ensure you have Python 3.8+ installed. You will need the following libraries:
```bash
pip install torch torchvision
pip install httpx coco-lib tenacity tqdm