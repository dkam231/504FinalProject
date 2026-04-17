# 🌊 Implementing U-Net Pipelines for Robust Underwater Image Segmentation

## Team Members
* Divyam Kamboj
* Shivam Udeshi
* Jaemin Jeon
* Zhe Jiang

## Project Description
FathomNet is an open-source, expert-annotated image database designed to train machine learning algorithms to understand ocean life and underwater environments. Its segmentation model provided by FathomNet utilizes a **ResNet backbone** with **Detectron2**. 

While ResNet excels at image classification, it tends to lose concrete spatial details as data moves through deep layers, especially in high-noise, low-contrast underwater environments. 

This project aims to replace the generic ResNet architecture with a custom **U-Net** pipeline. Unlike commaonly used model (baseline: ResNet-50 + DeepLabV3), U-Net’s symmetric encoder-decoder structure and skip connections allow it to recover fine spatial details and delicate boundaries of marine organisms far more effectively. By focusing on pixel-level Foreground/Background (FG/BG) segmentation, we aim to achieve higher precision (measured via mIoU) while simplifying the training pipeline.

## Key Features
* **Adaptive Asynchronous Downloader:** A custom, highly optimized data ingestion script using `asyncio` and `httpx` with adaptive worker scaling and exponential backoff.
* **Baseline Benchmarking:** Evaluation of the off-the-shelf FathomNet Detectron2 model on the out-of-sample dataset.
* **Custom U-Net Architecture:** A specialized segmentation model optimized for underwater imagery.

### 1. Requirements
Ensure you have Python 3.8+ installed. You will need the following libraries:
```bash
pip install torch torchvision
pip install httpx coco-lib tenacity tqdm
```

The FathomNet dataset could be get by downloading `Download train.json` (281 MB) and `Download test.json` (22.2 MB) from `https://database.fathomnet.org/fathomnet/#` to `root/Fathomnet_Code/fathomnet_data_download` and running the cpmmand below:
```bash
cd ./Fathomnet_Code/fathomnet_data_download
python ./download_all_fathomnet_data.py
```

The SUIM dataset could be get by downloading `SUIM.zip` from `https://irvlab.cs.umn.edu/resources/suim-dataset` to `root/SUIM` and running the cpmmand below:
```bash
cd ./SUIM
unzip SUIM.zip -d data
```

### 2. U-Net
```bash
cd ./SUIM
python train.py
python test.py
```

### 3. Baseline
ResNet-101 + Detectron2 could be executed by running the command below:
```bash
cd ./SUIM/baseline_models/detectron
python train.py
python test.py
```

ResNet-50 + DeepLabV3 could be executed by running the command below:
```bash
cd ./SUIM/baseline_models/deeplab
python train.py
python test.py
```