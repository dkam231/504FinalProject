# Setup

This project can be developed on Windows, WSL, or Linux. For the current SUIM U-Net workflow, Windows is fine as long as Python and PyTorch are installed correctly.

## 1. Create and activate a virtual environment

### Windows PowerShell

```powershell
git clone https://github.com/dkam231/504FinalProject.git
cd 504FinalProject
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux / WSL

```bash
git clone https://github.com/dkam231/504FinalProject.git
cd 504FinalProject
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

## 3. Install PyTorch

Install PyTorch first, using the command that matches your machine from the official PyTorch site:

https://pytorch.org/get-started/locally/

Examples:

### CPU only

```bash
pip install torch torchvision torchaudio
```

### NVIDIA GPU with CUDA

Use the exact command recommended by the PyTorch site for your CUDA version.

## 4. Install the rest of the project dependencies

```bash
pip install -r requirements.txt
```

## 5. Prepare the SUIM dataset


The SUIM dataset could be get by downloading `SUIM.zip` from `https://irvlab.cs.umn.edu/resources/suim-dataset` to `root/SUIM` and running the cpmmand below:
```bash
# From the project root
cd ./SUIM
unzip SUIM.zip -d data
```

Make sure the following data structure exists before running the code:

```text
504FinalProject/
  SUIM/
    data/
      train_val/
        images/
        masks/
      TEST/
        images/
        masks/
```

Important notes:

- The training code does not auto-download the dataset by default.
- `dataloader.py` contains a helper download function, but you must call it yourself if you want to use it.
- The mask palette is expected to match the SUIM RGB class colors exactly.

## 6. Sanity-check the data pipeline

```bash
# From the project root
cd ./SUIM
python check_dataset.py
```

You should see:

- nonzero train, val, and test sample counts
- image tensors shaped like `[B, 3, H, W]`
- mask tensors shaped like `[B, H, W]`
- mask labels in the range `0` through `7`
