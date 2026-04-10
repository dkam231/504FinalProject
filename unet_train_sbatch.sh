#!/bin/bash
#SBATCH --job-name=unet_train1
#SBATCH --mail-user=sudeshi@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=eecs504s001w26_class
#SBATCH --partition=gpu_mig40,gpu,spgpu
#SBATCH --gres=gpu:1
#SBATCH--partition=gpu_mig40,gpu,spgpu 
#SBATCH --output=unetTrain1.log
#SBATCH --error=unetTrain1_error.log

# Activate conda environment
source /home/sudeshi/.bashrc
conda activate unet

cd /scratch/eecs504s001w26_class_root/eecs504s001w26_class/sudeshi/Fin_Project/504FinalProject/SUIM

SUIM_NUM_WORKERS=1 python train.py 