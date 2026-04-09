#!/bin/bash
#SBATCH --job-name=download_fathomnet
#SBATCH --account=eecs504s001w26_class   
#SBATCH --partition=standard           # Use the standard CPU partition
#SBATCH --time=04:00:00                # Request 4 hours (should take ~2 hours, but 4 is safe)
#SBATCH --nodes=1                      # Use 1 node
#SBATCH --ntasks=1                     # Run 1 task
#SBATCH --cpus-per-task=1              # Use 1 CPU core
#SBATCH --mem=8GB                      # Request 8GB of RAM
#SBATCH --output=download_log_%j.txt   # Save terminal output to this file (%j adds the Job ID)
#SBATCH --error=download_error_%j.txt  # Save any errors to this separate file

# 1. Go to your project directory
cd /home/dkamboj/504FinalProject/fathomnet_data_download # you might need to change this for your own local env

# 2. (Optional but recommended) Load your Anaconda environment if you aren't using the default system Python
# module load python3.11-anaconda/2024.02-1
module load mamba/py3.12
# conda activate base
conda activate 504env

echo "Starting FathomNet download job..."
date

# 3. Run the Python script
python3 download_all_fathomnet_data.py

echo "Job completed!"
date