#!/bin/bash

#SBATCH --job-name=training # Job name
#SBATCH --open-mode=append
#SBATCH --output=/home/ngkanats/repos/lbs/analogical_manipulation/rlbench_output_%j.log
#SBATCH --error=/home/ngkanats/repos/lbs/analogical_manipulation/rlbench_error_%j.log
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00             # Time limit (hh:mm:ss)
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --gpus=4                    # Number of tasks (processes)
#SBATCH --cpus-per-task=24          # Number of cores per task
#SBATCH --mem=200G                  # Memory limit per node
#SBATCH --partition=general # Partition name
#SBATCH --requeue
#SBATCH --constraint="A100_80GB|L40S"

# Load any necessary modules (example: Python)
source /home/ngkanats/.bashrc
source /home/ngkanats/miniconda3/etc/profile.d/conda.sh
conda activate robot_26
# export LD_LIBRARY_PATH=/home/ngkanats/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
# export PATH=/home/ngkanats/local/cuda-12.4/bin:$PATH

# Run the Python script
nvidia-smi

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


cd /home/ngkanats/repos/lbs/analogical_manipulation
bash scripts/calvin/train.sh
