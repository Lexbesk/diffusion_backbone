#!/bin/bash

#SBATCH --job-name=ngkanats # Job name
#SBATCH --output=job_%j.out         # Output file (with job ID)
#SBATCH --time=48:00:00             # Time limit (hh:mm:ss)
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --gpus=2                    # Number of tasks (processes)
#SBATCH --cpus-per-task=16          # Number of cores per task
#SBATCH --mem=96G                  # Memory limit per node
#SBATCH --partition=preempt # Partition name

# Load any necessary modules (example: Python)
source /home/ngkanats/.bashrc
conda activate 3dda-pt24
export LD_LIBRARY_PATH=/home/ngkanats/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export PATH=/home/ngkanats/local/cuda-12.4/bin:$PATH

# Run the Python script
nvidia-smi

# if [ ! -d /scratch/Peract_packaged ]; then
#     cd /scratch
#     wget https://huggingface.co/katefgroup/3d_diffuser_actor/resolve/main/Peract_packaged.zip
#     unzip Peract_packaged.zip
#     rm Peract_packaged.zip
# fi
if [ ! -d /scratch/GNFactor_zarr ]; then
    cd /scratch
    cp -r /data/user_data/ngkanats/GNFactor_zarr /scratch/
fi


cd /home/ngkanats/repos/lbs/analogical_manipulation
# bash scripts/rlbench/train_.sh
# bash scripts/rlbench/train_gnfactor_fast.sh
# bash scripts/calvin/train_mix.sh
bash debug.sh
