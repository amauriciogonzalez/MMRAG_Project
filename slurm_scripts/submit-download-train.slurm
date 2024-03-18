#!/bin/bash

#SBATCH --time=03:00:00
# #SBATCH --mem=32G
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# #SBATCH --constraint=gpu32
# #SBATCH --mem=16G
#SBATCH --gres-flags=enforce-binding
# #SBATCH --exclude=evc[1-10]
#SBATCH --exclude=evc[1-14]
#SBATCH --job-name=final-project-download-train
#SBATCH --error=final-project-download-train-%J.err
#SBATCH --output=final-project-download-train-%J.out

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user ad375233@ucf.edu

# Give this process 1 task and 1 GPU, then assign four CPUs per task
# (so 4 cores overall).

# If you want two GPUs:
# #SBATCH --gres=gpu:2
# #SBATCH --cpus-per-task=8
# #SBATCH --mem-per-cpu=4G # memory per cpu-core
# #SBATCH --mem=2G           # total memory per node, alternative to above
# This example, however, only uses one GPU.

# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"


## Load Modules ##
module load anaconda/anaconda3

# Load the Python and CUDA modules
module load python/python-3.8.0-gcc-9.1.0
module load cuda/cuda-10.2

# List the modules that are loaded
module list

# Have Nvidia tell us the GPU/CPU mapping so we know
nvidia-smi topo -m

echo

# Activate the GPU version of PyTorch
source activate pytorch-1.8.0+cuda10_2


## Run job ##
source ~/vllava/bin/activate
cd ..
pip install -e ".[train]"
pip install flash-attn --no-build-isolation --no-cache-dir
deactivate

echo "Ending script..."
date
