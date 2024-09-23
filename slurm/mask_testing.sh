#!/bin/bash
#SBATCH -J val_50                # job name
#SBATCH --nodes=1                  #
#SBATCH --time=0-05:00:00          # set time limit
#SBATCH --partition=gpu-h100      # GPU partition
#SBATCH --account=awaken           # allocation
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100:4          # GPU request
#SBATCH --output=logs/%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=storm@mit.edu
#SBATCH --exclusive                # reserve entire node
#SBATCH --qos=high

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MPICH_GPU_SUPPORT_ENABLED=1
OMPI_MCA_opal_cuda_support=true

conda run --no-capture-output -n diffusion python3 run.py -c config/mask_testing.json -gpu 0,1,2,3 -b 32
