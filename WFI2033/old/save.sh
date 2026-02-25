#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=compositejackpot
#SBATCH --ntasks-per-node=48
#SBATCH --time=24:00:00
#SBATCH --output=log/output_log%j
#SBATCH --error=log/error_log%j
#SBATCH --mail-user=tian.li@port.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu.q


module load system
module load anaconda3/2024.02
echo `module list`

source /mnt/lustre2/shared_conda/envs/ckraw/gpu_herculens_3/bin/activate
cd /users/tianli/LensModelling/Slice_project


python -u save_data.py