#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=computefpd
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5
#SBATCH --output=log/output_log%j_%a
#SBATCH --error=log/error_log%j_%a
#SBATCH --mail-user=tian.li@port.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --partition=sciama4.q

module load system
module load anaconda3/2024.02
echo `module list`

source /mnt/lustre2/shared_conda/envs/tianli/herculens_tian/bin/activate
cd /users/tianli/LensModelling/Herculensedquasar/WFI2033

export HDF5_USE_FILE_LOCKING=FALSE
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""

python -u add_fpd_to_step6_imaging_only_nc.py --chain-index ${SLURM_ARRAY_TASK_ID}
