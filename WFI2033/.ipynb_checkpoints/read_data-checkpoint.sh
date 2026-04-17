#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=readnc
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --output=log/output_log%j
#SBATCH --error=log/error_log%j
#SBATCH --mail-user=tian.li@port.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --partition=sciama4.q


module load system
module load anaconda3/2024.02
echo `module list`

source /mnt/lustre2/shared_conda/envs/tianli/herculens_tian/bin/activate
cd /users/tianli/LensModelling/Herculensedquasar/WFI2033


python -u merge_step6_imaging_only_fpd_shards.py
#python -u Herculens_3DSPL_EPL.py
