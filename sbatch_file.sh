#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=NGPU
#SBATCH --nodelist=gpu-210
#SBATCH --exclude=gpu-120
# SBATCH --partition=CPU
cd /home/yalavarthi/seperable_flows/
srun -u /home/yalavarthi/miniconda3/envs/linodenet/bin/python $@