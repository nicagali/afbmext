#!/bin/bash
#SBATCH --job-name=afbmext
#SBATCH --output=slurm_jobs/run%a.out
#SBATCH --error=slurm_jobs/run%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=normal
#SBATCH --time=10-00:00:00
#SBATCH --array=0-1

ALPHAS=(0.9)

ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}
T = 10
h = 0.001
v = 10
mu = 1
realizations = 100

source $HOME/.bashrc
conda activate afbm

python scripts/msd.py $ALPHA T h v mu realizations


