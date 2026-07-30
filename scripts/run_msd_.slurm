#!/bin/bash
#SBATCH --job-name=afbmext
#SBATCH --output=slurm_jobs/run%a.out
#SBATCH --error=slurm_jobs/run%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10-00:00:00
#SBATCH --array=0-2


alpha=0.9
T=1000
h=0.001
mu=1
realizations=500
v=200

alphas=(0.9 0.7 0.5)
alpha=${alphas[$SLURM_ARRAY_TASK_ID]}
    
source $HOME/.bashrc
conda activate psearch
echo "alpha=$alpha T=$T h=$h v=$v mu=$mu realizations=$realizations"
python scripts/run_trajectories.py "$alpha" "$T" "$h" "$v" "$mu" "$realizations" 


