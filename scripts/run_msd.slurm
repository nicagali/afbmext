#!/bin/bash
#SBATCH --job-name=afbmext
#SBATCH --output=slurm_jobs/run%a.out
#SBATCH --error=slurm_jobs/run%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --partition=normal
#SBATCH --time=10-00:00:00
#SBATCH --array=0-5


alpha=0.9
T=200
h=0.001
mu=1
realizations=500

velocities=(0 10 50 100 200 500)
v=${velocities[$SLURM_ARRAY_TASK_ID]}
    
source $HOME/.bashrc
conda activate afbm
echo "alpha=$alpha T=$T h=$h v=$v mu=$mu realizations=$realizations"
python scripts/msd.py "$alpha" "$T" "$h" "$v" "$mu" "$realizations" 


