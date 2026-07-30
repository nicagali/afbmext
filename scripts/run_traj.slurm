#!/bin/bash
#SBATCH --job-name=afbmext
#SBATCH --output=slurm_jobs/run_traj09.out
#SBATCH --error=slurm_jobs/run_traj09.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --partition=normal
#SBATCH --time=10-00:00:00

alpha=0.9
T=100
h=0.01
v=200
mu=1
realizations=100
sets=2
start_set=0
n_jobs=25
    
source $HOME/.bashrc
conda activate afbm
echo "alpha=$alpha T=$T h=$h v=$v mu=$mu realizations=$realizations sets=$sets start_set=$start_set n_jobs=$n_jobs"
python scripts/run_trajectories.py "$alpha" "$T" "$h" "$v" "$mu" "$realizations" "$sets" "$start_set" "$n_jobs"


