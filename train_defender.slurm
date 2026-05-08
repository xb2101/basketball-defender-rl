#!/bin/bash
#SBATCH --job-name=basketball_defender
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/xb2101/basketball_project/logs/defender_%j.out
#SBATCH --error=/scratch/xb2101/basketball_project/logs/defender_%j.err
#SBATCH --account=torch_pr_155_tandon_priority
#SBATCH --comment="preemption=yes;requeue=true"


# Load anaconda and activate environment
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/xb2101/basketball_rl

# Set scratch as temp dir
export TMPDIR=/scratch/xb2101/tmp

# Go to project directory
cd /scratch/xb2101/basketball_project

# Run training
/scratch/xb2101/basketball_rl/bin/python train_defender_round3.py --steps 10000000 --checkpoint ./defender_hpc_v6_final --scorer ./scorer_ppo_hpc_v7_final --save-dir ./checkpoints_defender_r3/ --save-freq 500000

echo "Training complete!"
