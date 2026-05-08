#!/bin/bash
#SBATCH --job-name=basketball_scorer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/xb2101/basketball_project/logs/scorer_%j.out
#SBATCH --error=/scratch/xb2101/basketball_project/logs/scorer_%j.err
#SBATCH --account=torch_pr_155_general
#SBATCH --comment="preemption=yes;requeue=true"

module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/xb2101/basketball_rl

export TMPDIR=/scratch/xb2101/tmp

cd /scratch/xb2101/basketball_project

/scratch/xb2101/basketball_rl/bin/python train_scorer_hpc.py --steps 20000000 --defender ./defender_hpc_v6_final --save-dir ./checkpoints_scorer_v14/ --save-freq 500000

echo "Scorer training complete!"
