#!/bin/bash
#SBATCH --nodes=1 # Get one node
#SBATCH --cpus-per-task=2 # One cores per task
#SBATCH --ntasks=1 # But only one task
#SBATCH --gres=gpu:2 # And two GPUs
#SBATCH --gres-flags=enforce-binding # Insist on good CPU/GPU alignment
#SBATCH --time=71:59:59 # Run for 3 days, at most
#SBATCH --job-name=Proejct_St1_GVT # Name the job so I can see it in squeue
#SBATCH --mail-type=BEGIN,END,FAIL # Send me email for various states
#SBATCH --mail-user ma649596@ucf.edu # Use this address
#SBATCH --output=outputs/gvt_1.out
#SBATCH --error=outputs/gvt_1.err
#SBATCH --constraint=gpu80

# Load modules
module load anaconda/anaconda3

source /apps/anaconda/anaconda3/etc/profile.d/conda.sh

source activate lavis


torchrun train.py --cfg-path lavis/projects/blip2/train/pretrain_gvt_stage1.yaml