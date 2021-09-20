#!/bin/bash
#SBATCH -A zhenduow
#SBATCH -n 1
#SBATCH --mail-user=zhenduow94@gmail.com
#SBATCH --mail-type=FAIL,END

#SBATCH --output=job.%J.out
#SBATCH --partition=titan-giant


export CUDA_VISIBLE_DEVICES=3
srun python3.6 run.py \
--dataset_name MSDialog \
--cv 0 --reranker_name poly \
--user_tolerance 1 \
--user_patience 5 \
--batch_size 100 \
--lr 1e-5 \
--lrdc 0.95 \
--cq_reward 0.1 \
--cascade_p 0.95
#--load_checkpoint True \
#--checkpoint_il il_-441.71435260772705
