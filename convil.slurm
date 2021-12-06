#!/bin/bash
#SBATCH -A zhenduow
#SBATCH -n 1
#SBATCH --mail-user=zhenduow94@gmail.com
#SBATCH --mail-type=FAIL,END

#SBATCH --output=job.%J.out
#SBATCH --partition=titan-giant


export CUDA_VISIBLE_DEVICES=1
srun python3.6 run.py \
--dataset_name MSDialog \
--cv 1 \
--reranker_name poly \
--user_tolerance 1 \
--user_patience 10 \
--batch_size 100 \
--lr 1e-10 \
--lrdc 0.99 \
--cq_reward 0.1 \
--cascade_p 0.9 \
--max_d_kl 1e-2 \
--entropy_weight 0 \
--disc_train_ratio 5 \
--il_topn 10 \
--max_data_size 10000 \
--max_iter 500 \
--gan_name GAN \
#--load_checkpoint True \
#--checkpoint_il il_-10000
