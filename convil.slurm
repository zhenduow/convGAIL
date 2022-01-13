#!/bin/bash
#SBATCH -A zhenduow
#SBATCH -n 1
#SBATCH --mail-user=zhenduow94@gmail.com
#SBATCH --mail-type=FAIL,END

#SBATCH --output=job.%J.out
#SBATCH --partition=titan-giant


export CUDA_VISIBLE_DEVICES=0
srun python3.6 run.py \
--dataset_name UDC \
--cv 0 \
--reranker_name poly \
--user_tolerance 1 \
--user_patience 10 \
--batch_size 100 \
--lr 1e-4 \
--lrdc 0.97 \
--cq_reward 0.1 \
--cascade_p 0.7 \
--entropy_weight 0.01 \
--disc_train_ratio 5 \
--weight_decay 0.01 \
--il_topn 10 \
--max_data_size 10000 \
--max_iter 200 \
--gan_name WGAN \
--risk_run_epoch 0 \
--base_run_epoch 0 \
--disc_pretrain_epochs 30 \
--dropout_ratio 0
#--load_checkpoint True \
#--checkpoint_il il_31.666705399751663
