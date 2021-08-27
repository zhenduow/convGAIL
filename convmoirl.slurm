#!/bin/bash
#SBATCH -A zhenduow
#SBATCH -n 1
#SBATCH --mail-user=zhenduow94@gmail.com
#SBATCH --mail-type=FAIL,END

#SBATCH --output=job.%J.out
#SBATCH --partition=titan-giant

export CUDA_VISIBLE_DEVICES=1
srun python3.6 run_sampling.py --dataset_name MSDialog --reranker_name poly --topn 1 --user_patience 5 --cv 0 --batch_size 100 --lr 1e-4 --n_policy 1 --cq_reward 0.6 \
> MSDialog_experiments/poly_t1_p5_cv0_gail_lr1e-4_r0.6.log
