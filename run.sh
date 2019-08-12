#!/bin/bash
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
    # srun python ./utils/calculate_mean_std.py
    # srun python main.py --cuda True
    srun python eval.py --cuda True