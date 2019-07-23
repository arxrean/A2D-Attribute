#!/bin/bash
#SBATCH --partition=gpu  
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:4
    srun python extract_frames.py