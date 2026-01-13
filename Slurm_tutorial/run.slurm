#!/bin/bash
#SBATCH --job-name=rtx5090_test
#SBATCH --output=rtx5090_test_%j.out
#SBATCH --error=rtx5090_test_%j.err
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
##SBATCH --mem=30G
#SBATCH --gres=gpu:rtx5090:2
#### #####     gpu:gpu_model:gpuNumber
#
#place your code here; same as in terminal
./gpu_test
