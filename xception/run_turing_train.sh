#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem 100G
#SBATCH --gres=gpu:2
#SBATCH -p normal
#SBATCH -o output_train_Xcept.stdout
#SBATCH -t 12:00:00
#SBATCH -J T_Xcept
module load cuda10.0/toolkit/10.0.130
module load cuda10.0/fft/10.0.130
module load cudnn/7.4.2
/cm/local/apps/cuda/libs/current/bin/nvidia-smi
conda activate py37
python agent_motion_prediction_xception.py	
