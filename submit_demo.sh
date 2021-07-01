#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:k80:1
#SBATCH --job-name=G3DM
#SBATCH --output=Job-G3DM-%J.log

date
hostname
source activate env_G3DM
echo python test_train.py
python test_train.py