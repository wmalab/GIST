#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=2-20:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:k80:1
#SBATCH --job-name=G3DM_train
#SBATCH --output=Job-G3DM-train-%J.log

date
hostname
source activate env_G3DM
module load cuda/10.2.2
module load GCC/8.3.0
echo python test_train_1lvl.py
cp config_1lvl.json ../data/
python test_train_1lvl.py