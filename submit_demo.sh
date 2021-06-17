#!/bin/bash
#SBATCH --partition=wmalab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:k80:1
#SBATCH --job-name=G3DM
#SBATCH --output=submit-%J.log

# Change to HOME dir to give access to all folders within Jupyter-Lab
cd $HOME

source activate env_G3DM
python test_train.py