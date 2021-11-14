#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=1-10:00:00
#SBATCH -p wmalab
#SBATCH --job-name=G3DM_prepare
#SBATCH --output=Job-G3DM-prepare-%J.log

date
hostname
source activate env_G3DM
echo python test_prepare_data.py config_train_$1.json
cp config_train_$1.json ../data/
python test_prepare_data.py config_train_$1.json