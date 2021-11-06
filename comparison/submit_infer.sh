#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=2-20:00:00
#SBATCH -p wmalab
#SBATCH --job-name=infer
#SBATCH --output=./log/Job-infer-%J.log

date
hostname
source activate env_cmp
module load matlab
# 21 GEM
echo python infer_structure.py $1 $2
python infer_structure.py $1 $2