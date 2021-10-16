#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=2-20:00:00
#SBATCH -p wmalab
#SBATCH --job-name=prepare_${2}_${1}
#SBATCH --output=Job-prepare_${2}_${1}-%J.log

date
hostname
source activate env_cmp
echo python prepare_data.py $1 $2
python prepare_data.py $1 $2