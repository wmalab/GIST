#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=300G
#SBATCH --time=2-20:00:00
#SBATCH -p wmalab
#SBATCH --job-name=G3DM_prepare
#SBATCH --output=Job-G3DM-prepare-%J.log

date
hostname
source activate env_G3DM
echo python ./clustering/gmm_clustering.py ${1} ${2}