#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --time=2-20:00:00
#SBATCH -p wmalab
#SBATCH --job-name=G3DM_gmm
#SBATCH --output=/rhome/yhu/bigdata/proj/experiment_G3DM/chromosome_3D/log/Job-G3DM-gmm-%J.log

date
hostname
source activate env_G3DM
echo python /rhome/yhu/bigdata/proj/experiment_G3DM/chromosome_3D/clustering/gmm_clustering.py ${1} ${2} ${3}