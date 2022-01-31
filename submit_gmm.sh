#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=70G
#SBATCH --time=2-20:00:00
#SBATCH -p wmalab
#SBATCH --job-name=GIST_gmm
#SBATCH --output=/rhome/yhu/bigdata/proj/experiment_GIST/chromosome_3D/log/Job-GIST-gmm-%J.log

date
hostname
source activate env_GIST
echo python /rhome/yhu/bigdata/proj/experiment_GIST/chromosome_3D/clustering/gmm_clustering.py ${1} ${2} ${3}
python /rhome/yhu/bigdata/proj/experiment_GIST/chromosome_3D/clustering/gmm_clustering.py ${1} ${2} ${3}