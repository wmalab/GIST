#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=300G
#SBATCH --time=2-20:00:00
#SBATCH -p wmalab
#SBATCH --job-name=GIST_predict
#SBATCH --output=Job-GIST-predict-%J.log

date
hostname
source activate env_GIST
echo python test_prediction.py config_predict_$1.json
cp config_predict_$1.json ../data/
python test_prediction.py config_predict_$1.json