#! /bin/bash

# prepare data config_train_$1.json e.g. config_train_22.json
jid1=$(sbatch  submit_prepare.sh $1)
echo ${jid1}
echo ${jid1##* }
# train config_train_$1.json
jid2=$(sbatch  --dependency=afterok:${jid1##* } submit_train.sh $1)
echo ${jid2}
echo ${jid2##* }

#prdict config_predict_$1.json
jid3=$(sbatch  --dependency=afterok:${jid2##* } submit_predict.sh $1)
echo ${jid3}
echo ${jid3##* }

# show dependencies in squeue output:
squeue -u $USER # -o "%.8A %.4C %.10m %.20E"