#! /bin/bash

# prepare data
jid1=$(sbatch  submit_prepare.sh)
echo ${jid1}
echo ${jid1##* }
# train
jid2=$(sbatch  --dependency=afterok:${jid1##* } submit_demo.sh)
echo ${jid2}
echo ${jid2##* }

# show dependencies in squeue output:
squeue -u $USER -o "%.8A %.4C %.10m %.20E"