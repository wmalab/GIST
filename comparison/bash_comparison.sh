#! /bin/bash
chrom=(20 21 22)  # 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X)
method=(chromsde) # lordg pastis gem shrec3d
for m in "${method[@]}"
do
    for i in "${chrom[@]}"
    do
        # prepare data
        jid1=$(sbatch  submit_prepare.sh $i $m)
        echo ${jid1}
        echo ${jid1##* }
        # infer structure
        jid2=$(sbatch  --dependency=afterok:${jid1##* } submit_infer.sh $i $m)
        echo ${jid2}
        echo ${jid2##* }
    done
done


# show dependencies in squeue output:
squeue -u $USER -o "%.8A %.4C %.10m %.20E"