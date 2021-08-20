#! /bin/bash
chrom=( 22 )
lows=( 5 ) # 10 15 20 25 30 35 40 45 50)
num=( $(seq 3 4) )
for i in "${chrom[@]}"
do
    for j in "${lows[@]}"
    do
        for k in "${num[@]}"
        do
            sbatch submit_gmm.sh ${i} ${j} ${k}
        done
    done
done