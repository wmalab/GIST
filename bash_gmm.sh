#! /bin/bash
chrom=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X)
lows=( 5 10 15 20 25 30 35 40 45 50)
num=( $(seq 3 20) )
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