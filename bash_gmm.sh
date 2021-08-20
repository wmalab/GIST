#! /bin/bash
lows=( 5 ) # 10 15 20 25 30 35 40 45 50)
num=( $(seq 3 4) )
for i in "${lows[@]}"
do
    for j in "${num[@]}"
    do
        sbatch submit_gmm.sh ${i} ${j}
    done
done