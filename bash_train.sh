#! /bin/bash
date
hostname
# source activate env_G3DM

chrom=(20 21 22) # 15 16 17 18 19 20 21 22 X) # 1 2 3 4 5 6 7 8 9 10 11 12 13 
for i in "${chrom[@]}"
do
    # prepare data
    echo python test_prepare_data.py config_train_${i}.json
    cp config_train_${i}.json ../data/
    # python test_prepare_data.py config_train_${i}.json

    # train
    echo python test_train.py config_train_${i}.json
    cp config_train_${i}.json ../data/
    # python test_train.py config_train_${i}.json    
done

