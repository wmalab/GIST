import os, sys, shutil
import json, pickle
import numpy as np
import multiprocessing

from prepare.utils import load_graph
from prepare.build_data import create_fit_data
from prepare.build_feature import load_feature

from prepare.build_dataset import HiCDataset

import torch

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # root = '.'
    root = '/rhome/yhu/bigdata/proj/experiment_G3DM'

    configuration_src_path = os.path.join(root, 'data')
    configuration_name = 'config_train.json'
    with open(os.path.join(configuration_src_path, configuration_name)) as f:
        config_data = json.load(f)

    cool_file = config_data['cool_file']
    cell = cool_file.split('.')[0]
    hyper = '_'.join([cool_file.split('.')[1], config_data["id"]])

    configuration_dst_path = os.path.join(root, 'data', cell, hyper)
    os.makedirs(configuration_dst_path, exist_ok=True)
    shutil.copy(os.path.join(configuration_src_path,
                             configuration_name), configuration_dst_path)

    # '/rhome/yhu/bigdata/proj/experiment_G3DM'
    root = root
    cool_data_path = os.path.join( root, 'data', 'raw')
    graph_path = os.path.join( root, 'data', cell, hyper, 'graph')
    feature_path = os.path.join( root, 'data', cell, hyper, 'feature')
    dataset_path = os.path.join( root, 'data', cell, hyper)
    dataset_name = 'dataset.pt'
    output_path = os.path.join( root, 'data', cell, hyper, 'output')

    saved_model_path = os.path.join( root, 'data', cell, hyper, 'saved_model')
    saved_model_name = 'model_net'

    os.makedirs(graph_path, exist_ok=True)
    os.makedirs(feature_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(saved_model_path, exist_ok=True)

    all_chromosome = config_data['all_chromosomes']
    train_chromosomes = config_data['train_valid_chromosomes']

    num_clusters = config_data['parameter']['graph']['num_clusters']
    max_len = config_data['parameter']['graph']['max_len']
    cutoff_clusters_limits = config_data['parameter']['graph']['cutoff_clusters']
    cutoff_cluster = config_data['parameter']['graph']['cutoff_cluster']

    dim = config_data['parameter']['feature']['in_dim']

    for chromosome in all_chromosome:
        create_fit_data(num_clusters, chromosome, dim,
                        cutoff_clusters_limits, 
                        cutoff_cluster, 
                        max_len,
                        cool_data_path, cool_file,
                        [feature_path, graph_path])

    graph_dict = dict()
    feature_dict = dict()
    cluster_weight_dict = dict()
    train_list, test_list = list(), list()
    for chromosome in all_chromosome:
        feature_dict[str(chromosome)] = load_feature(feature_path, 'F_chr-{}'.format(chromosome))
        cluster_weight_dict[str(chromosome)] = feature_dict[str(chromosome)]['cluster_weight']

        graph_dict[str(chromosome)]=dict()
        g_path = os.path.join(graph_path, 'chr{}'.format(chromosome))
        files = [f for f in os.listdir(g_path) if 'chr-{}'.format(chromosome) in f]
        for file in files:
            gid = file.split('.')[0]
            gid = gid.split('_')[-1]
            gid = int(gid)
            g, _ = load_graph(g_path, file)
            graph_dict[str(chromosome)][gid] = g

        # if str(chromosome) in train_chromosomes:
        #     train_list.append(str(chromosome))
        # if str(chromosome) in test_chromosomes:
        #     test_list.append(str(chromosome))
    
    # create HiCDataset and save
    # HD = HiCDataset(graph_dict, feature_dict, cluster_weight_dict, train_list, test_list, dataset_path, dataset_name)
    HD = HiCDataset(graph_dict, feature_dict, cluster_weight_dict, dataset_path, dataset_name)
    torch.save(HD, os.path.join( dataset_path, dataset_name))

    '''load_HD = torch.load(os.path.join( dataset_path, dataset_name))
    load_HD[0]
    print(load_HD[0])'''
