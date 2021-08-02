import os, sys, shutil
import json, pickle
import numpy as np
import multiprocessing

from prepare.utils import log1p_hic, save_graph, load_graph, hic_prepare
from prepare.build_graph import create_hierarchical_graph_2lvl
from prepare.build_data import create_data
from prepare.build_feature import load_feature

from prepare.build_dataset import HiCDataset

import torch

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    root = '.'
    # root = '/rhome/yhu/bigdata/proj/experiment_G3DM'

    configuration_src_path = os.path.join(root, 'data')
    configuration_name = 'config_1lvl.json'
    with open(os.path.join(configuration_src_path, configuration_name)) as f:
        config_data = json.load(f)

    cool_file = config_data['cool_file']
    cell = cool_file.split('.')[0]
    hyper = '_'.join([cool_file.split('.')[1], 'id', config_data["id"]])

    configuration_dst_path = os.path.join(root, 'data', cell, hyper)
    os.makedirs(configuration_dst_path, exist_ok=True)
    shutil.copy(os.path.join(configuration_src_path,
                             configuration_name), configuration_dst_path)

    # '/rhome/yhu/bigdata/proj/experiment_G3DM'
    root = config_data['root'] if config_data['root'] else root
    cool_data_path = config_data['cool_data_path'] if config_data['cool_data_path'] else os.path.join( root, 'data', 'raw')
    graph_path = config_data['graph_path'] if config_data['graph_path'] else os.path.join( root, 'data', cell, hyper, 'graph')
    feature_path = config_data['feature_path'] if config_data['feature_path'] else os.path.join( root, 'data', cell, hyper, 'feature')
    dataset_path = config_data['dataset_path']['path'] if config_data['dataset_path']['path'] else os.path.join( root, 'data', cell, hyper)
    dataset_name = config_data['dataset_path']['name'] if config_data['dataset_path']['name'] else 'dataset.pt'
    output_path = config_data['output_path'] if config_data['output_path'] else os.path.join( root, 'data', cell, hyper, 'output')

    saved_model_path = config_data['saved_model']['path'] if config_data['saved_model']['path'] else os.path.join( root, 'saved_model')
    saved_model_name = config_data['saved_model']['name'] if config_data['saved_model']['name'] else 'model_net'

    os.makedirs(graph_path, exist_ok=True)
    os.makedirs(feature_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(saved_model_path, exist_ok=True)

    all_chromosome = config_data['all_chromosomes']
    train_chromosomes = config_data['train_valid_chromosomes']
    test_chromosomes = config_data['test_chromosomes']

    num_clusters = config_data['parameter']['graph']['num_clusters']
    max_len = config_data['parameter']['graph']['max_len']
    iteration = config_data['parameter']['graph']['iteration']
    # offset = config_data['parameter']['graph']['offset']
    cutoff_percents = config_data['parameter']['graph']['cutoff_percent']
    cutoff_cluster = config_data['parameter']['graph']['cutoff_cluster']

    dim = config_data['parameter']['feature']['in_dim']

    for chromosome in all_chromosome:
        for_test = True if chromosome in test_chromosomes else False
        create_data(num_clusters, chromosome, for_test,
                    dim,
                    cutoff_percents, cutoff_cluster, 
                    max_len, iteration,
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

        if str(chromosome) in train_chromosomes:
            train_list.append(str(chromosome))
        if str(chromosome) in test_chromosomes:
            test_list.append(str(chromosome))
    
    # create HiCDataset and save
    HD = HiCDataset(graph_dict, feature_dict, cluster_weight_dict, train_list, test_list, dataset_path, dataset_name)
    torch.save(HD, os.path.join( dataset_path, dataset_name))

    '''load_HD = torch.load(os.path.join( dataset_path, dataset_name))
    load_HD[0]
    print(load_HD[0])'''
