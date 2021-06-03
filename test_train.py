import time,os
import json
import dgl
import torch

import numpy as np
from G3DM.train import load_dataset, create_network, setup_train

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    device = 'cpu'

    root= os.path.join('.') # '/rhome/yhu/bigdata/proj/experiment_G3DM'
    configuration_src_path = os.path.join(root, 'data')
    configuration_name = 'config.json'
    with open(os.path.join(configuration_src_path, configuration_name)) as f:
        config_data = json.load(f)

    cool_file = config_data['cool_file']
    cell = cool_file.split('.')[0]
    hyper = '_'.join([cool_file.split('.')[1], 'id', config_data["id"]])
    
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

    '''ratio = config_data['hyper']['ratio']
    stride = config_data['hyper']['stride']
    max_dim = config_data['parameter']['feature']['max_feature_dim']'''
    all_chromosome = config_data['all_chromosomes']
    train_chromosomes = config_data['train_chromosomes']
    valid_chromosomes = config_data['valid_chromosomes']
    test_chromosomes = config_data['test_chromosomes']

    '''clusters = config_data['parameter']['graph']['num_clusters']
    num_clusters = dict()
    for key, value in clusters.items():
        num_clusters[int(key)] = int(value)

    percents = config_data['parameter']['graph']['cutoff_percent']
    cutoff_percents = dict()
    for key, value in percents.items():
        cutoff_percents[int(key)] = int(value)

    cutoff_cluster = int(config_data['parameter']['graph']['cutoff_cluster'])'''

    # load dataset 
    HiCDataset = load_dataset(dataset_path, dataset_name)
    graphs, features, label = HiCDataset[0]

    # creat network model
    sampler, em_networks, ae_networks, nll, opt = create_network(config_data, graphs)

    # setup and call train 
    itn, batch_size = setup_train(config_data)