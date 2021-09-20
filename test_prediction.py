import time
import os
import json
import dgl
import torch
from torch.utils import tensorboard
import GPUtil

import numpy as np
from G3DM.fit import load_dataset, create_network, setup_train, run_prediction
from G3DM.model import save_model_state_dict

import warnings
warnings.filterwarnings('ignore')

gpuIDs = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.05,
                             maxMemory=0.05, includeNan=False, 
                             excludeID=[], excludeUUID=[])
# device = torch.device('cpu' if len(gpuIDs) == 0 else 'cuda:{}'.format(gpuIDs[0]))
device = torch.device('cpu' if len(gpuIDs) == 0 else 'cuda:0')
print(device)

if __name__ == '__main__':
    # root = os.path.join('.') #
    root = os.path.join('/rhome/yhu/bigdata/proj/experiment_G3DM')
    configuration_src_path = os.path.join(root, 'data')
    configuration_name = 'config_train.json'
    with open(os.path.join(configuration_src_path, configuration_name)) as f:
        config_data = json.load(f)

    cool_file = config_data['cool_file']
    cell = cool_file.split('.')[0]
    hyper = '_'.join([cool_file.split('.')[1], config_data["id"]])

    # '/rhome/yhu/bigdata/proj/experiment_G3DM'
    root = root
    cool_data_path = os.path.join( root, 'data', 'raw')
    graph_path = os.path.join( root, 'data', cell, hyper, 'graph')
    feature_path = os.path.join( root, 'data', cell, hyper, 'feature')
    dataset_path = os.path.join( root, 'data', cell, hyper)
    dataset_name = 'dataset.pt'
    output_path = os.path.join( root, 'data', cell, hyper, 'output')

    all_chromosome = config_data['all_chromosomes']
    test_chromosomes = config_data['test_chromosomes']

    num_clusters = config_data['parameter']['graph']['num_clusters']
    max_len = config_data['parameter']['graph']['max_len']
    cutoff_clusters_limits = config_data['parameter']['graph']['cutoff_clusters']
    cutoff_cluster = config_data['parameter']['graph']['cutoff_cluster']

    dim = config_data['parameter']['feature']['in_dim']

    saved_model_path = config_data['saved_model']['path']
    saved_model_name = config_data['saved_model']['name']
    section_start = int(config_data['parameter']['start'])
    section_end = int(config_data['parameter']['end'])

    # prepare dataset
    for chromosome in all_chromosome:
        create_data(num_clusters, chromosome, dim,
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

    # load dataset
    print('load dataset: {}'.format(os.path.join( dataset_path, dataset_name)))
    HiCDataset = load_dataset(dataset_path, dataset_name)

    test_indices = np.array(HiCDataset.test_list)
    test_dataset = torch.utils.data.Subset(HiCDataset, test_indices)

    # creat network model
    em_networks, ae_networks, num_heads, num_clusters, _, _, _ = create_network(config_data, device)
   
    # predict
    model = [em_networks, ae_networks]
    predictions = run_prediction(test_dataset, model, saved_parameters_model, num_heads, num_clusters, device='cpu')

    

