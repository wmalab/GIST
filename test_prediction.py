import time
import os
import json
import dgl
import torch
from torch.utils import tensorboard
import GPUtil

import numpy as np
from G3DM.fit import load_dataset, create_network, setup_train, run_epoch, inference
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
    hyper = '_'.join([cool_file.split('.')[1], 'id', config_data["id"]])

    # '/rhome/yhu/bigdata/proj/experiment_G3DM'
    root = config_data['root'] if config_data['root'] else root
    cool_data_path = config_data['cool_data_path'] if config_data['cool_data_path'] else os.path.join( root, 'data', 'raw')
    graph_path = config_data['graph_path'] if config_data['graph_path'] else os.path.join( root, 'data', cell, hyper, 'graph')
    feature_path = config_data['feature_path'] if config_data['feature_path'] else os.path.join( root, 'data', cell, hyper, 'feature')
    dataset_path = config_data['dataset_path']['path'] if config_data['dataset_path']['path'] else os.path.join( root, 'data', cell, hyper)
    dataset_name = config_data['dataset_path']['name'] if config_data['dataset_path']['name'] else 'dataset.pt'
    output_path = config_data['output_path'] if config_data['output_path'] else os.path.join( root, 'data', cell, hyper, 'output')

    saved_model_path = config_data['saved_model']['path'] if config_data['saved_model']['path'] else os.path.join( root, 'data', cell, hyper, 'saved_model')
    saved_model_name = config_data['saved_model']['name'] if config_data['saved_model']['name'] else 'model_net'

    all_chromosome = config_data['all_chromosomes']
    train_chromosomes = config_data['train_valid_chromosomes']
    test_chromosomes = config_data['test_chromosomes']


    # load dataset
    print('load dataset: {}'.format(os.path.join( dataset_path, dataset_name)))
    HiCDataset = load_dataset(dataset_path, dataset_name)
    train_valid_indices = np.array(HiCDataset.train_list)
    test_indices = np.array(HiCDataset.test_list)

    train_size = int(0.7 * len(train_valid_indices))
    valid_size = len(train_valid_indices) - train_size
    dataset = torch.utils.data.Subset(HiCDataset, train_valid_indices)

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    test_dataset = torch.utils.data.Subset(HiCDataset, test_indices)

    # train_HiCDataset = torch.utils.data.DataLoader(HiCDataset[train_indices], shuffle=True).dataset
    # graphs, features, label, _ = HiCDataset[0]

    # creat network model
    em_networks, ae_networks, nll, opt, scheduler = create_network(config_data, device)

    #save init model
    models_dict = {
        'embedding_model': em_networks[0],
        'encoder_model': ae_networks[0],
        'decoder_distance_model': ae_networks[1],
        'decoder_gmm_model': ae_networks[2],
    }
    os.makedirs(saved_model_path, exist_ok=True)
    path = os.path.join(saved_model_path, 'ckpt_state_dict_' + saved_model_name)
    save_model_state_dict(models_dict, opt[0], path, 0, -1.0)

    # setup and call train
    itn = setup_train(config_data)
    log_fie = time.strftime("%Y%m%d-%H%M%S")
    log_dir = config_data['log_dir'] if config_data['log_dir'] else os.path.join( root, 'log', cell, hyper, log_fie)
    os.makedirs(log_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(log_dir)
    
    run_epoch([train_dataset, valid_dataset], [em_networks, ae_networks],
              nll, opt, scheduler, itn, device, writer=writer, config=config_data, saved_model=[saved_model_path, saved_model_name])
    writer.close()