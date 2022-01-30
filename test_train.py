import time
import os, sys
import json
import dgl
import torch
from torch.utils import tensorboard
import GPUtil

import numpy as np
from G3DM.fit import load_dataset, create_network, setup_train, run_epoch
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
    root = os.path.join('/rhome/yhu/bigdata/proj/experiment_GIST')
    configuration_src_path = os.path.join(root, 'data')
    configuration_name = sys.argv[1]# 'config_train.json'
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

    saved_model_path = os.path.join( root, 'data', cell, hyper, 'saved_model')
    saved_model_name = 'model_net'

    all_chromosome = config_data['all_chromosomes']
    train_chromosomes = config_data['train_valid_chromosomes']

    # load dataset
    print('load dataset: {}'.format(os.path.join( dataset_path, dataset_name)))
    HiCDataset = load_dataset(dataset_path, dataset_name)
    train_valid_indices = np.array(HiCDataset.list)

    train_size = int(0.8 * len(train_valid_indices))
    valid_size = len(train_valid_indices) - train_size
    dataset = torch.utils.data.Subset(HiCDataset, train_valid_indices)

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    # creat network model
    [em_networks, ae_networks, 
    num_heads, num_clusters, 
    loss_fc, opt, scheduler] = create_network(config_data, device)

    #save init model
    models_dict = {
        'embedding_model': em_networks[0],
        'encoder_model': ae_networks[0],
        'decoder_distance_model': ae_networks[1],
        'decoder_gmm_model': ae_networks[2],
        'decoder_euclidean_model': ae_networks[3],
        'decoder_similarity_model': ae_networks[4],
    }
    os.makedirs(saved_model_path, exist_ok=True)
    path = os.path.join(saved_model_path, 'ckpt_state_dict_' + saved_model_name)
    save_model_state_dict(models_dict, opt[0], path, 0, -1.0)

    # setup and call train
    itn, num_heads, num_clusters = setup_train(config_data)
    log_fie = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join( root, 'log', cell, hyper, log_fie)
    os.makedirs(log_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(log_dir)
    
    run_epoch([train_dataset, valid_dataset], [em_networks, ae_networks],
            num_heads=num_heads, num_clusters=num_clusters, 
            loss_fc=loss_fc, optimizer=opt, scheduler=scheduler, iterations=itn,
            device=device, writer=writer, saved_model=[saved_model_path, saved_model_name])
    writer.close()
