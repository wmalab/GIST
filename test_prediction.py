import time, os, json, pickle
import dgl, torch 
from torch.utils import tensorboard
import GPUtil

import numpy as np
from G3DM.fit import load_dataset, create_network, setup_train, run_prediction
from G3DM.model import save_model_state_dict

from prepare.build_data import create_predict_data
from prepare.build_feature import load_feature
from prepare.build_dataset import HiCDataset

from prepare.utils import load_graph

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
    configuration_name = 'config_predict.json'
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
    output_name = 'prediction.pkl'

    all_chromosome = config_data['all_chromosomes']
    test_chromosomes = config_data['test_chromosomes']

    num_clusters = config_data['parameter']['graph']['num_clusters']
    cutoff_clusters_limits = config_data['parameter']['graph']['cutoff_clusters']
    cutoff_cluster = config_data['parameter']['graph']['cutoff_cluster']

    dim = config_data['parameter']['feature']['in_dim']

    model_saved_path = config_data['saved_model']['path']
    model_saved_name = config_data['saved_model']['name']
    resolution = int(config_data['resolution'])
    section_start = int(config_data['parameter']['section']['start'])
    section_end = int(config_data['parameter']['section']['end'])

    section_start = np.floor(section_start/resolution)
    section_end = np.floor(section_end/resolution)

    os.makedirs(graph_path, exist_ok=True)
    os.makedirs(feature_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # prepare dataset
    for chromosome in all_chromosome:
        create_predict_data(num_clusters, chromosome, dim,
                            cutoff_clusters_limits, 
                            cutoff_cluster, 
                            [section_start, section_end],
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

    # save dataset
    HD = HiCDataset(graph_dict, feature_dict, cluster_weight_dict, dataset_path, dataset_name)
    torch.save(HD, os.path.join( dataset_path, dataset_name))

    # load dataset
    print('load dataset: {}'.format(os.path.join( dataset_path, dataset_name)))
    HiCDataset = load_dataset(dataset_path, dataset_name)

    test_indices = np.array(HiCDataset.list)
    test_dataset = torch.utils.data.Subset(HiCDataset, test_indices)

    # creat network model
    em_networks, ae_networks, num_heads, num_clusters, _, _, _ = create_network(config_data, device)
    
    # load parameters
    path = os.path.join(model_saved_path, model_saved_name)
    checkpoint = torch.load(path, map_location=device)
    em_networks[0].load_state_dict(checkpoint['embedding_model_state_dict'])
    ae_networks[0].load_state_dict(checkpoint['encoder_model_state_dict'])
    ae_networks[1].load_state_dict(checkpoint['decoder_distance_model_state_dict'])
    ae_networks[2].load_state_dict(checkpoint['decoder_gmm_model_state_dict'])
    ae_networks[3].load_state_dict(checkpoint['decoder_euclidean_model_state_dict'])
    ae_networks[4].load_state_dict(checkpoint['decoder_simlarity_model_state_dict'])
    # optimizer[0].load_state_dict(checkpoint['optimizer_state_dict'])

    # predict
    model = [em_networks, ae_networks]
    predictions = run_prediction(test_dataset, model, [model_saved_path, model_saved_name], num_heads, num_clusters, device=device)
    print(predictions)

    # os.makedirs(output_path, exist_ok=True)
    # file = os.path.join(output_path, output_name)
    # with open(file, 'wb') as handle:
    #     pickle.dump(predictions, handle)

    # with open('file.txt', 'rb') as handle:
    #     b = pickle.loads(handle.read())
