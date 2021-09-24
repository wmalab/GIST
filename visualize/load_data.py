import os, sys, pickle, json
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from prepare.utils import load_hic, load_hic, iced_normalization
from prepare.build_dataset import HiCDataset, load_dataset

"""
convert data to ndarray/dataframe before display

hic from .cool file
index from dataset: [featrues, postion], graph
The order of nodes from graph: g.nodes['bead'].data['id']
prediction[index] = {'structures': pred_X, 
                    'structures_weights': weights,
                    'predict_cluster': [pred_dist_cluster_mat, pdcm_list], 
                    'true_cluster': true_cluster_mat}
prediction['mixture model'] = dis_gmm
'structures_weights': the propotion of structures in the set.
'mixtrue model': from torch.distributions, using torch.save/torch.load
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cooler
from iced import normalization

def load_configuration(path, name):
    with open(os.path.join(path, name)) as f:
        config_data = json.load(f)
        f.close()
    # predict one chromosome per prediction
    cfile = config_data['cool_file']
    cell = cfile.split('.')[0]
    hyper = '_'.join([cool_file.split('.')[1], config_data["id"]])
    chromosome = config_data['test_chromosomes'][0]
    section_start = int(config_data['parameter']['section']['start'])
    section_end = int(config_data['parameter']['section']['end'])
    resolution = int(config_data['resolution'])
    num_clusters = int(config_data['parameter']['graph']['num_clusters']) 
    num_heads = int(config_data['parameter']['G3DM']['num_heads'])
    info = dict()
    info['chromosome'] = chromosome
    info['start'] = section_start
    info['end'] = section_end
    info['resolution'] = resolution
    info['ncluster'] = num_clusters
    info['nhead'] = num_heads
    info['cool_file'] = cfile
    info['cell'] = cell
    info['hyper'] = hyper
    return info, config_data

def load_prediction(path, name):
    file = os.path.join(path, name)
    with open(file, 'rb') as handle:
        res = torch.load(file, map_location=torch.device('cpu'))
        return res


if __name__ == '__main__':
    root = '../'
    # root = os.path.join('/rhome/yhu/bigdata/proj/experiment_G3DM')

    # load config .json
    configuration_path = '/Users/huyangyang/Desktop/chromosome_3D/'
    configuration_name = 'config_predict.json'

    info, config_data = load_configuration(configuration_path, configuration_name)

    # load dataset
    dataset_path = '/Users/huyangyang/Desktop/chromosome_3D/data/'
    dataset_name = 'dataset.pt'

    HD = load_dataset(dataset_path, dataset_name)
    graph, feat, ratio, indx  = HD[0]

    # load prediction
    prediction_path = '/Users/huyangyang/Desktop/chromosome_3D/data'
    prediction_name = 'prediction.pkl'

    prediction = load_prediction(prediction_path, prediction_name)
    
    # load .cool
    cool_file = info['cool_file']
    cool_data_path = os.path.join(root, 'data', 'raw')
    file = os.path.join(cool_data_path, cool_file)
    raw_hic, _, cooler = load_hic( rawfile=file, chromosome='chr{}'.format(str(info['chromosome'])))
    norm_hic = iced_normalization(raw_hic)