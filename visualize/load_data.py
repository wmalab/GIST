import os, sys, pickle
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from prepare.utils import load_hic
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
"""
import torch
from torch.utils.data import Dataset, DataLoader
import cooler
from iced import normalization

def load_configuration():
    pass

# def load_dataset(path, name):
#     '''graph_dict[chromosome] = {top_graph, top_subgraphs, bottom_graph, inter_graph}
#     feature_dict[chromosome] = {'feat', 'pos'} 'feat': hic features; 'pos': position features
#     HiCDataset[i]: graph[i], feature[i], cluster_weight[i], index[i]'''
#     HiCDataset = torch.load(os.path.join(path, name))
#     return HiCDataset

def load_prediction(path, name):
    file = os.path.join(path, name)
    with open(file, 'rb') as handle:
        # b = pickle.loads(handle.read())
        res = torch.load(file, map_location=torch.device('cpu'))
        return res


if __name__ == '__main__':
    # load dataset
    dataset_path = '/Users/huyangyang/Desktop/chromosome_3D/data/'
    dataset_name = 'dataset.pt'

    prediction_path = '/Users/huyangyang/Desktop/chromosome_3D/data'
    prediction_name = 'prediction.pkl'

    HD = load_dataset(dataset_path, dataset_name)
    graph, feat, ratio, indx  = HD[0]

    prediction = load_prediction(prediction_path, prediction_name)
    print(prediction.items())
