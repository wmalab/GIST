import os, sys, pickle, json
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from .utils import *
from visualize import load_data

def select_structure3d(data3d, index):
    """data3d: (N, K, 3) """
    N = data3d.shape[0]
    K = data3d.shape[1]
    M = len(index)
    res = torch.empty((M, K, 3))
    for i, idx in enumerate(index):
        res[i, :, :] = torch.mean( data3d[idx, :, 3], dim=0, keepdim=True)
    
    res = torch.transpose(res, 0, 1)
    return res.cpu().numpy()

if __name__ == '__main__':
    # root = '../'
    # root = os.path.join('/rhome/yhu/bigdata/proj/experiment_G3DM')

    # load config .json
    configuration_path = os.path.join(root, 'data')
    chrom = '21'
    configuration_name = 'config_predict_{}.json'.format(chrom)

    info, config_data = load_data.load_configuration(configuration_path, configuration_name)


    # load prediction
    prediction_path = os.path.join(root, 'data', info['cell'], info['hyper'], 'output')
    prediction_name = 'prediction.pkl'

    prediction = load_data.load_prediction(prediction_path, prediction_name)

    prediction = load_prediction(prediction_path, prediction_name)
    structure = prediction['{}_0'.format(chrom)]['structures']