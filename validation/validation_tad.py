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
    res = numpy.empty((M, K, 3))
    for i, idx in enumerate(index):
        res[i, :, :] = np.nanmean( data3d[idx.astype(int), :, :], axis=0, keepdim=True)
    res = res.transpose( (0, 1) )
    return res

if __name__ == '__main__':
    pass