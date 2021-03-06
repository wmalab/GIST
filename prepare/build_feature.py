import os, sys, shutil
import json, pickle
import numpy as np
import torch
import math

from .utils import log1p_hic

import warnings
warnings.filterwarnings('ignore')
 
def create_feature(norm_hic, dim):
    ''' create Hi-C feature '''
    log_hic = log1p_hic(norm_hic)
    n_idx = np.arange(log_hic.shape[0]).flatten()
    # n_idx = np.sort(np.argwhere(np.sum(log_hic, axis=0)!=0)).flatten()

    remove_hic = log_hic[n_idx,:]
    remove_hic = remove_hic[:, n_idx]

    # dim can't larger than int(x.shape[0]/2)-1 !
    features = feature_hic(remove_hic, check_dim(dim, remove_hic))

    mean_fs = np.nanmean(features[n_idx.flatten(),:],axis=0)
    for i in np.arange(features.shape[0]):
        for j in np.arange(features.shape[1]):
            features[i, j] = mean_fs[j] if features[i,j]==0 else features[i,j] 

    pe = position_hic(features, features.shape[1], idx=n_idx, scale=10)
    positions = np.array(pe)

    # features = torch.tensor(features)
    # positions = torch.tensor(positions)

    f_dict = {'feat':features, 'pos': positions}
    # save_feature(output_path, output_file, f_dict)
    return f_dict

def check_dim(dim, x):
    assert dim <= int(x.shape[0]/2)-1
    return int(dim)

def feature_hic(hic, dim):
    t_hic = tilt_hic(hic, dim)
    # t_hic, p_hic = position_hic(t_hic, dim)
    return t_hic

def tilt_hic(hic, dim):
    featrues = np.zeros((hic.shape[0], dim))
    for i in np.arange(hic.shape[0]):
        for l in np.arange(1, dim+1):
            if i-l >= 0 and i+l < hic.shape[1]:
                featrues[i,l-1] = max(hic[i, i+l], hic[i, i-l])
            else:
                if i-l < 0:
                    featrues[i,l-1] = hic[i, i+l]
                if i+l >= hic.shape[1]:
                    featrues[i,l-1] = hic[i, i-l]
    return featrues

def position_hic(hic_feat, dim, idx=None, scale=1):
    max_seq_len, d_model = hic_feat.shape[0], dim
    pe = np.zeros((max_seq_len, d_model))
    pos = np.arange(max_seq_len)

    step = pos if idx is None else idx[pos]
    # print(step.shape)
    for i in range(0, d_model-1, 2):
        iarry = np.ones(len(pos), dtype=int)*i
        pe[pos, iarry] = [math.sin( scale*x / (10000 ** ((2 * i)/d_model))) for x in step]
        pe[pos, iarry + 1] = [ math.cos( scale*x / (10000 ** ((2 * (i + 1))/d_model))) for x in step]
    x = hic_feat * math.sqrt(d_model)
    #add constant to embedding
    seq_len = x.shape[1]
    # x = x + pe[:,:seq_len]
    return pe[:,:seq_len]


def save_feature(path, file, feature_dict):
    file = file+'.pkl'
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, file), 'wb') as f:
        pickle.dump(feature_dict, f, pickle.HIGHEST_PROTOCOL)

def load_feature(path, file):
    file = file+'.pkl'
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, file), 'rb') as f:
        return pickle.load(f)