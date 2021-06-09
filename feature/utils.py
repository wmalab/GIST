import numpy as np
import math
import os, pickle

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

def position_hic(hic_feat, dim):
    max_seq_len, d_model = hic_feat.shape[0], dim
    pe = np.zeros((max_seq_len, d_model))
    for pos in range(max_seq_len):
        for i in range(0, d_model-1, 2):
            pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
            pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
    x = hic_feat * math.sqrt(d_model)
    #add constant to embedding
    seq_len = x.shape[1]
    # x = x + pe[:,:seq_len]
    return x, pe[:,:seq_len]


def save_feature(path, file, feature_dict):
    with open(os.path.join(path, file), 'wb') as f:
        pickle.dump(feature_dict, f, pickle.HIGHEST_PROTOCOL)

def load_feature(path, file):
    with open(os.path.join(path, file), 'rb') as f:
        return pickle.load(f)