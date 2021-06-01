import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

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
    x = x + pe[:,:seq_len]
    return x

class embedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(embedding, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim, bias=True)
        self.fc2 = nn.Linear(out_dim, out_dim, bias=True)

        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        nn.init.xavier_normal_(self.fc1.weight, gain=gain)
        nn.init.xavier_normal_(self.fc2.weight, gain=gain)

    def forward(self, h):
        X = self.fc1(h)
        X = torch.nn.functional.leaky_relu(X)
        X = self.fc2(X)
        X = torch.nn.functional.normalize(X, p=2.0, dim=1)
        # X = torch.nn.functional.leaky_relu(X)
        return X