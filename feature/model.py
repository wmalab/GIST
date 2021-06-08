"""import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class embedding(nn.Module):
    '''in_dim, out_dim'''
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
        return X"""