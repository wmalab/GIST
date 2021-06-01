from __future__ import print_function, division

import os, sys

import multiprocessing as mp
import torch
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from skimage import measure

from iced import normalization

from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

from .utils import load_hic, log1p_hic, remove_nan_col
from .build_graph import create_hierarchical_graph_1lvl

import warnings
warnings.filterwarnings('ignore')

def block_reduce(raw_hic, wl, reduce_fun):
    hic = measure.block_reduce(raw_hic, (wl, wl), reduce_fun)
    return hic

def iced_normalization(raw_hic):
    hic = normalization.ICE_normalization(raw_hic)
    return hic

def hic_prepare_block_reduce(rawfile, chromosome, ratios, reduce_fun=np.mean, remove_zero_col = True):
    raw_hic, resolution, cooler = load_hic(rawfile, chromosome = chromosome)
    hics, norm_hics = [], []
    win_len = [1]
    raw_hic = np.nan_to_num(raw_hic)
    if remove_zero_col:
        raw_hic, idxy = remove_nan_col(raw_hic)
    for r in ratios[1:]:
        win_len.append(win_len[-1]*r)

    for wl in win_len:
        hics.append( block_reduce(raw_hic, wl, reduce_fun) )

    for h in hics:
        norm_hics.append(iced_normalization(h))
    return norm_hics, ratios

def hic_prepare_pooling(rawfile, chromosome, ratios, strides, remove_zero_col = True):
    raw_hic, resolution, cooler = load_hic(rawfile, chromosome = chromosome)
    raw_hic = np.nan_to_num(raw_hic)
    raw_hic = torch.tensor(raw_hic).float()
    n = raw_hic.shape[0]
    hics, norm_hics = [], []
    if remove_zero_col:
        raw_hic, idxy = remove_nan_col(raw_hic)

    for i, wl in enumerate(ratios):
        pool1d = torch.nn.AvgPool2d(kernel_size=(wl,wl), stride=strides[i], padding=int(wl/2), count_include_pad=False)
        m = pool1d(raw_hic.view(1,1,n,n))
        m = np.array(torch.squeeze(m))
        m = (m+np.transpose(m))/2
        np.fill_diagonal(m, 0)
        hics.append( m )

    for h in hics:
        m = iced_normalization(h)
        norm_hics.append(m)
    return norm_hics, ratios

def graphs_save(save_path, name, graphs):
    graph_path = os.path.join(save_path, name + '_dgl_graph.bin')
    # print(graph_path)
    save_graphs(graph_path, graphs, {'label':torch.tensor(0), 'ratio':torch.tensor((1,2,2))})
    
    # save other information in python dict
    # info_path = os.path.join(save_path, name + '_info.pkl')
    # save_info(info_path, {norm_hics, nclusters, ratios, mat_hics, mat_probs})


if __name__ == '__main__':
    '''norm_hics, ratios = hic_prepare_block_reduce(
        rawfile='Dixon2012-H1hESC-HindIII-allreps-filtered.500kb.cool', 
        chromosome='chr20', ratios=[1])

    for i, m in enumerate(norm_hics):
        print('level id: {}, iced normalization Hi-C shape: {}'.format(i, m.shape))

    graph = create_hierarchical_graph_1lvl(norm_hics[0], percentile=10)
    # graphs_save('.', 'demo', [graph])'''

    norm_hics, ratios = hic_prepare_pooling(
        rawfile='Dixon2012-H1hESC-HindIII-allreps-filtered.500kb.cool', 
        chromosome='chr20', ratios=[1,2,2], strides=[1,1,1], remove_zero_col=False)

    for i, m in enumerate(norm_hics):
        print('level id: {}, iced normalization Hi-C shape: {}'.format(i, m.shape))

    # graph = create_hierarchical_graph_1lvl(norm_hics[0], percentile=10)
    # graphs_save('.', 'demo', [graph])