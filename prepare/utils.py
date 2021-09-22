from __future__ import print_function, division

import os, sys
import numpy as np
from numpy import inf
# import multiprocessing as mp

import torch
from torch.utils.data import Dataset, DataLoader

import cooler
from iced import normalization

from scipy.stats import percentileofscore
from sklearn import mixture
# from skimage import measure

import dgl
from dgl import save_graphs, load_graphs
# from dgl.data.utils import makedirs, save_info, load_info

import warnings
warnings.filterwarnings('ignore')

def save_graph(g_list, output_path, output_file):
    ''' g_list = [ top_graph, top_subgraphs] '''
    output_file = output_file + '.bin'
    dgl.data.utils.save_graphs(os.path.join(output_path, output_file), g_list )

def load_graph(path, file) -> dict() :
    ''' g_list = [ top_graph, top_subgraphs] '''
    if '.bin' not in file :
        file = file + '.bin'
    g_list, labels = dgl.data.utils.load_graphs(os.path.join(path, file))
    res = {'top_graph':g_list[0], 
            'top_subgraphs': g_list[1]
            }
    return res, labels

def load_hic(name='Dixon2012-H1hESC-HindIII-allreps-filtered.500kb.cool', chromosome='chr21'):
    # data from ftp://cooler.csail.mit.edu/coolers/hg19/
    #name = 'Rao2014-K562-MboI-allreps-filtered.500kb.cool'

    c = cooler.Cooler(name)
    resolution = c.binsize
    chro = chromosome
    mat = c.matrix(balance=True).fetch(chro)
    return mat, resolution, c

def log1p_hic(mat):
    HiC = np.log1p(mat)
    HiC = (HiC/np.max(HiC))
    # print('shape of HiC: {}'.format(HiC.shape))
    return np.array(HiC)

def remove_nan_col(hic):
    hic = np.nan_to_num(hic)
    col_sum = np.sum(hic, axis=1)
    idxy = np.array(np.where(col_sum>0)).flatten()
    mat = hic[idxy, :]
    mat = mat[:, idxy]
    return mat, idxy

# mat_ : (n,n)
# matpb_: (n, n, clusters)
def cluster_hic(data, fitdata, n_cluster=30):
    mat_len = data.shape[0]
    H = torch.tensor(data).flatten()
    fitdata = torch.tensor(fitdata).flatten()
    y_, ypb_, idx_nonzeros = _gmm(fitdata, H, n_cluster=n_cluster-1, order='D')
    np.set_printoptions(precision=2, suppress=True)
    mat_, matpb_ = _gmm_matrix(y_.int(), ypb_, idx_nonzeros, n_cluster, (mat_len, mat_len))
    return mat_, matpb_

def _gmm(fitX, X, n_cluster=20, idx_nonzeros=None, order='I'):  # 'I': increasing; 'D': descreasing
    if idx_nonzeros is None:
        idx_nonzeros = torch.nonzero(X.flatten(), as_tuple=False).flatten()
        X = X[idx_nonzeros]
        fitidx_nonzeros = torch.nonzero(fitX.flatten(), as_tuple=False).flatten()
        fitX = fitX[fitidx_nonzeros]

    # cl, c = KMeans(X, n_cluster)
    gmm = mixture.GaussianMixture(n_components=n_cluster, 
                                covariance_type='full', 
                                init_params='kmeans')
    gmm.fit(fitX.view(-1,1))
    cluster_ids_x = gmm.predict(X.view(-1,1))
    cluster_centers = torch.tensor(gmm.means_)

    tmp, _ = np.histogram(cluster_ids_x.flatten(),
                                    bins=np.arange(num_clusters),
                                    density=True)
    print(tmp)
    print('cluster_centers: ', cluster_centers)

    pb = gmm.predict_proba(X.view(-1,1))
    # cluster_ids_x, cluster_centers = kmeans(X=X, num_clusters=n_cluster, distance='euclidean')
    if order == 'I':
        idx = torch.squeeze(torch.argsort(
            cluster_centers.flatten(), descending=False))
    else:
        idx = torch.squeeze(torch.argsort(
            cluster_centers.flatten(), descending=True))
    # print(cluster_centers.flatten())
    # print(idx)
    # print(cluster_centers[idx].flatten())
    lut = torch.zeros_like(idx)
    lut[idx[:]] = torch.arange(n_cluster)
    Y = lut[cluster_ids_x]
    Ypb = pb[:, idx[:]]
    return Y, Ypb, idx_nonzeros.flatten()


def _gmm_matrix(labels, probability, idx, n_cluster, matrix_size):
    khop_m = torch.ones(matrix_size, dtype=torch.int)*(n_cluster-1)
    khop_m = torch.flatten(khop_m)
    khop_m[idx] = labels
    khop_m = torch.reshape(khop_m, matrix_size)

    khop_pba = np.zeros((matrix_size[0]*matrix_size[0], n_cluster))
    pba = np.hstack((probability, np.zeros((probability.shape[0], 1))))

    khop_pba[idx, :] = pba
    khop_pba[:, -1] = 1 - np.sum(khop_pba[:, 0:-1], axis=1)
    khop_pba = torch.tensor(khop_pba, dtype=torch.float)
    khop_pba = torch.reshape(
        khop_pba, (matrix_size[0], matrix_size[0], n_cluster))
    return khop_m, khop_pba


"""def block_reduce(raw_hic, wl, reduce_fun):
    hic = measure.block_reduce(raw_hic, (wl, wl), reduce_fun)
    return hic"""

def iced_normalization(raw_hic):
    hic = normalization.ICE_normalization(raw_hic)
    return hic

"""def hic_prepare_block_reduce(rawfile, chromosome, ratios, reduce_fun=np.mean, remove_zero_col = True):
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
    return norm_hics"""

def hic_prepare(rawfile, chromosome):
    raw_hic, resolution, cooler = load_hic(rawfile, chromosome = chromosome)
    # raw_hic = np.nan_to_num(raw_hic)
    # raw_hic = torch.tensor(raw_hic).float()
    # n = raw_hic.shape[0]

    # #smoothing
    # wl = 3
    # pool1d = torch.nn.AvgPool2d(kernel_size=(wl,wl), stride=1, padding=int(wl/2), count_include_pad=False)
    # m = pool1d(raw_hic.view(1,1,n,n))
    # m = np.array(torch.squeeze(m))
    # m = (m+np.transpose(m))/2
    np.fill_diagonal(raw_hic, 0)
    hic = raw_hic

    norm_hic = iced_normalization(hic)
    return norm_hic