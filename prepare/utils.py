import os
import cooler
import numpy as np
from numpy import inf
from scipy.stats import percentileofscore
from sklearn import mixture

import torch
import dgl

def save_graph(g_list, output_path, output_file):
    ''' g_list = [ top_graph, top_subgraphs, bottom_graph, inter_graph ] '''
    dgl.data.utils.save_graphs(os.path.join(output_path, output_file), g_list )

def load_graph(output_path, output_file):
    ''' g_list = [ top_graph, top_subgraphs, bottom_graph, inter_graph ] '''
    g_list, _ = dgl.data.utils.load_graphs(os.path.join(output_path, output_file))
    return g_list

def load_hic(name='Dixon2012-H1hESC-HindIII-allreps-filtered.500kb.cool', chromosome='chr1'):
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
    print('shape of HiC: {}'.format(HiC.shape))
    return HiC

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
    fitdata = torch.tensor(data).flatten()
    y_, ypb_, idx_nonzeros = _gmm(H, fitdata, n_cluster=n_cluster-1, order='D')
    np.set_printoptions(precision=2, suppress=True)
    mat_, matpb_ = _gmm_matrix(y_.int(), ypb_, idx_nonzeros, n_cluster, (mat_len, mat_len))
    return mat_, matpb_

def _gmm(fitX, X, n_cluster=20, idx_nonzeros=None, order='I'):  # 'I': increasing; 'D': descreasing
    X = X.reshape((-1, 1))
    fitX = fitX.reshape((-1, 1))
    if idx_nonzeros is None:
        idx_nonzeros = torch.nonzero(X.flatten(), as_tuple=False).flatten()
        X = X[idx_nonzeros]
        fitidx_nonzeros = torch.nonzero(fitX.flatten(), as_tuple=False).flatten()
        fitX = fitX[fitidx_nonzeros]

    # cl, c = KMeans(X, n_cluster)
    gmm = mixture.GaussianMixture(
        n_components=n_cluster, covariance_type='full', init_params='kmeans')
    gmm.fit(fitX)
    cluster_ids_x = gmm.predict(X)
    cluster_centers = torch.tensor(gmm.means_)
    pb = gmm.predict_proba(X)
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