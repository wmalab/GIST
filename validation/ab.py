'''
1. normalize by mean of each diagonal
- args: mat: numpy.ndarray
norm = normalizebydistance(mat)

2. calculate correlation between each column
corr = correlation(norm, method='pearson', center=True)

3. perform eigenvalue decomposition
- args: nc: number of eigenvectors returned
pc = decomposition(corr, method='eig', nc=nc)

4. plot AB compartment
fig = plot(corr, pc)
fig.savefig(...)
'''

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.linalg import eig
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform


def fill_diagonal(mat, k, val):
    '''
    Fill the k-th diagonal of the given 2-d square array.
    :param mat: array.
    :param k: int.
        if positive, above the main diagonal,
        else, below the main diagonal.
    :param val: scalar.
    '''
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("mat should be a 2-d square array.")
    n = mat.shape[1]
    if abs(k) > n - 1:
        raise ValueError("k should not larger than n-1.")
    if k >= 0:
        start = k
        end = n * n - k * n
    else:
        start = (-k) * n
        end = n * n + k
    step = n + 1
    mat.flat[start:end:step] = val


def normalizebydistance_(mat):
    x = np.array(mat, copy=True, dtype=float)
    n = x.shape[0]
    margin = x.sum(axis=0)
    # fill diagonal with np.nan
    np.fill_diagonal(x, np.nan)
    # fill the first diagonal with np.nan if all zeros
    if np.nansum(np.diagonal(x, offset=1)) == 0:
        fill_diagonal(x, k=1, val=np.nan)
        fill_diagonal(x, k=-1, val=np.nan)
    # fill row/col with np.nan if all zeros
    x[margin==0, :] = np.nan
    x[:, margin==0] = np.nan
    diagmean = np.zeros(n, dtype=float)
    
    for d in range(1, n):
        diag = np.diagonal(x, offset=d)
        m = np.nanmean(diag)
        if m > 0 or np.isnan(m):
            diagmean[d] = m
        else:
            diagmean[d] = np.inf

    for i in range(n):
        for j in range(i+1, n):
            x[i, j] = x[i, j] / diagmean[abs(i-j)]
            x[j, i] = x[i, j]
    
    return x

def fit_genomic_spatial_func(x, a, b):
    return (x**b)/a

def normalizebydistance(mat, genomic_index=None):
    # mtype='3d' or 'fish'
    if genomic_index is None:
        return normalizebydistance_(mat)
    
    gen_idx = np.zeros(len(genomic_index))
    for i, index in enumerate(genomic_index):
        gen_idx[i] = np.nanmean(index)
    genomic_dis = pdist(gen_idx.reshape(-1,1))

    msku = np.zeros_like(mat)
    msku[np.triu_indices_from(msku, k=1)] = True
    triu_mat = mat[msku.astype(bool)].flatten()

    popt, pcov = curve_fit(fit_genomic_spatial_func, genomic_dis, triu_mat) 
    print('power law parameters: ', popt)
    a = fit_genomic_spatial_func(genomic_dis, *popt)
    
    expected_triu = squareform(a)
    np.fill_diagonal(expected_triu, 1)
    res = mat.astype(float)/expected_triu.astype(float)
    return res, popt

def centering(mat):
    x = np.array(mat, copy=True, dtype=float)
    n = x.shape[0]
    # substract row mean from each row
    for i in range(n):
        m = np.nanmean(x[i, :])
        x[i, :] = x[i, :] - m
    return x

def correlation(mat, method='pearson', center=True):
    x = np.array(mat, copy=True, dtype=float)
    n = x.shape[0]
    # substract row mean from each row
    if center:
        for i in range(n):
            m = np.nanmean(x[i, :])
            x[i, :] = x[i, :] - m
    corr = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            nas = np.logical_or(np.isnan(x[:, i]), np.isnan(x[:, j]))
            if np.all(~nas) == False:
                corr[i,j] = np.nan
                corr[j,i] = np.nan
                continue

            if method == 'spearman':
                corr[i, j], _ = spearmanr(x[~nas, i], x[~nas, j])
            else:
                print(i, j, ~nas)
                corr[i, j], _ = pearsonr(x[~nas, i], x[~nas, j])
            corr[j, i] = corr[i, j]
    # keep NAs
    # corr[np.isnan(corr)] = 0
    return corr


def decomposition(mat, method='eig', nc=2):
    # remove row/col with NAs
    n = mat.shape[0]
    nas = (~np.isnan(mat)).sum(axis=0) == 0
    if method == 'eig':
        _, v = eig(mat[~nas, :][:, ~nas])
        eigenvec = np.full((n, nc), np.nan, dtype=float)
        eigenvec[~nas, :] = v[:, :nc]
        return eigenvec
    else:
        pca = PCA(n_components=nc)
        # nas = np.isnan(mat).sum(axis=0) > 0
        pc = pca.fit_transform(mat[~nas, :][:, ~nas])
        eigenvec = np.full((n, nc), np.nan, dtype=float)
        eigenvec[~nas, :] = pc 
        return eigenvec 


def plot(mat, pc, title=None, start=0, locs=None):
    fig = plt.figure(figsize=(5, 5))
    axmatrix = plt.subplot(111)
    n, nc = pc.shape
    im = axmatrix.matshow(mat, vmin=-1, vmax=1, cmap='bwr')
    if title is not None:
        plt.title(title)
    divider = make_axes_locatable(axmatrix)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    for i in range(nc):
        cax = divider.append_axes("bottom", size="6%", pad=0.25)
        cax.set_xlim([start, start+n])
        cax.set_yticks([0])
        rg = np.arange(start, start+n)
        pos = pc[:, i] > 0
        neg = pc[:, i] < 0
        cax.bar(rg[pos], pc[pos, i], color='red', width=1)
        cax.bar(rg[neg], pc[neg, i], color='blue', width=1)
        if locs is not None:
            for loc in locs:
                if loc[0] >= start and loc[0] < start+n:
                    cax.axvspan(loc[0], min(loc[1]+1, start+n-1), facecolor='black', alpha=0.3)
    return fig