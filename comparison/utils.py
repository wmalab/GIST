from __future__ import print_function, division
import os, sys
import cooler
from iced import normalization

import numpy as np

import warnings
warnings.filterwarnings('ignore')

def load_hic(name, chromosome):
    # data from ftp://cooler.csail.mit.edu/coolers/hg19/
    #name = 'Rao2014-K562-MboI-allreps-filtered.500kb.cool'
    c = cooler.Cooler(name)
    resolution = c.binsize
    mat = c.matrix(balance=True).fetch(chromosome)
    return mat, resolution, c

def iced_normalization(raw_hic):
    hic = normalization.ICE_normalization(raw_hic)
    return hic

def scn_normalization(raw_hic):
    hic = normalization.SCN_normalization(raw_hic)
    return hic

def remove_nan_col(hic):
    hic = np.nan_to_num(hic)
    col_sum = np.sum(hic, axis=1)
    idxy = np.array(np.where(col_sum>0)).flatten()
    mat = hic[idxy, :]
    mat = mat[:, idxy]
    return mat, idxy