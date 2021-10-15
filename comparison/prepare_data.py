import sys, os
import numpy as np
from numpy import inf
from scipy.sparse import coo_matrix

from utils import load_hic, remove_nan_col
from utils import scn_normalization, iced_normalization
import warnings
warnings.filterwarnings('ignore')

def run(chromosome, method, raw_hic_path, name, path):
    file = os.path.join(raw_hic_path, name)
    mat, resolution, cool = load_hic(file, 'chr{}'.format(chromosome))

    if method=='pastis':
        prepare_pastis(chromosome, mat, resolution, path)
    elif method=='shrec3d':
        prepare_shrec3d(mat, resolution, path)
    elif method=='gem':
        prepare_gem(mat, resolution, path)

def prepare_shrec3d(mat, resolution, path):
    mat, idx = remove_nan_col(mat)
    nmat = scn_normalization(mat)
    name = 'norm_mat.txt'
    file = os.path.join(path, name)
    print('mat shape {}, file name: {}'.format(nmat.shape, name))
    np.savetxt(file, nmat, delimiter='\t') 
    return 

def prepare_gem(mat, resolution, path):
    mat, idx = remove_nan_col(mat)
    nmat = iced_normalization(mat)
    name = 'norm_mat.txt'
    file = os.path.join(path, name)
    print('mat shape {}, file name: {}'.format(nmat.shape, name))
    np.savetxt(file, nmat, delimiter='\t')

    name = 'loci.txt'
    file = os.path.join(path, name)
    print('loci shape {}, file name: {}'.format(idx.shape, name))
    np.savetxt(file, idx*resolution, delimiter='\t')



def prepare_pastis(chro, mat, resolution, path):
    iced_mat = iced_normalization(mat)
    nmat, idx = remove_nan_col(iced_mat)
    nmat = np.triu(nmat, k=1)
    row, col = np.where(nmat>1e-10)
    data = mat[row, col]
    n = len(idx)
    coo_mat = coo_matrix((data, (row, col)), shape=(n, n))

    input_path = os.path.join(path, 'input')
    config = os.path.join(path, 'config.ini')
    counts = os.path.join(input_path, 'counts.matrix')
    lengths = os.path.join(input_path, 'lengths.bed')
    output_path = os.path.join(path)

    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    pastis_config(config, counts, lengths)
    pastis_count(coo_mat, counts)
    pastis_bed(chro, resolution, idx, lengths)
    return 

def pastis_config(output_path, counts_path, lengths_path, norm=True):
    with open(output_path, 'w') as fout:
        lines = "[all]\noutput_name: structure\n verbose: 1\nmax_iter: 1000\ncounts: {}\nlengths: {}\nnormalize: {}".format( counts_path, lengths_path, norm)
        fout.write(lines)
        fout.close()

def pastis_count(coo_mat, output_path):
    x = coo_mat.row
    y = coo_mat.col
    data = coo_mat.data
    np.savetxt(output_path, (x, y, data), delimiter='\t')  

def pastis_bed(chro, resolution, idx, output_path):
    "chr01   1       10000   0"
    if (len(chro)==1) & (chro!='X'):
        chrom = '0'+chro
    else:
        chrom = chro
    with open(output_path, 'w') as fout:
        for i, index in enumerate(idx):
            start = int(index*resolution)
            end = int(start) + int(resolution)
            line =  "chr{}\t{}\t{}\t{}\n".format(chrom, int(start), int(end), int(i))
            fout.write(line)
        fout.close()
    return 

if __name__ == '__main__':
    raw_hic_path = '/rhome/yhu/bigdata/proj/experiment_G3DM/data/raw'
    name = 'Rao2014-IMR90-MboI-allreps-filtered.10kb.cool'
    chromosome = '22'
    cell = name.split('.')[0]
    resolution = name.split('.')[1]

    path = '/rhome/yhu/bigdata/proj/experiment_G3DM'

    method = 'pastis'
    mpath = os.path.join(path, 'comparison', method, cell, resolution, chromosome)
    os.makedirs(mpath, exist_ok=True)
    print('prepate data for method {} \n\tsaved in {}'.format(method, mpath))
    run(chromosome, method, raw_hic_path, name, mpath)

    method = 'shrec3d'
    mpath = os.path.join(path, 'comparison', method, cell, resolution, chromosome)
    os.makedirs(mpath, exist_ok=True)
    print('prepate data for method {} \n\tsaved in {}'.format(method, mpath))
    run(chromosome, method, raw_hic_path, name, mpath)

    method = 'gem'
    mpath = os.path.join(path, 'comparison', method, cell, resolution, chromosome)
    os.makedirs(mpath, exist_ok=True)
    print('prepate data for method {} \n\tsaved in {}'.format(method, mpath))
    run(chromosome, method, raw_hic_path, name, mpath)
