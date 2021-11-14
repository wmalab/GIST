import sys, os
import cooler
import numpy as np
from iced import normalization
from scipy.stats import percentileofscore
from sklearn import mixture
import matplotlib.pyplot as plt


from matplotlib.ticker import PercentFormatter

from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans

def cluster_gmm(data, low, high, num_cluster):
    tmp = data # [data>1e-4]
    l = np.percentile(tmp, low)
    h = np.percentile(tmp, high)
    X = data[((data>l)&(data<h))].reshape(-1,1)
    gmm = GMM(num_cluster, covariance_type='full', init_params='kmeans')
    m = gmm.fit(X)
    # D = np.clip(data, a_min=X.min()*0.8, a_max=X.max()*1.2)
    # print(l, h, X.min()*0.8, X.max()*1.2)
    return m, X

def save_aic_bic(X, model, path, name):
    data = X.reshape(-1,1)
    aic_value = model.aic(data)
    bic_value = model.bic(data)
    file = os.path.join(path, 'aic')
    os.makedirs(file, exist_ok=True)
    file = os.path.join(file, name)
    line = "{}".format(aic_value)
    with open(file,'w') as fout:
        fout.write(line)
        fout.close()

    file = os.path.join(path, 'bic')
    os.makedirs(file, exist_ok=True)
    file = os.path.join(file, name)
    line = "{}".format(bic_value)
    with open(file, 'w') as fout:
        fout.write(line)
        fout.close()

def run(data, low, high, num_cluster, path):
    m, X = cluster_gmm(data, low, high, num_cluster)
    name = "cutoff_{}_{}.txt".format(int(low), num_cluster)
    save_aic_bic(X, m, path, name)

def remove_nan_col(hic):
    hic = np.nan_to_num(hic)
    col_sum = np.sum(hic, axis=1)
    idxy = np.array(np.where(col_sum>0)).flatten()
    mat = hic[idxy, :]
    mat = mat[:, idxy]
    return mat, idxy

if __name__ == '__main__':
    path = '/rhome/yhu/bigdata/proj/experiment_G3DM/data/raw'
    # name = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    name = 'Rao2014-IMR90-MboI-allreps-filtered.10kb.cool'
    cell = name.split('.')[0]
    resolu = name.split('.')[1]
    chromosome = str(sys.argv[1])
    chro = 'chr{}'.format(chromosome)
    file = os.path.join(path, name)
    c = cooler.Cooler(file)
    resolution = c.binsize
    raw_hic = c.matrix(balance=True).fetch(chro)
    norm_hic = normalization.ICE_normalization(raw_hic) 
    norm_hic = np.array(norm_hic)

    raw_hic, idxy = remove_nan_col(raw_hic)
    norm_hic, _ = remove_nan_col(norm_hic)

    log1p_rhic = np.log1p(raw_hic) + 1e-4
    log1p_nhic = np.log1p(norm_hic) + 1e-4
    
    # log1p_rhic = log1p_rhic/log1p_rhic.max() + 1e-4
    # log1p_nhic = log1p_nhic/log1p_nhic.max() + 1e-4

    fig, ax = plt.subplots(1, 2)
    im = ax[0].imshow(log1p_rhic, cmap='RdBu_r', interpolation='nearest')
    fig.colorbar(im, ax=ax[0])
    im = ax[1].imshow(log1p_nhic, cmap='RdBu_r', interpolation='nearest')
    fig.colorbar(im, ax=ax[1])
    save_path = os.path.join('/rhome/yhu', 'bigdata', 'proj', 'experiment_G3DM', 'figures', 'gmm_parameter')
    os.makedirs(save_path, exist_ok=True)
    title = 'chr{}'.format(chromosome)
    plt.savefig(os.path.join(save_path, '{}.pdf'.format(title)), format='pdf')
    plt.close()

    low = float(sys.argv[2]) # 0 5, 10, 15, 20, 25, 30, 35, 40
    high = float(100)
    num_cluster = int(sys.argv[3]) # 3 - 20
    save_path = os.path.join('/rhome/yhu', 'bigdata', 'proj', 'experiment_G3DM', 'figures', 'gmm_parameter', cell, resolu, 'chr{}'.format(chromosome))
    run(log1p_nhic, low, high, num_cluster, save_path)