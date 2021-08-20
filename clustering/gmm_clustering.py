import sys, os
import cooler
import numpy as np
from iced import normalization
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
from sklearn import mixture

from matplotlib.ticker import PercentFormatter

from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans

def cluster_gmm(data, low, high, num_cluster):
    l = np.percentile(data, low)
    h = np.percentile(data, high)
    data = data[data>0]
    X = data[((data>l)&(data<h))].reshape(-1,1)
    gmm = GMM(num_cluster, covariance_type='full', init_params='kmeans')
    m = gmm.fit(X)
    return m

def save_aic_bic(X, model, path, name):
    data = X.shape(-1,1)
    aic_value = model.aic(data)
    bic_value = model.bic(data)
    file = os.path.join(path, 'aic')
    os.makedirs(file, exist_ok=True)
    os.path.join(file, name)
    line = "{}".format(aic_value)
    with open(file,'w') as fout:
        fout.write(line)
        fout.close()

    file = os.paht.join(path, 'bic')
    os.makedirs(file, exist_ok=True)
    os.path.join(file, name)
    line = "{}".format(bic_value)
    with open(file, 'w') as fout:
        fout.write(line)
        fout.close()

def run(data, low, high, num_cluster, path):
    m = cluster_gmm(data, low, high, num_cluster)
    name = "{}_{}.txt".format(low, num_cluster)
    save_aic_bic(data, m, path, name)

# def plot_hist():
#     fig, axs = plt.subplots()
#     # axs[1].imshow(log1p_nhic, cmap='RdBu_r', interpolation=None)
#     data = np.triu(log1p_nhic, k=1)
#     data = data[data>0].reshape(-1,1)
#     # data = data/data.max()
#     print(data.shape)
#     n_bins = 100
#     axs.hist(data, bins=n_bins)
#     # axs.yaxis.set_major_formatter(PercentFormatter(xmax=1))
#     plt.show()

def remove_nan_col(hic):
    hic = np.nan_to_num(hic)
    col_sum = np.sum(hic, axis=1)
    idxy = np.array(np.where(col_sum>0)).flatten()
    mat = hic[idxy, :]
    mat = mat[:, idxy]
    return mat, idxy

if __name__ == '__main__':
    path = '/rhome/yhu/bigdata/proj/experiment_G3DM/data/raw'
    name = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    chromosome = str(sys.argv[1])
    chro = 'chr{}'.format(chromosome)
    file = os.path.join(path, name)
    c = cooler.Cooler(file)
    resolution = c.binsize
    raw_hic = c.matrix(balance=True).fetch(chro)
    raw_hic, idxy = remove_nan_col(raw_hic)
    norm_hic = normalization.ICE_normalization(raw_hic) 
    norm_hic = np.array(norm_hic)

    log1p_rhic = np.log1p(raw_hic)
    log1p_nhic = np.log1p(norm_hic)

    low = float(sys.argv[2]) # 0 5, 10, 15, 20, 25, 30, 35, 40
    high = float(99.5)
    num_cluster = int(sys.argv[3]) # 3 - 20
    save_path = os.path.join('/rhome/yhu', 'bigdata', 'proj', 'experiment_G3DM', 'gmm_parameter')
    run(log1p_nhic, low, high, num_cluster, save_path)