import os, sys
import numpy as np
import matplotlib.pyplot as plt


def load_aic_bic(path, chrom, low, num):
    name = '{}_{}.txt'.format(int(low), num)
    file = os.path.join(path, 'chr{}'.format(chrom), 'aic', name)
    if os.path.isfile(file):
        with open(file, 'r') as fin:
            aic = float(fin.readline())
            fin.close()
    else:
        aic = None
    
    file = os.path.join(path, 'chr{}'.format(chrom), 'bic', name)
    if os.path.isfile(file):
        with open(file, 'r') as fin:
            bic = float(fin.readline())
            fin.close()
    else:
        bic = None
    return aic, bic

def plot_mat(path, chrom, lows, nums):
    nrow = len(lows)
    ncol = len(nums)
    aic_mat = np.empty((nrow, ncol))
    bic_mat = np.empty((nrow, ncol))
    for i in np.arange(nrow):
        for j in np.arange(ncol):
            a, b = load_aic_bic(path, chrom, lows[i], nums[j])
            aic_mat[i, j] = a
            bic_mat[i, j] = b
    plot_hp(aic_mat, lows, nums, os.path.join(path, 'figure'), 'chr{}_aic'.format(chrom))
    plot_hp(bic_mat, lows, nums, os.path.join(path, 'figure'), 'chr{}_bic'.format(chrom))

def plot_hp(data, lows, nums, path, title):
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='hot')

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(nums)))
    ax.set_yticks(np.arange(len(lows)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(nums)
    ax.set_yticklabels(lows)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # for i in range(len(lows)):
    #     for j in range(len(nums)):
    #         text = ax.text(j, i, data[i, j],
    #                     ha="center", va="center", color="w")
    plt.colorbar()
    ax.set_title(title)
    fig.tight_layout()

    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, '{}.pdf'.format(title)), format='pdf')

if __name__ == '__main__':
    chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
    lows = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    nums = np.arange(3, 21)
    path = '/rhome/yhu/bigdata/proj/experiment_G3DM/gmm_parameter'

    plot_mat(path, '22', lows, nums)
