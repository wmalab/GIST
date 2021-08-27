import os, sys
import numpy as np
import matplotlib.pyplot as plt


def load_aic_bic(path, chrom, low, num):
    name = 'cutoff_{}_{}.txt'.format(int(low), num)
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

def get_mat(path, chrom, lows, nums):
    nrow = len(lows)
    ncol = len(nums)
    aic_mat = np.empty((nrow, ncol))
    bic_mat = np.empty((nrow, ncol))
    for i in np.arange(nrow):
        for j in np.arange(ncol):
            a, b = load_aic_bic(path, chrom, lows[i], nums[j])
            aic_mat[i, j] = a
            bic_mat[i, j] = b
    return aic_mat, bic_mat

def plot_hp(data, lows, nums, path, title):
    fig, ax = plt.subplots(2, 1)
    im = ax[0].imshow(data[0], cmap='RdBu_r')
    fig.colorbar(im, ax=ax[0])

    # We want to show all ticks...
    ax[0].set_xticks(np.arange(len(nums)))
    ax[0].set_yticks(np.arange(len(lows)))
    # ... and label them with the respective list entries
    ax[0].set_xticklabels(nums)
    ax[0].set_yticklabels(lows)
    ax[0].set_title('{} aic'.format(title))
    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # for i in range(len(lows)):
    #     for j in range(len(nums)):
    #         text = ax.text(j, i, data[i, j],
    #                     ha="center", va="center", color="w")

    im = ax[1].imshow(data[0], cmap='RdBu_r')
    # We want to show all ticks...
    ax[1].set_xticks(np.arange(len(nums)))
    ax[1].set_yticks(np.arange(len(lows)))
    # ... and label them with the respective list entries
    ax[1].set_xticklabels(nums)
    ax[1].set_yticklabels(lows)
    ax[1].set_title('{} bic'.format(title))
    fig.colorbar(im, ax=ax[1])
    fig.tight_layout()

    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, '{}.pdf'.format(title)), format='pdf')
    plt.close()

# cut_figure or figure
# cutoff_{}_{} or {}_{}.txt
if __name__ == '__main__':
    chromosomes = ['14'] #, '15', '16', '17', '18', '19', '20', '21', '22', 'X'] # '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', 
    lows = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    nums = np.arange(3, 21)
    path = '/rhome/yhu/bigdata/proj/experiment_G3DM/gmm_parameter'

    aic_list = []
    bic_list = []
    for chro in chromosomes:
        a, b = get_mat(path, chro, lows, nums)
        plot_hp([a, b], lows, nums, os.path.join(path, 'cut_figure'), 'chr{}'.format(chro))
        aic_list.append(a)
        bic_list.append(b)
        print('chromosome {} plot done'.format(chro))
    
    aic_all = np.stack(aic_list, axis=2)
    bic_all = np.stack(bic_list, axis=2)

    aic_mean = np.nanmean(aic_all, axis=2, keepdims=False)
    bic_mean = np.nanmean(bic_all, axis=2, keepdims=False)
    plot_hp([aic_mean, bic_mean], lows, nums, os.path.join(path, 'cut_figure'), 'all chromosomes mean')
    print('chromosomes mean plot done')

    aic_med = np.nanmedian(aic_all, axis=2, keepdims=False)
    bic_med = np.nanmedian(bic_all, axis=2, keepdims=False)
    plot_hp([aic_med, bic_med], lows, nums, os.path.join(path, 'cut_figure'), 'all chromosomes median')
    print('chromosomes median plot done')