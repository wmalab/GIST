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

def plot_line(data, lows, nums, path, title):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(nums, data[0].flatten(), '+-')
    ax.plot(nums, data[1].flatten(), 'x:')
    print(data)
    ax.set_xticks(nums)
    ax.set_xticklabels(nums)
    ax.legend(['aic', 'bic'])
    # ax[0].set_title('{} aic'.format(title))
    # im = ax[1].plot(nums, data[1].flatten())
    # ax[1].set_xticks(nums)
    # ax[1].set_xticklabels(nums)
    # ax[1].set_title('{} bic'.format(title))
    fig.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, '{}.pdf'.format(title)), format='pdf')
    plt.close()

def plot_hp(data, lows, nums, path, title):
    fig, ax = plt.subplots(2, 1)
    im = ax[0].imshow(data[0], cmap='RdBu_r')
    fig.colorbar(im, ax=ax[0])

    ax[0].set_xticks(np.arange(len(nums)))
    ax[0].set_yticks(np.arange(len(lows)))
    ax[0].set_xticklabels(nums)
    ax[0].set_yticklabels(lows)
    ax[0].set_title('{} aic'.format(title))


    im = ax[1].imshow(data[1], cmap='RdBu_r')
    ax[1].set_xticks(np.arange(len(nums)))
    ax[1].set_yticks(np.arange(len(lows)))
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
    chromosomes = ['21', '22'] #, '15', '16', '17', '18', '19', '20', '21', '22', 'X'] # '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', 
    lows = [5] # , 10, 15, 20, 25, 30, 35, 40, 45, 50]
    nums = np.arange(2, 15)
    
    name = 'Rao2014-IMR90-MboI-allreps-filtered.10kb.cool'
    cell = name.split('.')[0]
    resolu = name.split('.')[1]
    path = os.path.join('/rhome/yhu', 'bigdata', 'proj', 'experiment_G3DM', 'figures', 'gmm_parameter', cell, resolu)
    # path = '/rhome/yhu/bigdata/proj/experiment_G3DM/figures/gmm_parameter'

    aic_list = []
    bic_list = []
    for chro in chromosomes:
        a, b = get_mat(path, chro, lows, nums)
        # plot_hp([a, b], lows, nums, os.path.join(path, 'cut_figure'), 'chr{}'.format(chro))
        plot_line([a, b], lows, nums, os.path.join(path, 'cut_figure'), 'chr{}'.format(chro))
        aic_list.append(a)
        bic_list.append(b)
        print('chromosome {} plot done'.format(chro))
    
    aic_all = np.stack(aic_list, axis=2)
    bic_all = np.stack(bic_list, axis=2)

    aic_mean = np.nanmean(aic_all, axis=2, keepdims=False)
    bic_mean = np.nanmean(bic_all, axis=2, keepdims=False)
    # plot_hp([aic_mean, bic_mean], lows, nums, os.path.join(path, 'cut_figure'), 'all chromosomes mean')
    plot_line([aic_mean, bic_mean], lows, nums, os.path.join(path, 'cut_figure'), 'all chromosomes mean')
    print('chromosomes mean plot done')

    aic_med = np.nanmedian(aic_all, axis=2, keepdims=False)
    bic_med = np.nanmedian(bic_all, axis=2, keepdims=False)
    # plot_hp([aic_med, bic_med], lows, nums, os.path.join(path, 'cut_figure'), 'all chromosomes median')
    plot_line([aic_med, bic_med], lows, nums, os.path.join(path, 'cut_figure'), 'all chromosomes median')
    print('chromosomes median plot done')