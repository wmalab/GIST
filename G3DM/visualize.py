import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision

def plot_feature(feats, position, writer, item_dir):
    fig = plt.figure()
    fig, axs = plt.subplots(1, 2)
    cmaps = ['RdBu_r', 'viridis']
    for col in range(2):
        ax = axs[0, col]
        pcm = ax.pcolormesh(np.random.random((20, 20)) * (col + 1),
                            cmap=cmaps[col])
        fig.colorbar(pcm, ax=ax)
    writer.add_figure(item_dir, fig)
