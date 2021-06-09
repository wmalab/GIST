import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision

def plot_feature(feats, position, writer, item_dir):
    fig = plt.figure()
    fig, axs = plt.subplots(1, 2)
    cmaps = ['RdBu_r', 'viridis']
    data = [feats, position]
    for col in range(2):
        ax = axs[col]
        pcm = ax.pcolormesh(data[col] * (col + 1),
                            cmap=cmaps[col])
        fig.colorbar(pcm, ax=ax)
    writer.add_figure(item_dir, fig)
