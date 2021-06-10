import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import torch
import torchvision

def plot_feature(feats, position, writer, item_dir):
    data = [feats, position, feats+position]
    fig = plt.figure()
    fig, axs = plt.subplots(1, len(data))
    cmaps = ['RdBu_r', 'viridis', 'RdBu_r']
    for col in range(len(data)):
        ax = axs[col]
        pcm = ax.pcolormesh(data[col] * (col + 1), cmap=cmaps[col])
        fig.colorbar(pcm, ax=ax)
    writer.add_figure(item_dir, fig)

def plot_X(X, writer, item_dir, step):
    data = X[:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = cm.get_cmap(plt.get_cmap('RdBu_r'))
    ax.scatter(data[:,0], data[:,1], data[:,2], c = np.arange(data.shape[0]), cmap=cmap, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    step = 0 if step is None else step
    writer.add_figure(item_dir, fig, global_step=step)
