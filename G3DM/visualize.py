import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import torch
import torchvision

from sklearn import metrics

def plot_feature(feats, position, writer, item_dir):
    data = [feats, position]
    fig = plt.figure()
    fig, axs = plt.subplots(1, len(data))
    cmaps = ['RdBu_r', 'viridis', 'RdBu_r']
    for col in range(len(data)):
        ax = axs[col]
        pcm = ax.pcolormesh(data[col] * (col + 1), cmap=cmaps[col])
        fig.colorbar(pcm, ax=ax)
    writer.add_figure(item_dir, fig)

def plot_X(X, writer, item_dir, step=None):
    fig = plt.figure()
    for i in np.arange(4):
        data = X[:,i,:]
        ax = fig.add_subplot(2,2,i+1, projection='3d')
        cmap = cm.get_cmap(plt.get_cmap('autumn')) # RdBu_r
        ax.scatter(data[:,0], data[:,1], data[:,2], c = np.arange(data.shape[0]), cmap=cmap, marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_aspect('equal', 'box')
    step = 0 if step is None else step
    writer.add_figure(item_dir, fig, global_step=step)

def plot_cluster(mat, writer, ncluster, item_dir, step=None):
    fig = plt.figure()
    data = mat
    cmaps = ['RdBu_r']
    fig, axs = plt.subplots(1, 1)
    pcm = axs.pcolormesh(data, cmap=cmaps[0], vmin=0, vmax=ncluster)
    fig.colorbar(pcm, ax=axs)
    step = 0 if step is None else step
    writer.add_figure(item_dir, fig, global_step=step)

def plot_confusion_mat(y_pred, y_true, writer, item_dir, step=None):
    pred = y_pred.flatten()
    true = y_true.flatten()
    cm = metrics.confusion_matrix(true, pred, normalize='true')
    fig = plt.figure()
    cmaps = ['RdBu_r']
    fig, axs = plt.subplots(1, 1)
    pcm = axs.pcolormesh(cm, cmap=cmaps[0])
    for (i, j), z in np.ndenumerate(cm):
        axs.text(j+0.5, i+0.5, '{:0.1f}'.format(z), ha='center', va='center')
    fig.colorbar(pcm, ax=axs)
    axs.set_ylabel('True') # row of cm
    axs.set_xlabel('Prediction') # col of cm
    step = 0 if step is None else step
    writer.add_figure(item_dir, fig, global_step=step)

def plot_lines(x, writer, item_dir, step=None):
    y = np.ones_like(x.flatten())
    fig, axs = plt.subplots(1, 1)
    cmaps = ['tab20']
    z = np.arange(len(x.flatten()))
    axs.scatter(x.flatten(),y, c=z, cmap=cmaps[0])
    axs.plot(x.flatten(), y)
    plt.xlim(left=-0.01)
    step = 0 if step is None else step
    writer.add_figure(item_dir, fig, global_step=step)

def plot_scaler(value, writer, item_dir, step):
    writer.add_scalar(item_dir, value, step)