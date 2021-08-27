import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from matplotlib import cm
import numpy as np

import torch
import torchvision

from sklearn import metrics

def plot_feature(data, writer, item_dir):
    # data = [featsv, featsh, position]
    fig = plt.figure()
    fig, axs = plt.subplots(1, len(data))
    cmaps = ['RdBu_r', 'viridis']
    for col in range(len(data)):
        ax = axs[col]
        c = cmaps[col] if col < len(data)-1 else cmaps[-1]
        pcm = ax.pcolormesh(data[col] * (col + 1), cmap=c)
        fig.colorbar(pcm, ax=ax)
    writer.add_figure(item_dir, fig)

def plot_X(S, writer, item_dir, step=None):
    fig = plt.figure()
    for i in np.arange(4):
        data = S[:,i,:]
        data = data - np.mean(data, axis=0)
        ax = fig.add_subplot(2,2,i+1, projection='3d')
        cmap = cm.get_cmap(plt.get_cmap('autumn')) # RdBu_r
        X, Y, Z = data[:,0], data[:,1], data[:,2]
        ax.scatter(X, Y, Z, c = np.arange(data.shape[0]), cmap=cmap, marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        '''max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)'''
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
        axs.text(j+0.4, i+0.4, '{:0.2f}'.format(z), ha='center', va='center')
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
    scatter = axs.scatter(x.flatten(),y, c=z, cmap=cmaps[0])
    legend = axs.legend(*scatter.legend_elements(),
                    loc="lower right", title="Classes")
    axs.add_artist(legend)
    axs.plot(x.flatten(), y)
    plt.xlim(left=-0.01)
    step = 0 if step is None else step
    writer.add_figure(item_dir, fig, global_step=step)

def plot_distributions(inputs, writer, item_dir, step=None):
    [mu, x, pdfs] = inputs

    y = np.zeros_like(mu.flatten())
    z = np.arange(len(mu.flatten()))

    fig, axs = plt.subplots(1, 1)
    cmaps = ['tab20']
    scatter = axs.scatter(mu.flatten(), y, c=z, cmap=cmaps[0])
    legend = axs.legend(*scatter.legend_elements(),
                    loc="lower right", title="Classes")
    axs.add_artist(legend)

    n = pdfs.shape[1]
    colors = pl.cm.get_cmap['tab20'](np.linspace(0,1,n))
    for i in np.arange(pdfs.shape[1]):
        axs.plot(x.flatten(), pdfs[:,i], cmp=cmaps[i])
        plt.xlim(left=-0.01)

    step = 0 if step is None else step
    writer.add_figure(item_dir, fig, global_step=step)

def plot_scaler(value, writer, item_dir, step):
    writer.add_scalars(item_dir, value, step)