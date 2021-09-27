import os, sys
import numpy as np
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
from sklearn.metrics import confusion_matrix



def plot_mat_diag(matu, matl, cmap=['RdBu_r', 'RdBu_r']):
    assert matu.shape == matl.shape, 'must the same shape {} and {}'.format(matu.shape, matl.shape)
    mat = np.zeros_like(matu)

    mat = np.tril(matl, k=-1) + np.triu(matu, k=1)
    np.fill_diagonal(mat, (mat.max()+mat.min())/2)

    msku = np.zeros_like(mat)
    msku[np.triu_indices_from(msku)] = True
    mskl = np.zeros_like(mat)
    mskl[np.tril_indices_from(mskl)] = True

    fig = plt.figure()
    sns.heatmap(mat, annot=False, mask=msku, cmap=cmap[0], square=True)
    sns.heatmap(mat, annot=False, mask=mskl, cmap=cmap[1], square=True)
    plt.show()


# sns histogram by group
def plot_label_value_distribution(value, label):
    value = value.flatten()
    label = label.flatten()

    data = {'value': value, 'label': label}
    df = pd.DataFrame(data=data)
    sns.displot(data, x='value', hue='label',kde=True)
    plt.show()

def plot_confusion_mat(pred, true, figsize=(10,10)):
    y_pred = pred.flatten()
    y_true = true.flatten() 
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true), normalize=None)
    cm_norm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true), normalize='true')
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm_norm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap= 'YlGnBu', annot=annot, fmt='', ax=ax, square=True) #"YlGnBu"
    plt.show()


def plot_3D(X, idx, opacity=0.7, discrete=False):
    x = X[:,0].flatten()
    y = X[:,1].flatten()
    z = X[:,2].flatten()
    idx = idx.flatten()

    if discrete:
        data = {'x': x, 'y': y, 'z': z, 'id': idx.astype(str)}
    else:
        data = {'x': x, 'y': y, 'z': z, 'id': idx.astype(float)}
    df = pd.DataFrame(data=data)

#   color_discrete_sequence= px.colors.sequential.Plasma_r, 
    fig_scatter = px.scatter_3d(df, x='x', y='y', z='z', color='id', 
                                color_discrete_sequence= px.colors.sequential.icefire, 
                                size_max=8, opacity=opacity)
    # tight layout
    #this string can be 'data', 'cube', 'auto', 'manual'
    #a custom aspectratio is defined as follows:
    fig_scatter.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    scene=dict(camera=dict(eye=dict(x=1.15, y=1.15, z=0.8)),
               xaxis=dict(), yaxis=dict(), zaxis=dict(),
               aspectmode='manual', #this string can be 'data', 'cube', 'auto', 'manual'
               aspectratio=dict(x=1, y=1, z=1) )
    fig_scatter.update_layout(scene=scene)
    
    # fig.show()
    return fig_scatter

if __name__ == '__main__':
    pass
    # matu = np.arange(25).reshape(5,5)
    # matl = -1*np.arange(25).reshape(5,5)
    # plot_mat_diag(matu, matl)

    # v1 = np.random.normal(0, 1.0, 300)
    # v2 = np.random.normal(2, 0.3, 100)
    # v = np.concatenate([v1, v2], axis=0)
    # l1 = np.zeros_like(v1)
    # l2 = np.ones_like(v2)
    # l = np.concatenate([l1, l2], axis=0).astype(int)
    # plot_label_value_distribution(v, l)

    # v1 = (np.random.normal(0, 1.0, 500)*2).astype(int)
    # v2 = (np.random.normal(0, 1.0, 500)*2).astype(int)
    # v1 = np.sort(v1)
    # v2 = np.sort(v2)
    # plot_confusion_mat(v1, v2)