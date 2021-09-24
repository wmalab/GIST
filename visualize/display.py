import os, sys
import numpy as np
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly
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
    sns.heatmap(cm, cmap= 'YlGnBu', annot=annot, fmt='', ax=ax) #"YlGnBu"
    plt.show()



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