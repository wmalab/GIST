import os
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform

def load_excel_(path, name, sheets):
    file = os.path.join(path, name)
    WS = pd.read_excel(file, engine='openpyxl', sheet_name=sheets)
    return WS

def load_excel_fish3d(path, name, sheets):
    print('loading fish3d excel file {}'.format(name))
    return load_excel_(path, name, sheets)

def load_df_fish3d(path, name):
    print('loading fish3d excel file {}'.format(name))
    file = os.path.join(path, name)
    df = pd.read_csv(file, sep='\t')
    return df

def load_excel_loci_position(path, name, sheets):
    print('loading loci position excel file {}'.format(name))
    return load_excel_(path, name, sheets)

def load_tad_bed(path, name):
    file = os.path.join(path, name)
    df = pd.read_csv(file, sep='\t', header=None, names=["Chromosome", "Start", "End"])
    df['Chromosome'] = df['Chromosome'].str.lstrip('chr')
    return df

def bed_format(dataframe):
    """"chr4 100000 100001", 0-based"""
    # print(dataframe.head())
    # print(dataframe.columns)
    data = dict()
    data['chromosome'] = 'chr' + dataframe['Chromosome'].astype(str)
    data['start'] = dataframe['Start genomic coordinate']
    data['end'] = dataframe['End genomic coordinate']
    df = pd.DataFrame(data)
    return df

def remove_failed(data, failed):
    """
    chr20 12th row, 1 based
    chr21 1st row, 1 based
    chr22 27th row, 1 based
    """
    failed = np.array(failed)
    data = np.array(data)
    res = np.delete(data, failed, 0)
    res = np.delete(res, failed, 1)
    return res

def select_loci(data, resolution):
    """data as df from load_bed"""
    res = list()
    start = data['Start'].to_numpy().astype(int)
    end = data['End'].to_numpy().astype(int)

    for l, r in zip(start, end):
        tmp = np.unique( int(np.arange(l, r, step=resolution)/resolution) )
        res.append(tmp)
    return res

def fish3d_format(dataframe):
    # ( chromosomeID, tadID, 3)
    nChrID = dataframe['Serial # of the imaged chromosome'].max()
    nTADID = dataframe['ID # of the imaged TAD'].max()
    u = dataframe['Serial # of the imaged chromosome'].to_numpy(dtype=int) - 1
    v = dataframe['ID # of the imaged TAD'].to_numpy(dtype=int) - 1
    x,y,z = dataframe['x(um)'], dataframe['y(um)'], dataframe['z(um)']
    x = torch.tensor(x).view(-1,)
    y = torch.tensor(y).view(-1,)
    z = torch.tensor(z).view(-1,)
    xyz = torch.stack( [x,y,z], dim=-1)
    data = torch.empty((nChrID, nTADID, 3))
    data[u,v,:] = xyz.float()
    return data.cpu().numpy()

def pdist_3d(data):
    """ input: (serial#, N, 3)
        output: (serial#, N, N)"""
    m = data.shape[0]
    n = data.shape[1]
    res = np.empty((m, n, n))
    for i in np.arange(m):
        X = np.squeeze(data[i, :, :])
        res[i, :, :] = squareform( pdist(X, 'euclidean') )
    return res

def save_csv(df, path, name, index=False, header=False):
    file = os.path.join(path, name)
    df.to_csv(file, index=index, header=header, sep='\t')
    return

if __name__ == '__main__':
    path = '/Users/huyangyang/Desktop/chromosome_3D/validation/'
    name = 'aaf8084_supportingfile_suppl1._excel_seq1_v1.xlsx'
    # sheets = ['Chr20', 'Chr21', 'Chr22', 'ChrX']
    # loci_pos_hg18 = load_excel_loci_position(path,  name, sheets)
    # for i, sheet in enumerate(sheets):
    #     df = bed_format(loci_pos_hg18[sheet])
    #     path = '/Users/huyangyang/Desktop/chromosome_3D/validation/'
    #     name = 'hg18_{}.bed'.format(sheet)
    #     save_csv(df, path, name)

    # for i in [4,5,6]:
    #     name = 'aaf8084_supportingfile_suppl1._excel_seq{}_v1.xlsx'.format(i)
    #     fish3d_df =  load_excel_fish3d(path, name, 0)
    #     save_csv(fish3d_df, path, 'FISH_Chr{}.xyz'.format(i+16), True, True)
    fish3d_df = load_df_fish3d(path, 'FISH_Chr{}.xyz'.format(20))
    data = fish3d_format(fish3d_df)
    print(data.shape)
    pdist = pdist_3d(data)
    print(np.nanmean(pdist, axis=0))


    # name = 'hg19_Chr20.bed'
    # df = load_tad_bed(path, name)
    # select_loci(df, 10000)
