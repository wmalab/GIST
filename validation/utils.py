import numpy as np
import pandas as pd
import os

def load_excel_loci_position(path, name, sheets):
    file = os.path.join(path, name)
    WS = pd.read_excel(file, sheet_name=sheets)
    return WS
    
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

def save_csv(df, path, name):
    file = os.path.join(path, name)
    df.to_csv(file, index=False, header=False, sep='\t')
    return


if __name__ == '__main__':
    path = '/Users/huyangyang/Desktop/chromosome_3D/validation/'
    name = 'aaf8084_supportingfile_suppl1._excel_seq1_v1.xlsx'
    sheets = ['Chr20', 'Chr21', 'Chr22', 'ChrX']
    loci_pos_hg18 = load_excel_loci_position(path,  name, sheets)
    for i, sheet in enumerate(sheets):
        df = bed_format(loci_pos_hg18[sheet])
        path = '/Users/huyangyang/Desktop/chromosome_3D/validation/'
        name = 'hg18_{}.bed'.format(sheet)
        save_csv(df, path, name)
