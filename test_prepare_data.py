import os, sys, shutil
import json, pickle
import numpy as np
import multiprocessing

from feature.utils import feature_hic, position_hic, save_feature, load_feature
from prepare.utils import log1p_hic, save_graph, load_graph
from prepare.data_prepare import hic_prepare_pooling
from prepare.build_graph import create_hierarchical_graph_2lvl
from prepare.build_dataset import HiCDataset

import torch

import warnings
warnings.filterwarnings('ignore')

def check_dim(dim, x):
    assert dim <= int(x.shape[0]/2)-1
    return int(dim)
 
def create_feature(ratio, stride, dim, chromosome, cool_path, cool_file, output_path, output_file):
    ''' create Hi-C feature '''
    ratios = np.array([1, ratio])
    strides = np.array([1, stride])
    cool = os.path.join(cool_path, cool_file)
    # 2 Hi-C matrices, iced normalization, diagonal fill 0
    norm_hics = hic_prepare_pooling(
        rawfile=cool,
        chromosome='chr{}'.format(str(chromosome)),
        ratios=ratios, strides=strides,
        remove_zero_col=False)

    for i, m in enumerate(norm_hics):
        print('chromosome {} level {}, iced normalization Hi-C shape: {}'.format(chromosome, i, m.shape))

    log_hics = [log1p_hic(x) for x in norm_hics]
    idxs = [np.arange(len(hic)) for hic in norm_hics]

    # ! dim can't larger than int(x.shape[0]/2)-1
    feats = [feature_hic(x, check_dim(dim[i], x)) for i, x in enumerate(log_hics)]

    features = []
    nrepeats = strides
    nrepeats = nrepeats.astype(int)
    features.append(feats[-1])
    for i in np.arange(len(feats)-2, -1, -1):
        f0 = feats[i]
        f1 = np.repeat(features[0], nrepeats[i+1], axis=0)[0:f0.shape[0], :]
        f = np.concatenate((f0, f1), axis=1)
        features.insert(0, f)

    positions = []
    for f in features:
        positions.append(position_hic(f, f.shape[1]))

    f_dict = {'hic_h0': {'feat':features[0], 'pos': positions[0]}, 'hic_h1': {'feat':features[1], 'pos': positions[1]}}
    save_feature(output_path, output_file, f_dict)

def create_graph(ratio, stride, num_clusters, chromosome, cutoff_percent, cutoff_cluster, cool_path, cool_file, output_path, output_file):
    ''' create Hi-C graph. 2 levels, high lvl 0 as center, low lvl 1 as bead '''
    ratios = np.array([1, ratio], dtype=np.int)
    strides = np.array([1, stride], dtype=np.int)
    cool = os.path.join(cool_path, cool_file)

    # 2 Hi-C matrices, iced normalization, diagonal fill 0
    norm_hics = hic_prepare_pooling(
        rawfile=cool,
        chromosome='chr{}'.format(str(chromosome)),
        ratios=ratios, strides=strides,
        remove_zero_col=False)

    for i, m in enumerate(norm_hics):
        print('chromosome {} level {}, iced normalization Hi-C shape: {}'.format(chromosome, i, m.shape))

    g, g_list = create_hierarchical_graph_2lvl(
        norm_hics, num_clusters, ratios, strides, cutoff_percent=cutoff_percent, cutoff_cluster=cutoff_cluster)
    save_graph(g_list, output_path, output_file)


if __name__ == '__main__':
    # root = '.'
    root = '/rhome/yhu/bigdata/proj/experiment_G3DM'

    configuration_src_path = os.path.join(root, 'data')
    configuration_name = 'config.json'
    with open(os.path.join(configuration_src_path, configuration_name)) as f:
        config_data = json.load(f)

    cool_file = config_data['cool_file']
    cell = cool_file.split('.')[0]
    hyper = '_'.join([cool_file.split('.')[1], 'id', config_data["id"]])

    configuration_dst_path = os.path.join(root, 'data', cell, hyper)
    os.makedirs(configuration_dst_path, exist_ok=True)
    shutil.copy(os.path.join(configuration_src_path,
                             configuration_name), configuration_dst_path)

    # '/rhome/yhu/bigdata/proj/experiment_G3DM'
    root = config_data['root'] if config_data['root'] else root
    cool_data_path = config_data['cool_data_path'] if config_data['cool_data_path'] else os.path.join(
        root, 'data', 'raw')
    graph_path = config_data['graph_path'] if config_data['graph_path'] else os.path.join(
        root, 'data', cell, hyper, 'graph')
    feature_path = config_data['feature_path'] if config_data['feature_path'] else os.path.join(
        root, 'data', cell, hyper, 'feature')
    dataset_path = config_data['dataset_path']['path'] if config_data['dataset_path']['path'] else os.path.join(
        root, 'data', cell, hyper)
    dataset_name = config_data['dataset_path']['name'] if config_data['dataset_path']['name'] else 'dataset.pt'
    output_path = config_data['output_path'] if config_data['output_path'] else os.path.join(
        root, 'data', cell, hyper, 'output')

    saved_model_path = config_data['saved_model']['path'] if config_data['saved_model']['path'] else os.path.join(
        root, 'saved_model')
    saved_model_name = config_data['saved_model']['name'] if config_data['saved_model']['name'] else 'model_net'

    os.makedirs(graph_path, exist_ok=True)
    os.makedirs(feature_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(saved_model_path, exist_ok=True)

    ratio = config_data['hyper']['ratio']
    stride = config_data['hyper']['stride']
    dim = [ config_data['parameter']['feature']['in_dim']['h0'], config_data['parameter']['feature']['in_dim']['h1'] ]
    all_chromosome = config_data['all_chromosomes']
    train_chromosomes = config_data['train_chromosomes']
    valid_chromosomes = config_data['valid_chromosomes']
    test_chromosomes = config_data['test_chromosomes']

    clusters = config_data['parameter']['graph']['num_clusters']
    # load dict
    num_clusters = dict()
    for key, value in clusters.items():
        num_clusters[int(key)] = int(value)

    percents = config_data['parameter']['graph']['cutoff_percent']
    cutoff_percents = dict()
    for key, value in percents.items():
        cutoff_percents[int(key)] = int(value)

    cutoff_cluster = int(config_data['parameter']['graph']['cutoff_cluster'])

    pool_num = len(all_chromosome)*2 if multiprocessing.cpu_count() > len(
        all_chromosome)*2 else multiprocessing.cpu_count()
    pool = multiprocessing.Pool(pool_num)
    for chromosome in all_chromosome:
        feat_args = (ratio, stride, dim, chromosome,
                    cool_data_path, cool_file,
                    feature_path, 'F_chr-{}.pkl'.format(chromosome))
        pool.apply_async(create_feature, args=feat_args)
        graph_args = (ratio, stride, num_clusters, chromosome,
                    cutoff_percents, cutoff_cluster,
                    cool_data_path, cool_file,
                    graph_path, 'G_chr-{}.bin'.format(chromosome))
        pool.apply_async(create_graph, args=graph_args)
    pool.close()
    pool.join()

    graph_dict = dict()
    feature_dict = dict()
    train_list, valid_list, test_list = list(), list(), list()
    for chromosome in all_chromosome:
        # graph_dict[chromosome] = {top_graph, top_subgraphs, bottom_graph, inter_graph}
        # feature_dict[chromosome] = {'hic_feat_h0', 'hic_feat_h1'}
        graph_dict[str(chromosome)] = load_graph(graph_path, 'G_chr-{}.bin'.format(chromosome))
        feature_dict[str(chromosome)] = load_feature(feature_path, 'F_chr-{}.pkl'.format(chromosome))
        if str(chromosome) in train_chromosomes:
            train_list.append(str(chromosome))
        if str(chromosome) in valid_chromosomes:
            valid_list.append(str(chromosome))
        if str(chromosome) in test_chromosomes:
            test_list.append(str(chromosome))
    
    # create HiCDataset and save
    HD = HiCDataset(graph_dict, feature_dict, train_list, valid_list, test_list, dataset_path, dataset_name)
    torch.save(HD, os.path.join( dataset_path, dataset_name))

    '''load_HD = torch.load(os.path.join( dataset_path, dataset_name))
    load_HD[0]
    print(load_HD[0])'''
