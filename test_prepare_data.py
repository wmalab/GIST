import os, sys
import shutil, json, pickle
import numpy as np

from feature.utils import feature_hic, save_feature
from prepare.data_prepare import hic_prepare_pooling
from prepare.utils import log1p_hic

from prepare.build_graph import create_hierarchical_graph_2lvl, save_graph


def create_feature(ratio, stride, max_dim, chromosome, cool_path, cool_file, output_path, output_file):
    ''' create Hi-C feature
    '''
    ratios= np.array([1, ratio])
    strides=np.array([1, stride])
    cool = os.path.join(cool_path, cool_file)
    # 2 Hi-C matrices, iced normalization, diagonal fill 0
    norm_hics = hic_prepare_pooling(
        rawfile=cool, 
        chromosome='chr{}'.format(str(chromosome)), 
        ratios=ratios, strides=strides, 
        remove_zero_col=False)

    for i, m in enumerate(norm_hics):
        print('level {}, iced normalization Hi-C shape: {}'.format(i, m.shape))

    log_hics = [log1p_hic(x) for x in norm_hics]
    idxs = [np.arange(len(hic)) for hic in norm_hics]
    d_model =  lambda x: min(max_dim, int(x.shape[0]/2)-1)
    feats = [feature_hic(x , d_model(x)) for x in log_hics]

    features = []
    nrepeats = strides
    nrepeats = nrepeats.astype(int)
    features.append(feats[-1])
    for i in np.arange(len(feats)-2, -1, -1):
        f0 = feats[i]
        f1 = np.repeat(features[0], nrepeats[i+1], axis=0)[0:f0.shape[0],:]
        f = np.concatenate((f0, f1),axis=1)
        features.insert(0, f)

    f_dict = {'hic_feat_h0': features[0], 'hic_feat_h1': features[1]}
    save_feature(output_path, output_file, f_dict)

def create_graph(ratio, stride, num_clusters, chromosome, cool_path, cool_file, output_path, output_file):
    ''' create Hi-C graph
        2 levels, high lvl 0 as center, low lvl 1 as bead
    '''
    ratios= np.array([1, ratio], dtype=np.int)
    strides=np.array([1, stride], dtype=np.int)
    cool = os.path.join(cool_path, cool_file)

    # 2 Hi-C matrices, iced normalization, diagonal fill 0
    norm_hics = hic_prepare_pooling(
        rawfile=cool, 
        chromosome='chr{}'.format(str(chromosome)), 
        ratios=ratios, strides=strides, 
        remove_zero_col=False)

    for i, m in enumerate(norm_hics):
        print('level {}, iced normalization Hi-C shape: {}'.format(i, m.shape))

    g = create_hierarchical_graph_2lvl(norm_hics, num_clusters, ratios, strides)
    save_graph(g, output_path, output_file)


if __name__ == '__main__':
    root = '.'
    # root = '/rhome/yhu/bigdata/proj/experiment_G3DM'

    configuration_src_path = os.path.join(root, 'data')
    configuration_name = 'config.json'

    with open( os.path.join(configuration_src_path, configuration_name) ) as f:
       config_data = json.load(f)
    
    cool_file= config_data['cool_file']
    cell = cool_file.split('.')[0]
    hyper = '_'.join([cool_file.split('.')[1], 'id', config_data["id"]])

    configuration_dst_path = os.path.join(root, 'data', cell, hyper)
    os.makedirs(configuration_dst_path, exist_ok=True)
    shutil.copy(os.path.join(configuration_src_path, configuration_name), configuration_dst_path)

    root = config_data['root'] if config_data['root'] else root # '/rhome/yhu/bigdata/proj/experiment_G3DM'
    cool_data_path = config_data['cool_data_path'] if config_data['cool_data_path'] else os.path.join(root, 'data', 'raw')
    graph_path = config_data['graph_path'] if config_data['graph_path'] else os.path.join(root, 'data', cell, hyper, 'graph')
    feature_path = config_data['feature_path'] if config_data['feature_path'] else os.path.join(root, 'data', cell, hyper, 'feature')
    output_path = config_data['output_path'] if config_data['output_path'] else os.path.join(root, 'data', cell, hyper, 'output')

    saved_model_path = config_data['saved_model']['path'] if config_data['saved_model']['path'] else os.path.join(root, 'saved_model')
    saved_model_name = config_data['saved_model']['name'] if config_data['saved_model']['name'] else 'model_net'

    os.makedirs(graph_path, exist_ok=True)
    os.makedirs(feature_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(saved_model_path, exist_ok=True)

    ratio = config_data['hyper']['ratio']
    stride = config_data['hyper']['stride']
    max_dim = config_data['parameter']['feature']['max_feature_dim']
    all_chromosome = config_data['all_chromosomes']
    clusters = config_data['parameter']['graph']['num_clusters']
    # convert dict to list
    num_clusters = np.zeros(len(clusters), dtype=np.int)
    for key, value in clusters.items():
        num_clusters[int(key)] = int(value)

    for chromosome in all_chromosome:
        create_feature(ratio, stride, max_dim, chromosome, 
                        cool_data_path, cool_file, 
                        output_path=feature_path, output_file='F_chr-{}.pkl'.format(chromosome))

    for chromosome in all_chromosome:
        create_graph(ratio, stride, num_clusters, chromosome, 
                        cool_data_path, cool_file, 
                        output_path=graph_path, output_file='G_chr-{}.bin'.format(chromosome))