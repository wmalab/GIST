from __future__ import print_function, division

import os
import sys
import numpy as np

from .utils import hic_prepare
from .build_graph import create_graph_1lvl
from .build_feature import create_feature, save_feature


def create_data(num_clusters, chromosome,
                dim,
                cutoff_clusters_limits, cutoff_cluster, 
                max_len, iteration,
                cool_data_path, cool_file,
                data_path
                ):
    [feature_path, graph_path] = data_path

    cool = os.path.join(cool_data_path, cool_file)
    norm_hic = hic_prepare( rawfile=cool, chromosome='chr{}'.format(str(chromosome)))

    print('create featrue chromosome {}, iced normalization Hi-C shape: {}'.format(chromosome, norm_hic.shape))

    feature_dict = create_feature(norm_hic, dim)
    feature = feature_dict['feat']
    position = feature_dict['pos']

    output_g_path = os.path.join(graph_path, 'chr{}'.format(chromosome))
    os.makedirs(output_g_path, exist_ok=True)
    print('saved graph in: ', output_g_path)
    output_prefix_file = 'G_chr-{}'.format(chromosome)
    cluster_weight, cw_mat = create_graph_1lvl(norm_hic,
                                       num_clusters, max_len, iteration,
                                       cutoff_clusters_limits, 
                                       cutoff_cluster,
                                       output_g_path, output_prefix_file)

    feature_dict['cluster_weight'] = {'cw':cluster_weight, 'mat':cw_mat}
    output_f_path = feature_path
    output_f_file = 'F_chr-{}'.format(chromosome)
    save_feature(output_f_path, output_f_file, feature_dict)

    return
