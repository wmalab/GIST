import dgl
import torch
import numpy as np
import os
import multiprocessing
from .utils import cluster_hic, log1p_hic
from .utils import save_graph
# from embedding.model import embedding, tilt_hic, position_hic


def create_subgraph_(ID, mat_hic, mat_chic, idx,
                     cutoff_percent, cutoff_cluster,
                     output_path, output_prefix_file):
    '''
        mat_hic: entire Hi-C
        mat_chic: entire clusters of Hi-C
        idx: subgraph index
    '''
    hic = mat_hic[idx, :]
    hic = hic[:, idx]
    chic = mat_chic[idx, :]
    chic = chic[:, idx]

    # creat graph
    graph_data = dict()
    cutoff = np.percentile(hic, cutoff_percent)
    fid = np.where(hic > cutoff)
    if len(fid[0])==0 or len(fid[1])==0:
        return False
    fid_interacts = fid
    u = np.concatenate(fid[0].reshape(-1, 1))
    v = np.concatenate(fid[1].reshape(-1, 1))
    graph_data[('bead', 'interacts', 'bead')] = (u, v)

    c_list = [r for r in range(cutoff_cluster)]  # int(n_cluster)-1)
    fid = []
    for i in c_list:
        [src, dst] = np.where(chic == i)
        u = np.concatenate([src])
        v = np.concatenate([dst])
        if len(u)==0 or len(v)==0:
            return False
        fid.append(np.where(chic == i))
        graph_data[('bead', 'interacts_c{}'.format(str(i)), 'bead')] = (u, v)

    num_nodes_dict = {'bead': len(idx)}
    g = dgl.heterograph(graph_data, num_nodes_dict, idtype=torch.long)

    g.nodes['bead'].data['id'] = torch.tensor(idx.flatten(), dtype=torch.long)

    g.edges['interacts'].data['label'] = chic[tuple(
            fid_interacts)].clone().detach().flatten().type(torch.int8)

    top_list = ['interacts_c{}'.format(i) for i in np.arange(cutoff_cluster)]
    top_subgraphs = g.edge_type_subgraph(top_list)
    top_graph = g.edge_type_subgraph(['interacts'])

    g_list = [top_graph, top_subgraphs]

    output_file = output_prefix_file+'_{}'.format(ID)
    save_graph(g_list, output_path, output_file)
    print('#{} Done graphs saved in \n \t{}'.format(ID, output_path))

    return True


def create_graph_1lvl(norm_hic,
                      num_clusters, max_len, itn,
                      cutoff_percent, cutoff_cluster,
                      output_path, output_prefix_filename):
    log_hic = log1p_hic(norm_hic)

    # # fill diagonal offset 1 to make a chain of one chromosome
    # diag_1 = np.diagonal(log_hic, offset=1)
    # mean_diag_1 = np.nanmean(diag_1)
    # x_nan_diag_1 = np.argwhere(np.isnan(diag_1))
    # y_nan_diag_1 = x_nan_diag_1 + 1
    # log_hic[x_nan_diag_1, y_nan_diag_1] = mean_diag_1

    n_idx = np.sort(np.argwhere(np.sum(log_hic, axis=0)!=0)).flatten()
    idxs = n_idx
    # only 1 log Hi-C

    mats_, matpbs_ = cluster_hic(log_hic, log_hic, n_cluster=num_clusters)
    cluster_weight, _ = np.histogram(mats_.view(-1, 1),
                                     bins=np.arange(num_clusters),
                                     density=True)
    cluster_weight = np.append(cluster_weight, [1.0])
    # 1/density
    cluster_weight = (1.0/(cluster_weight+10e-7).astype(np.double))
    print('# hic: {} clusters, weights: {}'.format(num_clusters, cluster_weight))
    # -----------------------------------------------------------------------------

    # permutation idx in idex
    if len(idxs) <= max_len:
        create_subgraph_(0, log_hic, mats_, idxs,
                         cutoff_percent, cutoff_cluster,
                         output_path, output_prefix_filename)
    else:
        idx_list = permutation_list(idxs, max_len)
        pool_num = np.min([len(idx_list), multiprocessing.cpu_count()])
        pool = multiprocessing.Pool(pool_num)
        result_objs=[]
        for i, idx in enumerate(idx_list):
            data_args = (i, log_hic, mats_, idx,
                         cutoff_percent, cutoff_cluster,
                         output_path, output_prefix_filename)
            res = pool.apply_async(create_subgraph_, args=data_args)
            result_objs.append(res)
            # create_subgraph_(i, log_hic, mats_, idx,
            #              cutoff_percent, cutoff_cluster,
            #              output_path, output_prefix_filename)
        pool.close()
        pool.join()

    # -----------------------------------------------------------------------------
    remove_mats_ = mats_[n_idx, :]
    remove_mats_ = remove_mats_[:, n_idx]
    return cluster_weight, remove_mats_

def permutation_list(idx, max_len, iteration=10):
    idx_list = []

    # 1 continous idx
    step = np.ceil(max_len/10).astype(int)
    for i in np.arange(0, len(idx), step=step):
        l,r = i, min(i+max_len, len(idx))
        sub = idx[l:r]
        sub = np.sort(sub)
        idx_list.append(sub)
    idx_list.append(idx[-max_len:])

    # # 2 random
    # num = np.ceil(len(idx)/max_len).astype(int)
    # offset = np.min([offset, 2*num])
    # for epoch in np.arange(iteration):
    #     rand_idx = np.random.permutation(idx)
    #     sub_idx = np.array_split(rand_idx, 2*num)
    #     for i in np.arange(0, 2*num):
    #         for j in np.arange(i+1, offset):
    #             sub = np.concatenate((sub_idx[i], sub_idx[j]), axis=0)
    #             sub = np.unique(sub.flatten())
    #             idx_list.append( np.sort(sub) )

    # 2 random
    num = np.ceil(len(idx)/max_len).astype(int)
    for epoch in np.arange(iteration):
        rand_idx = np.random.permutation(idx)
        sub_idx = np.array_split(rand_idx, num)
        for i in np.arange(0, num):
            sub = np.array(sub_idx[i])
            sub = np.unique(sub.flatten())
            idx_list.append( np.sort(sub) )
    return idx_list

def create_hierarchical_graph_2lvl(norm_hics, num_clusters, ratios, strides, cutoff_percent={0: 10, 1: 10}, cutoff_cluster={0: 4, 1: 6}):
    log_hics = [log1p_hic(x) for x in norm_hics]
    idxs = [np.arange(len(hic)) for hic in norm_hics]

    '''
    d_model =  lambda x: int(x.shape[0]/2)-1
    fs = [tilt_hic(x , d_model(x)) for x in log_hics]
    feats = [position_hic(x, x.shape[1]) for x in fs]
    features = []
    nrepeats = strides
    nrepeats = nrepeats.astype(int)
    features.append(feats[-1])
    for i in np.arange(len(feats)-2, -1, -1):
        f0 = feats[i]
        f1 = np.repeat(features[0], nrepeats[i+1], axis=0)[0:f0.shape[0],:]
        f = np.concatenate((f0, f1),axis=1)
        features.insert(0, f)'''

    mats_ = []
    matpbs_ = []
    cweights_ = []
    for i, hic in enumerate(log_hics):
        m, pb = cluster_hic(hic, hic, n_cluster=num_clusters[i])
        mats_.append(m)
        matpbs_.append(pb)
        cluster_weight, _ = np.histogram(
            m.view(-1, 1), bins=np.arange(num_clusters[i]), density=True)
        cluster_weight = np.append(cluster_weight, [1.0])
        cluster_weight = (
            1.0/(cluster_weight+10e-7).astype(np.double))  # 1/density
        cweights_.append(cluster_weight)
        print('# hic: {}, {} clusters, weights: {}'.format(
            i, num_clusters[i], cluster_weight))

    # log_hics, features, mats_, matpbs_,
    # [0, 1] from low to high level

    # creat graph
    graph_data = dict()

    fid_interacts = dict()
    for i, hic in enumerate(log_hics):
        cutoff = np.percentile(hic, cutoff_percent[i])
        # [src, dst] = np.nonzero(hic)
        fid = np.where(hic > cutoff)
        fid_interacts[i] = fid
        u = np.concatenate(fid[0].reshape(-1, 1))
        v = np.concatenate(fid[1].reshape(-1, 1))
        graph_data[('h{}_bead'.format(str(i)), 'interacts_{}'.format(
            str(i)), 'h{}_bead'.format(str(i)))] = (u, v)
    print('# create interacts_0,1')

    c_lists = dict()
    fid_clusters = dict()

    k = 0
    [src, dst] = np.where(mats_[k] < cutoff_cluster[str(k)])
    u = np.concatenate([src])
    v = np.concatenate([dst])
    graph_data[('h{}_bead'.format(str(k)), 'interacts_{}_sub'.format(
        str(k)), 'h{}_bead'.format(str(k)))] = (u, v)
    fid_clusters[k] = np.where(mats_[k] < cutoff_cluster[str(k)])
    print('# create interacts_0_sub')

    k = 1
    c_list = [r for r in range(cutoff_cluster[str(k)])]  # int(n_cluster)-1)
    c_lists[k] = c_list
    fid = []
    for i in c_list:
        [src, dst] = np.where(mats_[k] == i)
        u = np.concatenate([src])
        v = np.concatenate([dst])
        fid.append(np.where(mats_[k] == i))
        graph_data[('h{}_bead'.format(str(k)), 'interacts_{}_c{}'.format(
            str(k), str(i)), 'h{}_bead'.format(str(k)))] = (u, v)
    fid_clusters[k] = fid
    print('# create interacts_1_c')

    # chain constrain
    src = np.arange(0, log_hics[-1].shape[0]-1)
    dst = np.arange(1, log_hics[-1].shape[0])
    u = np.concatenate([src])
    v = np.concatenate([dst])
    graph_data[('h1_bead', 'bead_chain', 'h1_bead')] = (u, v)
    print('# create bead_chain')

    # src: center -> dst: bead
    for p in [[0, 1]]:
        x, y = p[0], p[1]
        array = idxs[x]
        W, S = ratios[y]/ratios[x], strides[y]/strides[x]
        start = np.arange(0, array.shape[-1], S).astype(int)

        src, dst = [], []
        for l in start:
            r = int(W/2) if l == 0 else int(W)+l
            r = min(r, array.shape[0])
            v = array[l:r].flatten()
            u = np.repeat(l/S, v.shape[-1]).flatten().astype(int)
            u, v = torch.tensor(u), torch.tensor(v)
            src.append(u)
            dst.append(v)
        src = torch.cat(src).int()
        dst = torch.cat(dst).int()
        graph_data[('h{}_bead'.format(y), 'center_{}_{}'.format(
            y, x), 'h{}_bead'.format(x))] = (src, dst)
    print('# create center')

    num_nodes_dict = {'h0_bead': len(idxs[0]), 'h1_bead': len(idxs[1])}
    g = dgl.heterograph(graph_data, num_nodes_dict, idtype=torch.long)
    print('# create heterograph')

    for i, idx in enumerate(idxs):
        g.nodes['h{}_bead'.format(str(i))].data['id'] = torch.tensor(
            idx.flatten(), dtype=torch.long)
        # g.nodes['h{}_bead'.format(str(i))].data['feat'] = torch.tensor(features[i]).float()

    for i, (m_hic, log_hic) in enumerate(zip(mats_, log_hics)):
        g.edges['interacts_{}'.format(i)].data['label'] = m_hic[tuple(
            fid_interacts[i])].clone().detach().flatten().type(torch.int8)
        # g.edges['interacts_{}'.format(i)].data['w'] = torch.tensor(log_hic[tuple(fid_interacts[i])]).clone().detach().flatten().float()
    print('# assign id & label')

    '''for key, c_list in c_lists.items():
        for i, n in enumerate(c_list):
            x = fid_clusters[key][i][0]
            y = fid_clusters[key][i][1]
            counts = torch.tensor(log_hics[key][x,y].reshape((-1,1)))
            value = counts*matpbs_[key][x,y,i].reshape((-1,1))
            g.edges['interacts_{}_c{}'.format(str(key), str(i))].data['w'] = torch.tensor(value).float()'''

    x = fid_clusters[0][0]
    y = fid_clusters[0][1]
    counts = torch.tensor(log_hics[0][x, y].reshape((-1, 1)))
    value = counts*matpbs_[0][x, y, i].reshape((-1, 1))
    g.edges['interacts_0_sub'.format(str(0), str(
        i))].data['w'] = torch.tensor(value).float()
    print('# assign weight')

    # print(g)
    top_list = ['interacts_1_c{}'.format(i)
                for i in np.arange(cutoff_cluster[str(1)])]
    top_list.append('bead_chain')
    top_subgraphs = g.edge_type_subgraph(top_list)
    top_graph = g.edge_type_subgraph(['interacts_1'])
    bottom_subgraphs = g.edge_type_subgraph(['interacts_0_sub'])
    bottom_graph = g.edge_type_subgraph(['interacts_0'])
    inter_graph = g.edge_type_subgraph(['center_1_0'])

    g_list = [top_graph, top_subgraphs,
              bottom_graph, bottom_subgraphs, inter_graph]

    return g, g_list, cweights_, mats_
