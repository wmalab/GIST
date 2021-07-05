import dgl
import torch
import numpy as np
import os
from .utils import cluster_hic, log1p_hic
# from embedding.model import embedding, tilt_hic, position_hic

'''def create_hierarchical_graph_1lvl(hic, feats, num_cluster, percentile=40):
    cutoff = np.percentile(hic, percentile)
    idxy = np.arange(hic.shape[0])
    tmp = hic.copy()
    tmp[tmp<=cutoff] = 0
    mat_, matbp_ = cluster_hic(tmp, tmp, n_cluster=num_cluster)
    [src, dst] = np.where(hic > cutoff)
    u = np.concatenate([src])
    v = np.concatenate([dst])
    graph_data = {('bead', 'interacts', 'bead'): (u, v)}

    num_nodes_dict = {'bead': len(idxy)}
    g = dgl.heterograph(graph_data, num_nodes_dict)
    counts = torch.tensor(hic[u,v].reshape((-1,1)))
    label = torch.tensor(mat_[u,v].view((-1,)))
    g.edges['interacts'].data['counts'] = counts.float()
    g.edges['interacts'].data['label'] = label.long()
    # np.random.seed(10)
    g.nodes['bead'].data['id'] = torch.tensor(idxy.flatten())
    g.nodes['bead'].data['feat'] = torch.tensor(feats).float()
    return g, [u, v], mat_, matbp_ '''

def create_hierarchical_graph_2lvl(norm_hics, num_clusters, ratios, strides, cutoff_percent={0:10, 1:10}, cutoff_cluster={0:4, 1:6}):
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
        cluster_weight, _ = np.histogram(m.view(-1,1), bins=np.arange(num_clusters[i]), density=True)
        cluster_weight = np.append(cluster_weight, [1.0])
        cluster_weight = (1.0/(cluster_weight+10e-7).astype(np.double)) # 1/density
        cweights_.append(cluster_weight)
        print('# hic: {}, {} clusters, weights: {}'.format(i, num_clusters[i], cluster_weight))

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
        u = np.concatenate(fid[0].reshape(-1,1))
        v = np.concatenate(fid[1].reshape(-1,1))
        graph_data[('h{}_bead'.format(str(i)), 'interacts_{}'.format(str(i)), 'h{}_bead'.format(str(i)))] = (u, v)
    print('# create interacts_0,1')

    c_lists = dict()
    fid_clusters = dict()

    k = 0
    [src, dst] = np.where( mats_[k]< cutoff_cluster[str(k)] )
    u = np.concatenate([src])
    v = np.concatenate([dst])
    graph_data[('h{}_bead'.format(str(k)), 'interacts_{}_sub'.format(str(k)), 'h{}_bead'.format(str(k)))] = (u, v)
    fid_clusters[k] = np.where( mats_[k]< cutoff_cluster[str(k)] )
    print('# create interacts_0_sub')

    k = 1
    c_list = [r for r in range(cutoff_cluster[str(k)])] # int(n_cluster)-1)
    c_lists[k] = c_list
    fid = []
    for i in c_list:
        [src, dst] = np.where( mats_[k]==i )
        u = np.concatenate([src])
        v = np.concatenate([dst])
        fid.append(np.where(mats_[k]==i))
        graph_data[('h{}_bead'.format(str(k)), 'interacts_{}_c{}'.format(str(k), str(i)), 'h{}_bead'.format(str(k)))] = (u, v)
    fid_clusters[k] = fid
    print('# create interacts_1_c')

    # chain constrain
    src = np.arange(0,log_hics[-1].shape[0]-1)
    dst = np.arange(1,log_hics[-1].shape[0])
    u = np.concatenate([src])
    v = np.concatenate([dst])
    graph_data[('h1_bead', 'bead_chain', 'h1_bead')] = (u, v)
    print('# create bead_chain')

    # src: center -> dst: bead
    for p in [[0,1]]:
        x,y = p[0],p[1]
        array = idxs[x]
        W, S = ratios[y]/ratios[x], strides[y]/strides[x]
        start = np.arange(0, array.shape[-1], S).astype(int)
 
        src, dst = [], []
        for l in start:
            r = int(W/2) if l==0 else int(W)+l
            r = min(r, array.shape[0])
            v = array[l:r].flatten()
            u = np.repeat(l/S, v.shape[-1]).flatten().astype(int)
            u, v = torch.tensor(u), torch.tensor(v)
            src.append(u)
            dst.append(v)
        src = torch.cat(src).int()
        dst = torch.cat(dst).int()
        graph_data[('h{}_bead'.format(y), 'center_{}_{}'.format(y,x), 'h{}_bead'.format(x))] = (src, dst)
    print('# create center')

    num_nodes_dict = {'h0_bead': len(idxs[0]), 'h1_bead':len(idxs[1])}
    g = dgl.heterograph(graph_data, num_nodes_dict, idtype=torch.long)
    print('# create heterograph')

    for i, idx in enumerate(idxs):
        g.nodes['h{}_bead'.format(str(i))].data['id'] = torch.tensor(idx.flatten(), dtype=torch.long)
        # g.nodes['h{}_bead'.format(str(i))].data['feat'] = torch.tensor(features[i]).float()

    for i, (m_hic, log_hic) in enumerate(zip(mats_, log_hics)):
        g.edges['interacts_{}'.format(i)].data['label'] = m_hic[tuple(fid_interacts[i])].clone().detach().flatten().type(torch.int8)
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
    counts = torch.tensor(log_hics[0][x,y].reshape((-1,1)))
    value = counts*matpbs_[0][x,y,i].reshape((-1,1))
    g.edges['interacts_0_sub'.format(str(0), str(i))].data['w'] = torch.tensor(value).float()
    print('# assign weight')

    # print(g)
    top_list = ['interacts_1_c{}'.format(i) for i in np.arange(cutoff_cluster[str(1)])]
    top_list.append('bead_chain')
    top_subgraphs = g.edge_type_subgraph(top_list)
    top_graph = g.edge_type_subgraph(['interacts_1'])
    bottom_subgraphs = g.edge_type_subgraph(['interacts_0_sub'])
    bottom_graph = g.edge_type_subgraph(['interacts_0'])
    inter_graph = g.edge_type_subgraph(['center_1_0'])
    print(top_graph, top_subgraphs, bottom_graph, bottom_subgraphs, inter_graph, sep='\n')
    g_list = [top_graph, top_subgraphs, bottom_graph, bottom_subgraphs, inter_graph]

    print("idtype: ", g.idtype, top_subgraphs.idtype, sep=' ')
    return g, g_list, cweights_, mats_