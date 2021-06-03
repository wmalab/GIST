import os, sys
import dgl
import torch
import torch_optimizer as optim
import numpy as np

from .model import *
from .loss import *

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from feature.model import embedding

device = 'cpu'

def load_dataset(path, name):
    '''graph_dict[chromosome] = {top_graph, top_subgraphs, bottom_graph, inter_graph}
    feature_dict[chromosome] = {'hic_feat_h0', 'hic_feat_h1'}
    HiCDataset[i]: graph[i], feature[i], label[i](chromosome)'''
    HiCDataset = torch.load(os.path.join( path, name))
    return HiCDataset

def create_network(configuration, graph):
    config = configuration['parameter']
    top_graph = graph['top_graph'].to(device)
    top_subgraphs = graph['top_subgraphs'].to(device)
    bottom_graph = graph['bottom_graph'].to(device)
    inter_graph = graph['inter_graph'].to(device)

    sampling_num = config['G3DM']['sampling_num']
    sampler = dgl.dataloading.MultiLayerNeighborSampler([
        {('h0_bead', 'interacts_0', 'h0_bead'): int(sampling_num['l0'])},
        {('h0_bead', 'interacts_0', 'h0_bead'): int(sampling_num['l1'])},
        {('h0_bead', 'interacts_0', 'h0_bead'): int(sampling_num['l2'])}])  # ,('h1_bead', 'center_1_0', 'h0_bead'):0
    
    ind0 = int(config['feature']['in_dim']['h0'])
    outd0 = int(config['feature']['out_dim']['h0'])
    ind1 = int(config['feature']['in_dim']['h1'])
    outd1 = int(config['feature']['out_dim']['h1'])
    em_h0_bead = embedding(ind0, outd0).to(device)
    em_h1_bead = embedding(ind1, outd1).to(device)
    
    nh0 = int(config['G3DM']['num_heads']['0'])
    nh1 = int(config['G3DM']['num_heads']['1'])
    nhout = int(config['G3DM']['num_heads']['out'])
    
    chain = config['G3DM']['graph_chain_dim']
    cin, chidden, cout = int(chain['in_dim']), int(chain['hidden_dim']), int(chain['out_dim'])
    e_list = ['interacts_1_c{}'.format(i) for i in np.arange(int(config['graph']['cutoff_cluster']))]
    en_chain_net = encoder_chain(cin, chidden, cout, num_heads=nh1, etypes=e_list).to(device)

    bead = config['G3DM']['graph_bead_dim']
    bdin, bdhidden, bdout = int(bead['in_dim']), int(bead['hidden_dim']), int(bead['out_dim'])

    en_bead_net = encoder_bead(bdin, bdhidden, bdout).to(device)
    en_union = encoder_union(in_h1_dim=cout, in_h0_dim=bdout, out_dim=3,
                             in_h1_heads=nh1, in_h0_heads=nh0, out_heads=nhout).to(device)

    nc0 = int(config['graph']['num_clusters']['0'])
    nc1 = int(config['graph']['num_clusters']['1'])
    de_center_net = decoder(nh1, nc1, 'h1_bead', 'interacts_1').to(device)
    de_bead_net = decoder(nh0, nc0, 'h0_bead', 'interacts_0').to(device)

    nll = nllLoss()

    opt = optim.AdaBound(list(em_h0_bead.parameters()) + list(em_h1_bead.parameters()) +
                         list(en_chain_net.parameters()) + list(en_bead_net.parameters()) + list(en_union.parameters()) +
                         list(de_center_net.parameters()) +
                         list(de_bead_net.parameters()),
                         lr=1e-3, betas=(0.9, 0.999),
                         final_lr=0.1, gamma=1e-3,
                         eps=1e-8, weight_decay=0,
                         amsbound=False,
                         )
    em_networks = [em_h0_bead, em_h1_bead]
    ae_networks = [en_chain_net, en_bead_net, en_union, de_center_net, de_bead_net]
    return sampler, em_networks, ae_networks, nll, opt

def setup_train(configuration):
    itn = int(configuration['parameter']['G3DM']['iteration'])
    batch_size = int(configuration['parameter']['G3DM']['batchsize'])
    return itn, batch_size

def fit_one_step(graphs, features, sampler, batch_size, em_networks, ae_networks, loss_fc, optimizer):
    top_graph, top_subgraphs, bottom_graph, inter_graph = graphs[0], graphs[1], graphs[2], graphs[3]
    h0_feat = features[0]
    h1_feat = features[1]

    em_h0_bead, em_h1_bead = em_networks[0], em_networks[1]
    en_chain_net, en_bead_net = ae_networks[0], ae_networks[1]
    en_union = ae_networks[2]
    de_center_net, de_bead_net = ae_networks[3], ae_networks[4]

    eid_dict = {etype: bottom_graph.edges(etype=etype, form='eid') for etype in bottom_graph.etypes}
    dataloader = dgl.dataloading.EdgeDataLoader(bottom_graph, {'interacts_0': eid_dict['interacts_0']}, sampler,
                                                batch_size=batch_size, shuffle=True, drop_last=True)
    top_list = [e for e in top_graph.etypes if 'interacts_1_c' in e]
    top_list.append('bead_chain')

    loss_list = []
    for input_nodes, pair_graph, blocks in dataloader:
        # print('input: ', input_nodes)
        # print('output: ', pair_graph)
        X1 = em_h1_bead(h1_feat)
        h_center = en_chain_net(top_graph, X1, top_list, ['w'], ['h1_bead'])

        inputs0 = h0_feat[:]  # ['h0_bead']
        X0 = em_h0_bead(inputs0)
        h_bead = en_bead_net(blocks, X0, ['interacts_0'], ['w'])

        h0 = dgl.in_subgraph(inter_graph, {'h0_bead': blocks[2].dstnodes()}).edges()[1]  # dst
        h0, _ = torch.sort(torch.unique(h0))
        h1 = dgl.in_subgraph(inter_graph, {'h0_bead': blocks[2].dstnodes()}).edges()[0]  # src
        h1, _ = torch.sort(torch.unique(h1))
        inter = dgl.node_subgraph(inter_graph, {'h0_bead': h0, 'h1_bead': h1})

        c = h_center[h1, :, :]
        res = en_union(inter, c, h_bead)

        xp1, _ = de_center_net(top_subgraphs, h_center)
        # xp1 = xp1[('h1_bead', 'interacts_1', 'h1_bead')]
        xp0, _ = de_bead_net(pair_graph, res)

        xt1 = top_subgraphs.edges['interacts_1'].data['label']
        xt0 = pair_graph.edges['interacts_0'].data['label']

        loss1 = loss_fc(xp1, xt1)
        loss0 = loss_fc(xp0, xt0)
        loss = loss0 + loss1
        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # retain_graph=False,
        optimizer.step()
        loss_list.append(loss.item())
        print(" Loss {:f} {:f}".format(loss1.item(), loss0.item()))