import time
import dgl
import torch
import torch_optimizer as optim
import numpy as np
from G3DM.build_graph import create_hierarchical_graph_1lvl, create_hierarchical_graph_2lvl

from G3DM.utils import load_hic, log1p_hic
from G3DM.loss import nllLoss
from G3DM.data_prepare import hic_prepare_pooling
from embedding.model import embedding, tilt_hic

from G3DM.model import encoder_chain, encoder_bead, encoder_union, decoder

device = 'cpu'


def stochastic(graph):
    sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 5, 5])
    # The following arguments are specific to NodeDataLoader.
    train_dataloader = dgl.dataloading.EdgeDataLoader(
        graph,                                  # The graph
        torch.arange(graph.number_of_edges()),  # The edges to iterate over
        sampler,                                # The neighbor sampler
        device=device,                          # Put the MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=1,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )
    input_nodes, pos_graph, mfgs = next(iter(train_dataloader))
    print('Number of input nodes:', len(input_nodes))
    print('Positive graph # nodes:', pos_graph.number_of_nodes(),
          '# edges:', pos_graph.number_of_edges())
    print(mfgs)


'''def run_1lvl(hic, graph, num_clusters):

    or_dim = 15
    in_dim, hidden_dim, out_dim = 15, 6, 3
    num_heads = 4
    num_clusters = num_clusters

    sampler = dgl.dataloading.MultiLayerNeighborSampler([10, 20, 80])
    dataloader = dgl.dataloading.EdgeDataLoader(
        # The following arguments are specific to NodeDataLoader.
        graph,                                  # The graph
        torch.arange(graph.number_of_edges()),  # The edges to iterate over
        sampler,                                # The neighbor sampler
        device=device,                          # Put the MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=60,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )

    em_bead = embedding(or_dim, in_dim)
    net_encoder = encoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, num_heads=num_heads)
    # nll = nllLoss()
    hic_decoder = decoder(num_heads=num_heads, num_clusters=num_clusters, ntype='bead')
    X_decoder = decoder(num_heads=num_heads, num_clusters=num_clusters, ntype='bead')

    opt = optim.AdaBound( list(em_bead.parameters()) + list(net_encoder.parameters()),
                        lr= 1e-3, betas= (0.9, 0.999),
                        final_lr = 0.1, gamma=1e-3,
                        eps= 1e-8, weight_decay=0,
                        amsbound=False,
                        )

    dur = []
    itn = 100
    nll = nllLoss()
    for epoch in range(itn):
        if epoch >= 3:
            t0 = time.time()
        loss_list = []
        for input_nodes, pair_graph, mfgs in dataloader:
            inputs = mfgs[0].srcdata['feat']
            X = em_bead(inputs)
            X = net_encoder(mfgs, X)
            xp, mp = X_decoder(pair_graph, X)
            xt = pair_graph.edges['interacts'].data['label']
            # H = mfgs[2].edges['ineracts'].ndata['counts']
            # xp = torch.softmax(xp, dim=-1)
            # xt = torch.softmax(xt, dim=-1)

            loss = nll(xp, xt)
            opt.zero_grad()
            loss.backward(retain_graph = True) # retain_graph=False, 
            opt.step()
            loss_list.append(loss.item())
            # print(loss.item())

        if epoch >= 3:
            dur.append(time.time() - t0)
        print("Epoch {:05d} | Loss {:f} | Time(s) {:.4f}".format(epoch, np.mean(loss_list), np.mean(dur)))

    X = inference(em_bead, net_encoder, graph, 'cpu', num_heads, 3)
'''


def run_2lvl(graph, hics, num_clusters):
    or0_dim = 93
    or1_dim = 31
    in_dim, hidden_dim, out_dim = 15, 6, 3
    num_heads = 4

    print(graph)
    nid_dict = {ntype: graph.nodes(ntype=ntype) for ntype in graph.ntypes}
    eid_dict = {etype: graph.edges(etype=etype, form='eid')
                for etype in graph.etypes}

    top_all_graph = graph.edge_type_subgraph(['interacts_1']).to(device)
    top_graph = graph.edge_type_subgraph(['interacts_1_c0', 'interacts_1_c1',
                                          'interacts_1_c2', 'interacts_1_c3', 'bead_chain']).to(device)
    bottom_graph = graph.edge_type_subgraph(['interacts_0']).to(device)
    inter_graph = graph.edge_type_subgraph(['center_1_0']).to(device)

    print('top: ', top_graph)
    print('bottom: ', bottom_graph)

    sampler = dgl.dataloading.MultiLayerNeighborSampler([
        {('h0_bead', 'interacts_0', 'h0_bead'): 5},
        {('h0_bead', 'interacts_0', 'h0_bead'): 10},
        {('h0_bead', 'interacts_0', 'h0_bead'): 15}])  # ,('h1_bead', 'center_1_0', 'h0_bead'):0
    dataloader = dgl.dataloading.EdgeDataLoader(bottom_graph, {'interacts_0': eid_dict['interacts_0']}, sampler,
                                                batch_size=100, shuffle=True, drop_last=True)

    em_h0_bead = embedding(or0_dim, in_dim).to(device)
    em_h1_bead = embedding(or1_dim, in_dim).to(device)

    en_chain_net = encoder_chain(15, 6, 3, num_heads=4,
                                 etypes=['interacts_1_c0', 'interacts_1_c1',
                                         'interacts_1_c2', 'interacts_1_c3']).to(device)
    en_bead_net = encoder_bead(15, 6, 4).to(device)
    en_union = encoder_union(in_h1_dim=3, in_h0_dim=4, out_dim=3,
                             in_h1_heads=4, in_h0_heads=5, out_heads=8).to(device)
    de_center_net = decoder(4, 6, 'h1_bead', 'interacts_1').to(device)
    de_bead_net = decoder(8, 10, 'h0_bead', 'interacts_0').to(device)

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

    print('loading ...')
    loss_list = []
    for input_nodes, pair_graph, blocks in dataloader:
        # print('input: ', input_nodes)
        # print('output: ', pair_graph)
        inputs1 = top_graph.srcdata['feat']
        X1 = em_h1_bead(inputs1)
        h_center = en_chain_net(top_graph, X1,
                                ['interacts_1_c0', 'interacts_1_c1',
                                    'interacts_1_c2', 'interacts_1_c3', 'bead_chain'],
                                ['w'], ['h1_bead'])

        inputs0 = blocks[0].srcdata['feat']  # ['h0_bead']
        X0 = em_h0_bead(inputs0)
        h_bead = en_bead_net(blocks, X0, ['interacts_0'], ['w'])

        h0 = dgl.in_subgraph(inter_graph, {'h0_bead': blocks[2].dstnodes()}).edges()[
            1]  # dst
        h0, _ = torch.sort(torch.unique(h0))
        h1 = dgl.in_subgraph(inter_graph, {'h0_bead': blocks[2].dstnodes()}).edges()[
            0]  # src
        h1, _ = torch.sort(torch.unique(h1))
        inter = dgl.node_subgraph(inter_graph, {'h0_bead': h0, 'h1_bead': h1})

        c = h_center[h1, :, :]
        res = en_union(inter, c, h_bead)

        xp1, _ = de_center_net(top_all_graph, h_center)
        # xp1 = xp1[('h1_bead', 'interacts_1', 'h1_bead')]
        xp0, _ = de_bead_net(pair_graph, res)

        xt1 = top_all_graph.edges['interacts_1'].data['label']
        xt0 = pair_graph.edges['interacts_0'].data['label']

        loss1 = nll(xp1, xt1)
        loss0 = nll(xp0, xt0)
        loss = loss0 + loss1
        opt.zero_grad()
        loss.backward(retain_graph=True)  # retain_graph=False,
        opt.step()
        loss_list.append(loss.item())
        print(" Loss {:f} {:f}".format(loss1.item(), loss0.item()))

    # inference(em_bead, net_encoder, X_decoder, graph, 'cpu', num_heads, 3)


if __name__ == '__main__':
    '''norm_hics, ratios = hic_prepare(
        rawfile='Dixon2012-H1hESC-HindIII-allreps-filtered.500kb.cool', 
        chromosome='chr20', ratios=[4], remove_zero_col = True)

    for i, m in enumerate(norm_hics):
        print('level id: {}, iced normalization Hi-C shape: {}'.format(i, m.shape))

    log_hic = log1p_hic(norm_hics[0])
    feats = tilt_hic(log_hic, dim=15)
    num_clusters = 7
    graph, _, _, _ = create_hierarchical_graph_1lvl(log_hic, feats, percentile=65, num_cluster=num_clusters)

    run(log_hic, graph, num_clusters=num_clusters)'''

    ratios = np.array([1, 8])
    strides = np.array([1, 2])
    norm_hics, ratios = hic_prepare_pooling(
        rawfile='Dixon2012-H1hESC-HindIII-allreps-filtered.500kb.cool',
        chromosome='chr20', ratios=ratios, strides=strides, remove_zero_col=False)

    for i, m in enumerate(norm_hics):
        print('level id: {}, iced normalization Hi-C shape: {}'.format(i, m.shape))

    num_clusters = np.array([10, 6])
    g = create_hierarchical_graph_2lvl(
        norm_hics, num_clusters, ratios, strides)
    run_2lvl(g, None, None)
