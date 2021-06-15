import os
import sys
import dgl
import torch
import torch_optimizer as optim
import numpy as np

from .model import embedding, encoder_bead, encoder_chain, encoder_union, decoder
from .loss import nllLoss
from .visualize import plot_feature, plot_X, plot_cluster, plot_confusion_mat

# import GPUtil
# gpuIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.05, maxMemory = 0.05, includeNan=False, excludeID=[], excludeUUID=[])
# device =  'cpu' if len(gpuIDs)==0 else 'cuda:{}'.format(gpuIDs[0])


def load_dataset(path, name):
    '''graph_dict[chromosome] = {top_graph, top_subgraphs, bottom_graph, inter_graph}
    feature_dict[chromosome] = {'hic_feat_h0', 'hic_feat_h1'}
    HiCDataset[i]: graph[i], feature[i], label[i](chromosome)'''
    HiCDataset = torch.load(os.path.join(path, name))
    return HiCDataset


def create_network(configuration, graph, device):
    config = configuration['parameter']
    top_graph = graph['top_graph']
    top_subgraphs = graph['top_subgraphs']
    bottom_graph = graph['bottom_graph']
    inter_graph = graph['inter_graph']

    sampling_num = config['G3DM']['sampling_num']
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [ int(sampling_num['l0']), int(sampling_num['l1']),int(sampling_num['l2'])] )
    '''sampler = dgl.dataloading.MultiLayerNeighborSampler([
        {('h0_bead', 'interacts_0', 'h0_bead'): int(sampling_num['l0'])},
        {('h0_bead', 'interacts_0', 'h0_bead'): int(sampling_num['l1'])},
        {('h0_bead', 'interacts_0', 'h0_bead'): int(sampling_num['l2'])}])'''  # ,('h1_bead', 'center_1_0', 'h0_bead'):0

    ind0 = int(config['feature']['in_dim']['h0'])
    outd0 = int(config['feature']['out_dim']['h0'])
    ind1 = int(config['feature']['in_dim']['h1'])
    outd1 = int(config['feature']['out_dim']['h1'])
    em_h0_bead = embedding(ind0+ind1, outd0).to(device)
    em_h1_bead = embedding(ind1, outd1).to(device)

    nh0 = int(config['G3DM']['num_heads']['0'])
    nh1 = int(config['G3DM']['num_heads']['1'])
    nhout = int(config['G3DM']['num_heads']['out'])

    chain = config['G3DM']['graph_chain_dim']
    cin, chidden, cout = int(chain['in_dim']), int(
        chain['hidden_dim']), int(chain['out_dim'])
    e_list = ['interacts_1_c{}'.format(i) for i in np.arange(
        int(config['graph']['cutoff_cluster']))]
    en_chain_net = encoder_chain(
        cin, chidden, cout, num_heads=nh1, etypes=e_list).to(device)

    bead = config['G3DM']['graph_bead_dim']
    bdin, bdhidden, bdout = int(bead['in_dim']), int(
        bead['hidden_dim']), int(bead['out_dim'])

    en_bead_net = encoder_bead(bdin, bdhidden, bdout).to(device)
    en_union = encoder_union(in_h1_dim=cout, in_h0_dim=bdout, out_dim=3,
                             in_h1_heads=nh1, in_h0_heads=nh0, out_heads=nhout).to(device)

    nc0 = int(config['graph']['num_clusters']['0'])
    nc1 = int(config['graph']['num_clusters']['1'])
    de_center_net = decoder(nh1, nc1, 'h1_bead', 'interacts_1').to(device)
    # de_bead_net = decoder(nhout, nc0, 'h0_bead', 'interacts_0').to(device)
    de_bead_net = decoder(nhout, nc0, '_N', '_E').to(device)

    nll = nllLoss().to(device)

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
    ae_networks = [en_chain_net, en_bead_net,
                   en_union, de_center_net, de_bead_net]
    return sampler, em_networks, ae_networks, nll, opt


def setup_train(configuration):
    itn = int(configuration['parameter']['G3DM']['iteration'])
    batch_size = int(configuration['parameter']['G3DM']['batchsize'])
    return itn, batch_size


def fit_one_step(graphs, features, sampler, batch_size, em_networks, ae_networks, loss_fc, optimizer, device):
    top_graph = graphs['top_graph'].to(device)
    top_subgraphs = graphs['top_subgraphs'].to(device)
    bottom_graph = dgl.to_homogeneous(graphs['bottom_graph'], edata=['w', 'label'], store_type=True)
    inter_graph = graphs['inter_graph'].to(device)

    h0_feat = features[0]
    h1_feat = features[1]

    em_h0_bead, em_h1_bead = em_networks[0], em_networks[1]
    en_chain_net, en_bead_net = ae_networks[0], ae_networks[1]
    en_union = ae_networks[2]
    de_center_net, de_bead_net = ae_networks[3], ae_networks[4]

    # eid_dict = {etype: bottom_graph.edges( etype=etype, form='eid') for etype in bottom_graph.etypes}
    # dataloader = dgl.dataloading.EdgeDataLoader(bottom_graph, {'interacts_0': eid_dict['interacts_0']}, sampler, device=device,
    #                                             batch_size=batch_size, shuffle=True, drop_last=True)
    dataloader = dgl.dataloading.EdgeDataLoader(bottom_graph, bottom_graph.nodes(), sampler, device=device,
                                                batch_size=batch_size, shuffle=True, drop_last=True)
    top_list = [e for e in top_subgraphs.etypes if 'interacts_1_c' in e]
    top_list.append('bead_chain')

    loss_list = []
    for input_nodes, pair_graph, blocks in dataloader:
        blocks = [b.to(device) for b in blocks]
        X1 = em_h1_bead(h1_feat)
        h_center = en_chain_net(
            top_subgraphs, X1, top_list, ['w'], ['h1_bead'])

        inputs0 = torch.tensor(h0_feat[input_nodes.cpu().detach(
        ), :], dtype=torch.float).to(device)  # ['h0_bead']
        X0 = em_h0_bead(inputs0)
        h_bead = en_bead_net(blocks, X0, ['interacts_0'], ['w'])

        h0 = dgl.in_subgraph(inter_graph, {'h0_bead': blocks[2].dstnodes()}).edges()[1]  # dst
        h0, _ = torch.sort(torch.unique(h0))
        h1 = dgl.in_subgraph(inter_graph, {'h0_bead': blocks[2].dstnodes()}).edges()[0]  # src
        h1, _ = torch.sort(torch.unique(h1))
        inter = dgl.node_subgraph(inter_graph, {'h0_bead': h0, 'h1_bead': h1})

        c = h_center[h1, :, :].to(device)
        res = en_union(inter, c, h_bead)

        xp1, _ = de_center_net(top_graph, h_center)
        xp0, _ = de_bead_net(pair_graph, res)

        xt1 = top_graph.edges['interacts_1'].data['label']
        xt0 = pair_graph.edges['_E'].data['label']

        loss1 = loss_fc(xp1, xt1)
        loss0 = loss_fc(xp0, xt0)
        loss = (loss0 + loss1)/2.0
        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # retain_graph=False,
        optimizer.step()
        loss_list.append(loss.item())

    ll = np.array(loss_list)
    return np.nan if len(ll)==0 else np.mean(ll)


def inference(graphs, features, num_heads, em_networks, ae_networks, device):
    top_graph = graphs['top_graph'].to(device)
    top_subgraphs = graphs['top_subgraphs'].to(device)
    bottom_graph = (dgl.to_homogeneous(graphs['bottom_graph'], edata=['w', 'label']))
    inter_graph = graphs['inter_graph'].to(device)

    h0_feat = features[0]
    h1_feat = features[1]

    em_h0_bead, em_h1_bead = em_networks[0], em_networks[1]
    en_chain_net, en_bead_net = ae_networks[0], ae_networks[1]
    en_union = ae_networks[2]
    de_center_net, de_bead_net = ae_networks[3], ae_networks[4]

    # eid_dict = {etype: bottom_graph.edges(etype=etype, form='eid') for etype in bottom_graph.etypes}
            
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    batch_size = 20 # bottom_graph.number_of_nodes()
    dataloader = dgl.dataloading.NodeDataLoader(bottom_graph, 
                                                torch.arange(bottom_graph.number_of_nodes()), 
                                                sampler, device=device,
                                                batch_size=batch_size, shuffle=False, drop_last=False)
    top_list = [e for e in top_subgraphs.etypes if 'interacts_1_c' in e]
    top_list.append('bead_chain')

    loss_list = []
    result = torch.tensor(torch.empty((bottom_graph.number_of_nodes(), num_heads, 3)))

    with torch.no_grad():

        for input_nodes, output_nodes, blocks in dataloader:
            # input_nodes = input_nodes.to(device)
            # output_nodes = input_nodes.to(device)
            blocks = [b.to(device) for b in blocks]

            X1 = em_h1_bead(h1_feat)
            h_center = en_chain_net( top_subgraphs, X1, top_list, ['w'], ['h1_bead'])

            inputs0 = torch.tensor(h0_feat[input_nodes.cpu().detach(), :], dtype=torch.float).to(device)  # ['h0_bead']
            X0 = em_h0_bead(inputs0)
            # h_bead = en_bead_net(blocks, X0, ['interacts_0'], ['w'])
            h_bead = en_bead_net(blocks, X0, [], ['w'])

            h0 = dgl.in_subgraph(inter_graph, {'h0_bead': blocks[2].dstnodes()}).edges()[1]  # dst
            h0, _ = torch.sort(torch.unique(h0))
            h1 = dgl.in_subgraph(inter_graph, {'h0_bead': blocks[2].dstnodes()}).edges()[0]  # src
            h1, _ = torch.sort(torch.unique(h1))
            inter = dgl.node_subgraph(inter_graph, {'h0_bead': h0, 'h1_bead': h1})

            c = h_center[h1, :, :].to(device)
            res = en_union(inter, c, h_bead)
            result[output_nodes.cpu().detach(),:,:] = res.cpu().detach()

        xp1, _ = de_center_net(top_graph, h_center)
        xp0, _ = de_bead_net(bottom_graph.to(device), result.to(device))

        p1 = xp1.cpu().detach().numpy()
        tp1 = graphs['top_graph'].edges['interacts_1'].data['label'].cpu().detach().numpy()
        center_X = h_center.cpu().detach().numpy()
        center_cluster_mat = np.ones((center_X.shape[0], center_X.shape[0]))*tp1.max()
        xs,ys = graphs['top_graph'].edges(etype='interacts_1', form='uv')[0], graphs['top_graph'].edges(etype='interacts_1', form='uv')[1]
        center_cluster_mat[xs, ys] = np.argmax(p1, axis=1)
        true_center = np.ones((center_X.shape[0], center_X.shape[0]))*tp1.max()
        true_center[xs, ys] = tp1
        print(p1.shape, tp1.shape, tp1.max(), p1.max())

        p0 = xp0.cpu().detach().numpy()
        tp0 = bottom_graph.edges['_E'].data['label'].cpu().detach().numpy()
        bead_X = result.cpu().detach().numpy()
        bead_cluster_mat = np.ones((bead_X.shape[0], bead_X.shape[0]))*tp0.max()
        xs, ys = bottom_graph.edges()[0], bottom_graph.edges()[1]
        bead_cluster_mat[xs, ys] = np.argmax(p0, axis=1)

        true_bead = np.ones((bead_X.shape[0], bead_X.shape[0]))*tp0.max()
        true_bead[xs, ys] = tp0
        return center_X, bead_X, center_cluster_mat, bead_cluster_mat, true_center, true_bead


def run_epoch(dataset, model, loss_fc, optimizer, sampler, batch_size, iterations, device, writer=None, config=None):
    em_networks, ae_networks = model
    loss_list = []

    for i in np.arange(iterations):
        for j, data in enumerate(dataset):
            graphs, features, _ = data
            h0_f = features['hic_h0']['feat']
            h0_p = features['hic_h0']['pos']
            h0_feat = torch.stack( [torch.tensor(h0_f, dtype=torch.float), 
                                    torch.tensor(h0_p, dtype=torch.float)], 
                                    dim=1).to(device)
            h1_f = features['hic_h1']['feat']
            h1_p = features['hic_h1']['pos']
            h1_feat = torch.stack( [torch.tensor(h1_f, dtype=torch.float),
                                    torch.tensor(h1_p, dtype=torch.float)], 
                                    dim=1).to(device)

            ll = fit_one_step(graphs, [h0_feat, h1_feat], sampler, batch_size, em_networks, ae_networks, loss_fc, optimizer, device)
            loss_list.append(ll)

            if i == 0 and j == 0 and writer is not None:
                plot_feature(h0_f, h0_p, writer, '0, features/h0')
                plot_feature(h1_f, h1_p, writer, '0, features/h1')
            
            if i%5==0 and j == 0 and writer is not None and config is not None:
                num_heads = int(config['parameter']['G3DM']['num_heads']['out'])
                center_X, bead_X, center_cluster_mat, bead_cluster_mat, center_true, bead_true = inference(graphs, [h0_feat, h1_feat], num_heads, em_networks, ae_networks, device)
                plot_X(center_X, writer, '1, 3D/center', step=i)
                plot_X(bead_X, writer, '1, 3D/bead', step=i)

                plot_cluster(center_cluster_mat, writer, '2,1 cluster/center', step=i)
                plot_cluster(bead_cluster_mat, writer, '2,1 cluster/bead', step=i)
                plot_confusion_mat(center_cluster_mat, center_true,  writer, '2,2 confusion matrix/center', step=i)
                plot_confusion_mat(bead_cluster_mat, bead_true, writer, '2,2 confusion matrix/bead', step=i)


        print("epoch {:d} Loss {:f}".format(i, np.nanmean(np.array(loss_list))))

