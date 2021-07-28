import os
import sys, time, GPUtil
import dgl
import torch
import torch_optimizer as optim
import numpy as np

from .model import embedding, encoder_chain, decoder
from .loss import nllLoss, stdLoss, WassersteinLoss, ClusterWassersteinLoss
from .visualize import plot_feature, plot_X, plot_cluster, plot_confusion_mat, plot_lines
from .visualize import plot_scaler

# import GPUtil
# gpuIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.05, maxMemory = 0.05, includeNan=False, excludeID=[], excludeUUID=[])
# device =  'cpu' if len(gpuIDs)==0 else 'cuda:{}'.format(gpuIDs[0])


def load_dataset(path, name):
    '''graph_dict[chromosome] = {top_graph, top_subgraphs, bottom_graph, inter_graph}
    feature_dict[chromosome] = {'hic_feat_h0', 'hic_feat_h1'}
    HiCDataset[i]: graph[i], feature[i], label[i](chromosome)'''
    HiCDataset = torch.load(os.path.join(path, name))
    return HiCDataset


def create_network(configuration, device):
    config = configuration['parameter']
    # top_graph = graph['top_graph']
    # top_subgraphs = graph['top_subgraphs']

    ind = int(config['feature']['in_dim'])
    outd = int(config['feature']['out_dim'])

    em_bead = embedding(in_dim=ind, out_dim=outd, in_num_channels=3).to(device)

    nh = int(config['G3DM']['num_heads'])

    chain = config['G3DM']['graph_dim']
    cin, chidden, cout = int(chain['in_dim']), int( chain['hidden_dim']), int(chain['out_dim'])
    e_list = ['interacts_c{}'.format(i) for i in np.arange( int(config['graph']['cutoff_cluster']))]
    en_net = encoder_chain( cin, chidden, cout, num_heads=nh, etypes=e_list).to(device)

    nc = int(config['graph']['num_clusters'])
    de_net = decoder(nh, nc, 'bead', 'interacts').to(device)

    nll = nllLoss().to(device)
    stdl = stdLoss().to(device)
    cwnl = ClusterWassersteinLoss(device).to(device)

    opt = optim.AdaBound(list(em_bead.parameters()) + list(en_net.parameters()) + list(de_net.parameters()),
                        lr=1e-2, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3, eps=1e-8, weight_decay=0,
                        amsbound=False,
                        )
                         

    em_networks = [em_bead]
    ae_networks = [en_net, de_net]
    return em_networks, ae_networks, [nll, cwnl], [opt]


def setup_train(configuration):
    itn = int(configuration['parameter']['G3DM']['iteration'])
    batch_size = int(configuration['parameter']['G3DM']['batchsize'])
    return itn, batch_size


def fit_one_step(graphs, features, cluster_weights, batch_size, em_networks, ae_networks, loss_fc, optimizer, device):
    top_graph = graphs['top_graph'].to(device)
    top_subgraphs = graphs['top_subgraphs'].to(device)

    cw = cluster_weights
    ncluster = len(cluster_weights)

    h_feat = features

    em_bead = em_networks[0]
    en_net = ae_networks[0]
    de_net = ae_networks[1]

    top_list = [e for e in top_subgraphs.etypes if 'interacts_c' in e]

    loss_list = []
    X1 = em_bead(h_feat)
    h_center = en_net(top_subgraphs, X1, top_list, ['w'], ['bead'])

    xp, std = de_net(top_graph, h_center)
    xt = top_graph.edges['interacts'].data['label']

    if xp.shape[0]==0 or xp.shape[0]!= xt.shape[0]:
        return

    l_nll = loss_fc[0](xp, xt, cw)
    l_wnl = loss_fc[1](xp, xt, ncluster)
    loss = l_nll + l_wnl

    optimizer[0].zero_grad()
    loss.backward(retain_graph=True)  # retain_graph=False,
    optimizer[0].step()

    loss_list.append([l_nll.item(), l_wnl.item()])

    ll = np.array(loss_list)
    ll = np.nanmean(ll, axis=0, keepdims=False)
    return ll


def inference(graphs, features, num_heads, num_clusters, em_networks, ae_networks, device):
    top_graph = graphs['top_graph'].to(device)
    top_subgraphs = graphs['top_subgraphs'].to(device)

    h_feat = features

    em_bead = em_networks[0]
    en_net = ae_networks[0]
    de_net = ae_networks[1]

    top_list = [e for e in top_subgraphs.etypes if 'interacts_c' in e]

    loss_list = []

    with torch.no_grad():

        X1 = em_bead(h_feat)
        h_center = en_net( top_subgraphs, X1, top_list, ['w'], ['bead'])

        xp1, _ = de_net(top_graph, h_center)

        p1 = xp1.cpu().detach().numpy()
        tp1 = graphs['top_graph'].edges['interacts'].data['label'].cpu().detach().numpy()

        pred_X = h_center.cpu().detach().numpy()
        pred_cluster_mat = np.ones((pred_X.shape[0], pred_X.shape[0]))*(num_clusters[1]-1)
        xs,ys = graphs['top_graph'].edges(etype='interacts', form='uv')[0], graphs['top_graph'].edges(etype='interacts', form='uv')[1]
        pred_cluster_mat[xs, ys] = np.argmax(p1, axis=1)

        true_cluster_mat = np.ones((pred_X.shape[0], pred_X.shape[0]))*(num_clusters[1]-1)
        true_cluster_mat[xs, ys] = tp1

        return pred_X, pred_cluster_mat, true_cluster_mat


def run_epoch(dataset, model, loss_fc, optimizer, batch_size, iterations, device, writer=None, config=None):
    em_networks, ae_networks = model
    loss_list = []
    dur = []

    for epoch in np.arange(iterations):
        print("epoch {:d} chromosome: ".format(epoch), end=' ')
        if epoch >=3:
            t0 = time.time()
        for j, data in enumerate(dataset):
            graphs, features, chro, cluster_weights = data
            print(chro, end=' ')

            # 1 over density of cluster
            cw = torch.tensor(cluster_weights['cw']).to(device)
 
            h_f = features['feat']
            h_p = features['pos']

            h_f_vn = torch.nn.functional.normalize(torch.tensor(h_f, dtype=torch.float), p=2.0, dim=0)
            h_f_hn = torch.nn.functional.normalize(torch.tensor(h_f, dtype=torch.float), p=2.0, dim=1)

            h_feat = torch.stack( [h_f_vn, h_f_hn,
                                    torch.tensor(h_p, dtype=torch.float)], 
                                    dim=1).to(device)

            ll = fit_one_step(graphs, h_feat, cw, batch_size, em_networks, ae_networks, loss_fc, optimizer, device)
            loss_list.append(ll)

            if epoch == 0 and j == 0 and writer is not None:
                m = cluster_weights['mats']
                plot_feature(h_f_vn, h_f_hn, h_p, writer, '0, features/h')
                plot_cluster(m, writer, int(config['parameter']['graph']['num_clusters']),'0 cluster/bead', step=None)

            if epoch%5==0 and j == 0 and writer is not None and config is not None:
                num_heads = int(config['parameter']['G3DM']['num_heads']['out'])
                [center_X,
                center_cluster_mat, 
                center_true] = inference(graphs, h_feat, num_heads, 
                                        int(config['parameter']['graph']['num_clusters']), 
                                        em_networks, ae_networks, device)
                plot_X(center_X, writer, '1, 3D/center', step=epoch)
                plot_cluster(center_cluster_mat, writer, 
                            int(config['parameter']['graph']['num_clusters']),
                            '2,1 cluster/center', step=epoch)
                plot_confusion_mat(center_cluster_mat, center_true,  writer, '2,2 confusion matrix/center', step=epoch)

                for name, param in ae_networks[1].named_parameters():
                    if name == 'r_dist':
                        x1 = param.to('cpu').detach().numpy()

                plot_lines(np.cumsum(np.abs(x1+1e-4))+0.1, writer, '2,3 hop_dist/center', step=epoch)

        ll = np.array(loss_list)
        plot_scaler(np.nanmean(ll[:,0]), writer, 'Loss/l_nll' ,step = epoch)
        plot_scaler(np.nanmean(ll[:,1]), writer, 'Loss/l_wnl' ,step = epoch)
        if epoch >=3:
            dur.append(time.time() - t0)
        print("Loss:", np.nanmean(ll, axis=0), "| Time(s) {:.4f} ".format( np.mean(dur)), sep =" " )
        if epoch%10==0:
            GPUtil.showUtilization()

