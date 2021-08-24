import os
import sys, time, GPUtil
import dgl
import torch
import torch_optimizer as optim
import numpy as np

from .model import embedding, encoder_chain, decoder, save_model_state_dict
from .loss import nllLoss, stdLoss, ClusterWassersteinLoss
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

    em_bead = embedding(in_dim=ind, out_dim=outd, in_num_channels=2).to(device)

    nh = int(config['G3DM']['num_heads'])

    chain = config['G3DM']['graph_dim']
    cin, chidden, cout = int(chain['in_dim']), int( chain['hidden_dim']), int(chain['out_dim'])
    e_list = ['interacts_c{}'.format(i) for i in np.arange( int(config['graph']['cutoff_cluster']))]
    en_net = encoder_chain( cin, chidden, cout, num_heads=nh, etypes=e_list).to(device)

    nc = int(config['graph']['num_clusters']) - 1
    de_net = decoder(nh, nc, 'bead', 'interacts').to(device)

    nll = nllLoss().to(device)
    stdl = stdLoss().to(device)
    cwnl = ClusterWassersteinLoss(device).to(device)

    opt0 = optim.AdaBound(list(em_bead.parameters()) + list(en_net.parameters()) + list(de_net.parameters()),
                        lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3, eps=1e-8, weight_decay=0,
                        amsbound=False)
    opt1 = optim.AdaBound(list(de_net.parameters()),
                    lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3, eps=1e-8, weight_decay=0,
                    amsbound=False)
    # opt = torch.optim.AdamW(list(em_bead.parameters()) + list(en_net.parameters()) + list(de_net.parameters()), 
    #                         lr=0.001, betas=(0.9, 0.999), eps=1e-08, 
    #                         weight_decay=0.01, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt0, gamma=0.9)

    em_networks = [em_bead]
    ae_networks = [en_net, de_net]
    return em_networks, ae_networks, [nll, cwnl, stdl], [opt0, opt1], scheduler


def setup_train(configuration):
    itn = int(configuration['parameter']['G3DM']['iteration'])
    # batch_size = int(configuration['parameter']['G3DM']['batchsize'])
    return itn


def fit_one_step(require_grad, graphs, features, cluster_weights, em_networks, ae_networks, loss_fc, optimizer, device):
    top_graph = graphs['top_graph'].to(device)
    top_subgraphs = graphs['top_subgraphs'].to(device)

    cw = cluster_weights
    ncluster = len(cluster_weights)

    h_feat = features

    em_bead = em_networks[0]
    en_net = ae_networks[0]
    de_net = ae_networks[1]

    top_list = [e for e in top_subgraphs.etypes if 'interacts_c' in e]

    X1 = em_bead(h_feat)
    h_center = en_net(top_subgraphs, X1, top_list, ['w'], ['bead'])
    xt = top_graph.edges['interacts'].data['label']

    for k in torch.arange(4):
        xp, std = de_net(top_graph, h_center)
        if xp.shape[0]==0 or xp.shape[0]!= xt.shape[0]:
            break
        l_nll = loss_fc[0](xp, xt, cw) 
        l_wnl = loss_fc[1](xp, xt, ncluster)
        if l_nll > 1e4 or l_wnl > 1e4:
            continue
        if require_grad:
            loss = l_nll + 10*l_wnl # + 100*l_wnl + l_stdl 
            optimizer[1].zero_grad()
            loss.backward(retain_graph=True)  # retain_graph=False,
            optimizer[1].step()

    xp, std = de_net(top_graph, h_center)
    if xp.shape[0]==0 or xp.shape[0]!= xt.shape[0]:
        return [None, None, None, None]

    l_nll = loss_fc[0](xp, xt, cw) 
    l_nll_noweight = loss_fc[0](xp, xt, None)
    l_wnl = loss_fc[1](xp, xt, ncluster)
    l_stdl = loss_fc[2](std, xt, ncluster)

    if l_nll > 1e4 or l_wnl > 1e4 or l_stdl > 1e4:
        print(xp.min(), xp.max(), xp.mean())
        print(X1.mean(dim=0))
        return [None, None, None, None]

    if require_grad:
        loss = l_nll + l_nll_noweight + 10*l_stdl + 10*l_wnl # + 100*l_wnl + l_stdl 
        optimizer[0].zero_grad()
        loss.backward()  # retain_graph=False,
        optimizer[0].step()

    return [l_nll.item(), l_wnl.item(), l_stdl.item(), l_nll_noweight.item()]


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
        tp1 = top_graph.edges['interacts'].data['label'].cpu().detach().numpy()

        # tp1 = graphs['top_graph'].edges['interacts'].data['label'].cpu().detach().numpy()

        pred_X = h_center.cpu().detach().numpy()
        xs,ys = graphs['top_graph'].edges(etype='interacts', form='uv')[0], graphs['top_graph'].edges(etype='interacts', form='uv')[1]

        pred_cluster_mat = np.ones((pred_X.shape[0], pred_X.shape[0]))*(num_clusters-1)
        pred_cluster_mat[xs, ys] = np.argmax(p1, axis=1)

        true_cluster_mat = np.ones((pred_X.shape[0], pred_X.shape[0]))*(num_clusters-1)
        true_cluster_mat[xs, ys] = tp1
        return pred_X, pred_cluster_mat, true_cluster_mat


def run_epoch(datasets, model, loss_fc, optimizer, scheduler, iterations, device, writer=None, config=None, saved_model=None):
    train_dataset = datasets[0]
    valid_dataset = datasets[1] if len(datasets) > 1 else None
    test_dataset = datasets[2] if len(datasets) > 2 else None

    model_saved_path = saved_model[0] if saved_model is not None else None
    model_saved_name = saved_model[1] if saved_model is not None else None

    em_networks, ae_networks = model
    models_dict = {
        'embedding_model': em_networks[0],
        'encoder_model': ae_networks[0],
        'decoder_model': ae_networks[1]
    }

    test_loss_list, valid_loss_list, dur = [], [], []
    best_nll_loss = 10
    for epoch in np.arange(iterations):
        print("epoch {:d} ".format(epoch), end=' ')
        t0 = time.time()
        for j, data in enumerate(train_dataset):
            graphs, features, _, cluster_weights, _ = data

            # 1 over density of cluster
            cw = torch.tensor(cluster_weights['cw']).to(device)

            h_f, h_p = features['feat'], features['pos']
            h_f_n = torch.nn.functional.normalize(torch.tensor(h_f, dtype=torch.float), p=1.0, dim=1)*h_f.shape[1]
            h_p_n =torch.nn.functional.normalize(torch.tensor(h_p, dtype=torch.float), p=1.0, dim=1)*h_p.shape[1]
            h_feat = torch.stack([h_f_n, h_p_n], dim=1).to(device)

            ll = fit_one_step( True, graphs, h_feat, cw, em_networks, ae_networks, loss_fc, optimizer, device)
            if None in ll:
                continue
            
            test_loss_list.append(ll)

            for key, m in models_dict.items():
                for param in list(m.parameters()):
                    if param.isnan().any() and model_saved_path is not None:
                        path = os.path.join(model_saved_path, 'ckpt_state_dict_'+ model_saved_name)
                        checkpoint = torch.load(path, map_location=device)
                        em_networks[0].load_state_dict(checkpoint['embedding_model_state_dict'])
                        ae_networks[0].load_state_dict(checkpoint['encoder_model_state_dict'])
                        ae_networks[1].load_state_dict(checkpoint['decoder_model_state_dict'])
                        optimizer[0].load_state_dict(checkpoint['optimizer_state_dict'])
                        rollback_epoch = checkpoint['epoch']
                        rollback_nll = checkpoint['nll_loss']
                        print('Detected NaN in the parameters, rollback to epoch #{}, nll loss: {}'.format(rollback_epoch, rollback_nll))

            if ll[0] < best_nll_loss:
                os.makedirs(model_saved_path, exist_ok=True)
                path = os.path.join(model_saved_path, 'ckpt_state_dict_' + model_saved_name)
                best_nll_loss = ll[0]
                save_model_state_dict(models_dict, optimizer[0], path, epoch, best_nll_loss)

            torch.cuda.empty_cache()

        for j, data in enumerate(valid_dataset):
            graphs, features, _, cluster_weights, _ = data

            # 1 over density of cluster
            cw = torch.tensor(cluster_weights['cw']).to(device)
 
            h_f, h_p = features['feat'], features['pos']
            h_f_n = torch.nn.functional.normalize(torch.tensor(h_f, dtype=torch.float), p=1.0, dim=1)*h_f.shape[1]
            h_p_n =torch.nn.functional.normalize(torch.tensor(h_p, dtype=torch.float), p=1.0, dim=1)*h_p.shape[1]
            h_feat = torch.stack( [h_f_n, h_p_n], dim=1).to(device)

            ll = fit_one_step(False, graphs, h_feat, cw, em_networks, ae_networks, loss_fc, optimizer, device)
            if None in ll:
                continue
            valid_loss_list.append(ll)

            if epoch == 0 and j == 0 and writer is not None:
                # pass
                m = cluster_weights['mat']
                plot_feature([h_f_n, h_p_n], writer, '0, features/h')
                plot_cluster(m, writer, int(config['parameter']['graph']['num_clusters']),'0, cluster/bead', step=None)

            if epoch%3==0 and j == 0 and writer is not None and config is not None:
                num_heads = int(config['parameter']['G3DM']['num_heads'])
                [center_X, 
                center_pred_mat, 
                center_true_mat] = inference(graphs, h_feat, num_heads, 
                                            int(config['parameter']['graph']['num_clusters']), 
                                            em_networks, ae_networks, device)
                plot_X(center_X, writer, '1, 3D/center', step=epoch)
                plot_cluster(center_pred_mat, writer, 
                            int(config['parameter']['graph']['num_clusters']),
                            '2,1 cluster/prediction', step=epoch)
                if epoch==0:
                    plot_cluster(center_true_mat, writer, 
                                int(config['parameter']['graph']['num_clusters']),
                                '2,1 cluster/true', step=epoch)
                plot_confusion_mat(center_pred_mat, center_true_mat,  writer, '2,2 confusion matrix/center', step=epoch)

                # x1 = np.linspace(0.0, 0.01, num=50)
                # for name, param in ae_networks[1].named_parameters():
                #     if name == 'in_dist':
                #         mat = param.to('cpu').detach()
                #         mat = torch.softmax(mat, dim=0).numpy()
                #     if name == 'r':
                #         r = param.to('cpu').detach().numpy()
                # x1 = np.matmul(x1*r, mat)
                # x = np.concatenate([[0], x1, [2.0]])
                # x = np.sort(x)
                for name, param in ae_networks[1].named_parameters():
                    if name == 'in_dist':
                        x1 = param.to('cpu').detach().numpy()
                    if name == 'r':
                        r = param.to('cpu').detach().numpy()
                x = np.clip( np.cumsum(np.abs(x1)*r), a_min=1.0, a_max=None)
                x = np.concatenate([[0], x])
                plot_lines(x, writer, '2,3 hop_dist/center', step=epoch) 

            torch.cuda.empty_cache()
        scheduler.step()
        test_ll = np.array(test_loss_list)
        valid_ll = np.array(valid_loss_list)

        plot_scaler({'test':np.nanmean(test_ll[:,0]), 'validation': np.nanmean(valid_ll[:,0])}, writer, 'NL Loss' ,step = epoch)
        plot_scaler({'test':np.nanmean(test_ll[:,1]), 'validation': np.nanmean(valid_ll[:,1])}, writer, 'Wasserstein Loss' ,step = epoch)
        plot_scaler({'test':np.nanmean(test_ll[:,2]), 'validation': np.nanmean(valid_ll[:,2])}, writer, 'STD Loss' ,step = epoch)
        plot_scaler({'test':np.nanmean(test_ll[:,3]), 'validation': np.nanmean(valid_ll[:,3])}, writer, 'NL Loss no weight' ,step = epoch)

        dur.append(time.time() - t0)
        print("Loss:", np.nanmean(test_ll, axis=0), "| Time(s) {:.4f} ".format( np.mean(dur)), sep =" " )
        # if epoch%10==0:
        #     GPUtil.showUtilization()

    os.makedirs(model_saved_path, exist_ok=True)
    path = os.path.join(model_saved_path, 'finial_' + model_saved_name)
    save_model_state_dict(models_dict, optimizer[0], path)
