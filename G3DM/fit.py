import os
import sys, time, GPUtil
import dgl
import torch
import torch_optimizer as optim
import numpy as np

from .model import embedding, encoder_chain, decoder_distance, decoder_gmm, decoder_dotproduct_euclidian, save_model_state_dict
from .loss import nllLoss, WassersteinLoss, ClusterLoss, RMSLELoss # stdLoss, 
from .visualize import plot_feature, plot_X, plot_cluster, plot_confusion_mat, plot_distributions, plot_histogram2d
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

    em_bead = embedding(in_dim=ind, out_dim=outd, in_num_channels=2).to(device).half()

    nh = int(config['G3DM']['num_heads'])

    chain = config['G3DM']['graph_dim']
    cin, chidden, cout = int(chain['in_dim']), int( chain['hidden_dim']), int(chain['out_dim'])
    e_list = ['interacts_c{}'.format(i) for i in np.arange( int(config['graph']['cutoff_cluster']))]
    en_net = encoder_chain( cin, chidden, cout, num_heads=nh, etypes=e_list).to(device).half()

    nc = int(config['graph']['num_clusters']) - 1
    de_distance_net = decoder_distance(nh, nc, 'bead', 'interacts').to(device).half()
    de_gmm_net = decoder_gmm(nc).to(device).half()
    de_doteuc_net = decoder_dotproduct_euclidian().to(device).half()

    nll = nllLoss().to(device).half()
    # stdl = stdLoss().to(device)
    cwnl = WassersteinLoss(nc).to(device).half()
    cl = ClusterLoss(nc).to(device).half()
    rmslel = RMSLELoss().to(device).half()

    parameters_list = list(em_bead.parameters()) + \
                list(en_net.parameters()) + \
                list(de_distance_net.parameters()) + \
                list(de_gmm_net.parameters())

    opt = optim.AdaBound( parameters_list, 
                        lr=1e-3, betas=(0.9, 0.999), 
                        final_lr=0.1, gamma=1e-3, 
                        eps=1e-8, weight_decay=0,
                        amsbound=False)

    # - opt = optim.AdamP(parameters_list,
    #                     lr= 1e-3, betas=(0.9, 0.999),
    #                     eps=1e-8, weight_decay=0,
    #                     delta = 0.1, wd_ratio = 0.1 )

    # -  opt = optim.RAdam( parameters_list,
    #                     lr= 1e-3, betas=(0.9, 0.999),
    #                     eps=1e-8, weight_decay=0)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)

    em_networks = [em_bead]
    ae_networks = [en_net, de_distance_net, de_gmm_net, de_doteuc_net]
    return em_networks, ae_networks, [nll, cwnl, cl, rmslel], [opt], scheduler


def setup_train(configuration):
    itn = int(configuration['parameter']['G3DM']['iteration'])
    # batch_size = int(configuration['parameter']['G3DM']['batchsize'])
    return itn


def fit_one_step(require_grad, graphs, features, cluster_ranges, em_networks, ae_networks, loss_fc, optimizer, device):
    top_graph = graphs['top_graph'].to(device)
    top_subgraphs = graphs['top_subgraphs'].to(device)

    ncluster = len(cluster_ranges)

    h_feat = features

    em_bead = em_networks[0]
    en_net = ae_networks[0]
    de_dis_net = ae_networks[1]
    de_gmm_net = ae_networks[2]
    de_DotEuc_net = ae_networks[3]

    top_list = [e for e in top_subgraphs.etypes if 'interacts_c' in e]

    X = em_bead(h_feat)
    h_center, h_highdim = en_net(top_subgraphs, X, cluster_ranges, top_list, ['bead'])

    l_sim = torch.empty(1) # len(top_list))
    l_diff = torch.empty(1) # len(top_list))
    for i, et in enumerate(top_list):
        pred_dot, pred_hd_dist = de_DotEuc_net(top_subgraphs, h_highdim, et)
        # true_l = top_subgraphs.edges[et].data['label']
        true_v = top_subgraphs.edges[et].data['value']
        l_sim[i] = loss_fc[3](pred_dot, true_v)
        l_diff[i] = loss_fc[3](pred_hd_dist, cluster_ranges[i])
        break


    xp, std = de_dis_net(top_graph, h_center)
    lt = top_graph.edges['interacts'].data['label']
    if xp.shape[0]==0 or xp.shape[0]!= lt.shape[0] : return [None]

    [dis_cmpt_lp], [dis_gmm] = de_gmm_net(xp) 

    tmp = torch.div( torch.ones(ncluster), ncluster) # torch.softmax( 1.0+torch.div(1, cw), dim=0) #
    ids, n = list(), (lt.shape[0])*0.8*tmp
    for i in torch.arange(ncluster):
        idx = ((lt == i).nonzero(as_tuple=True)[0]).view(-1,)
        if idx.nelement()==0: continue      
        p = torch.ones_like(idx)/idx.shape[0]
        # ids.append(idx[p.multinomial(num_samples=int( torch.minimum(n[i], torch.tensor(idx.shape[0])) ), replacement=False)])
        ids.append(idx[ p.multinomial( num_samples=int( n[i]), replacement=True)])
    mask = torch.cat(ids, dim=0)
    mask, _ = torch.sort(mask)
    # mask = torch.unique(mask, sorted=True, return_inverse=False, return_counts=False)

    sample_dis_cmpt_lp = dis_cmpt_lp[mask, :]
    sample_lt = lt[mask]
    sample_std = std[mask]

    weight = torch.linspace(np.pi*0.1, np.pi*0.75, steps=ncluster, device=device)
    weight = torch.sin(weight) + 1.0
    # weight = torch.ones((ncluster), dtype=torch.float, device=device)  

    l_nll = loss_fc[0](dis_cmpt_lp, lt, weight)
    sample_l_nll = loss_fc[0](sample_dis_cmpt_lp, sample_lt, weight)
    one_hot_lt = torch.nn.functional.one_hot(sample_lt.long(), ncluster)
    l_wnl = loss_fc[1](sample_dis_cmpt_lp, one_hot_lt, weight)
    l_cl = loss_fc[2](sample_dis_cmpt_lp, one_hot_lt, weight)

    if require_grad:
        loss = sample_l_nll*10 + l_wnl + l_cl + l_sim.sum() + l_diff.sum() #+ l_stdl #  + l_stdl
        optimizer[0].zero_grad()
        loss.backward()  # retain_graph=False, create_graph = True
        optimizer[0].step()

    return [l_nll.item(), l_cl.item(), l_wnl.item(), l_sim.sum().item(), l_diff.sum().item()], dis_gmm


def inference(graphs, features, lr_ranges, num_heads, num_clusters, em_networks, ae_networks, device):
    top_graph = graphs['top_graph'].to(device)
    top_subgraphs = graphs['top_subgraphs'].to(device)

    h_feat = features

    em_bead = em_networks[0]
    en_net = ae_networks[0]
    de_dis_net = ae_networks[1]
    de_gmm_net = ae_networks[2]

    top_list = [e for e in top_subgraphs.etypes if 'interacts_c' in e]

    loss_list = []

    with torch.no_grad():

        X = em_bead(h_feat)
        h_center, h_highdim = en_net( top_subgraphs, X, lr_ranges, top_list, ['bead'])

        xp1, _ = de_dis_net(top_graph, h_center)

        [dis_cmpt_lp], [dis_gmm] = de_gmm_net(xp1)

        dp1 = torch.exp(dis_cmpt_lp).cpu().detach().numpy()
        tp1 = top_graph.edges['interacts'].data['label'].cpu().detach().numpy()

        pred_X = h_center.cpu().detach().numpy()
        xs,ys = graphs['top_graph'].edges(etype='interacts', form='uv')[0], graphs['top_graph'].edges(etype='interacts', form='uv')[1]

        pred_distance_cluster_mat = np.ones((pred_X.shape[0], pred_X.shape[0]))*(num_clusters-1)
        pred_distance_cluster_mat[xs, ys] = np.argmax(dp1, axis=1)

        true_cluster_mat = np.ones((pred_X.shape[0], pred_X.shape[0]))*(num_clusters-1)
        true_cluster_mat[xs, ys] = tp1

        distance_mat = np.zeros((pred_X.shape[0], pred_X.shape[0]))
        distance_mat[xs, ys] = xp1.view(-1,).cpu().detach().numpy()

        return pred_X, pred_distance_cluster_mat, true_cluster_mat, [dis_gmm], distance_mat


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
        'decoder_distance_model': ae_networks[1],
        'decoder_gmm_model': ae_networks[2]
    }

    test_loss_list, valid_loss_list, dur = [], [], []
    best_nll_loss = 10
    lr_ranges = torch.linspace(1.0, 23.0, 
                            steps=int(config['parameter']['graph']['num_clusters'])-1, 
                            requires_grad=False).to(device)
    for epoch in np.arange(iterations):
        print("epoch {:d} ".format(epoch), end=' ')
        t0 = time.time()
        for j, data in enumerate(train_dataset):
            graphs, features, _, cluster_weights, _ = data
            h_f, h_p = features['feat'], features['pos']
            h_f_n = torch.nn.functional.normalize(torch.tensor(h_f, dtype=torch.half), p=1.0, dim=1)*h_f.shape[1]
            h_p_n =torch.nn.functional.normalize(torch.tensor(h_p, dtype=torch.half), p=1.0, dim=1)*h_p.shape[1]
            h_feat = torch.stack([h_f_n, h_p_n], dim=1).to(device)

            ll, dis_gmm = fit_one_step(True, graphs, h_feat, lr_ranges, em_networks, ae_networks, loss_fc, optimizer, device)
            mu = dis_gmm.component_distribution.mean.detach()
            stddev = dis_gmm.component_distribution.stddev.detach()
            lr_ranges = torch.exp(mu - stddev**2)
            # lr_ranges = dis_gmm.component_distribution.mean.detach()
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
                        ae_networks[1].load_state_dict(checkpoint['decoder_distance_model_state_dict'])
                        ae_networks[2].load_state_dict(checkpoint['decoder_gmm_model_state_dict'])
                        optimizer[0].load_state_dict(checkpoint['optimizer_state_dict'])
                        rollback_epoch = checkpoint['epoch']
                        rollback_nll = checkpoint['nll_loss']
                        print('Detected NaN in the parameters of {}, rollback to epoch #{}, nll loss: {}'.format(key, rollback_epoch, rollback_nll))

            if ll[0] < best_nll_loss:
                os.makedirs(model_saved_path, exist_ok=True)
                path = os.path.join(model_saved_path, 'ckpt_state_dict_' + model_saved_name)
                best_nll_loss = ll[0]
                save_model_state_dict(models_dict, optimizer[0], path, epoch, best_nll_loss)

            torch.cuda.empty_cache()

        for j, data in enumerate(valid_dataset):
            graphs, features, _, cluster_weights, _ = data

            # 1 over density of cluster
            # cw = torch.tensor(cluster_weights['cw']).to(device)
 
            h_f, h_p = features['feat'], features['pos']
            h_f_n = torch.nn.functional.normalize(torch.tensor(h_f, dtype=torch.half), p=1.0, dim=1)*h_f.shape[1]
            h_p_n =torch.nn.functional.normalize(torch.tensor(h_p, dtype=torch.half), p=1.0, dim=1)*h_p.shape[1]
            h_feat = torch.stack( [h_f_n, h_p_n], dim=1).to(device)

            ll, _ = fit_one_step(False, graphs, h_feat, lr_ranges, em_networks, ae_networks, loss_fc, optimizer, device)
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
                [X, 
                pred_distance_mat, 
                center_true_mat, [dis_gmm], distance_mat ] = inference(graphs, h_feat, lr_ranges, num_heads, 
                                            int(config['parameter']['graph']['num_clusters']), 
                                            em_networks, ae_networks, device)

                plot_X(X, writer, '1, 3D/center', step=epoch)
                plot_cluster(pred_distance_mat, writer, 
                            int(config['parameter']['graph']['num_clusters']),
                            '2,1 cluster/prediction distance', step=epoch)

                plot_cluster(center_true_mat, writer, 
                            int(config['parameter']['graph']['num_clusters']),
                            '2,1 cluster/true', step=epoch) if epoch==0 else None

                plot_confusion_mat(pred_distance_mat, center_true_mat,  writer, '2,2 confusion matrix/normalize: ture, predicted distance - true contact', step=epoch, normalize='true')
                plot_confusion_mat(pred_distance_mat, center_true_mat,  writer, '2,2 confusion matrix/normalize: all, predicted distance - true contact', step=epoch, normalize='all')
                
                mu = (dis_gmm.component_distribution.mean)
                std = (dis_gmm.component_distribution.stddev)
                # std = (dis_gmm.component_distribution.variance)
                x = torch.linspace(start=0.1, end=7.0, steps=150, device=device) # mu.max()*1.5,
                log_pdfs = dis_gmm.component_distribution.log_prob(x.view(-1,1))
                # log_pdfs = log_pdfs + torch.log(dis_gmm.mixture_distribution.probs).view(1, -1)
                normal_pdfs = torch.nn.functional.normalize( torch.exp(log_pdfs)*(dis_gmm.mixture_distribution.probs).view(1, -1), p=1, dim=1)
                normal_pdfs = normal_pdfs.to('cpu').detach().numpy()
                weights = (dis_gmm.mixture_distribution.probs).to('cpu').detach().numpy()
                plot_distributions([ mu.to('cpu').detach().numpy(), 
                                    x.to('cpu').detach().numpy(), 
                                    normal_pdfs,
                                    weights], 
                                    writer, '2,3 hop_dist/Normal ln(x)~N(,)', step=epoch) 

                lognormal_pdfs = torch.empty(normal_pdfs.shape)
                lognormal_mu = torch.empty(mu.shape)
                lognormal_mode = torch.empty(mu.shape)
                x = torch.linspace(start=0.5, end=60.0, steps=150, device=device) # mu.max()*(1+1e-4)
                for i in np.arange(len(mu)):
                    A = torch.div( torch.ones(1, device=device), x*std[i]*torch.sqrt(2.0*torch.tensor(np.pi, device=device)))
                    B = (torch.log(x)-mu[i])**2
                    C = 2*std[i]**2
                    lognormal_pdfs[:,i] = (A * torch.exp(-1.0*torch.div(B, C)))*weights[i]
                    lognormal_mu[i] = torch.exp(mu[i])*torch.sqrt( torch.exp(std[i]**2.0))
                    lognormal_mode[i] = torch.exp(mu[i] - std[i]**2)
                lognormal_pdfs = torch.nn.functional.normalize( lognormal_pdfs, p=1, dim=1)
                plot_distributions( [lognormal_mode.to('cpu').detach().numpy(), 
                                    x.to('cpu').detach().numpy(), 
                                    lognormal_pdfs.to('cpu').detach().numpy(),
                                    weights], 
                                    writer, '2,3 hop_dist/LogNormal x~LogNormal(,)', step=epoch) 
                
                inputs = [distance_mat, pred_distance_mat]
                plot_histogram2d(inputs, writer, '2,4 historgram/distance, predict', step=epoch)
                inputs = [distance_mat, center_true_mat]
                plot_histogram2d(inputs, writer, '2,4 historgram/distance, true', step=epoch)

            torch.cuda.empty_cache()
        scheduler.step()
        test_ll = np.array(test_loss_list)
        valid_ll = np.array(valid_loss_list)

        plot_scaler({'test':np.nanmean(test_ll[:,0]), 'validation': np.nanmean(valid_ll[:,0])}, writer, 'NL Loss' ,step = epoch)
        plot_scaler({'test':np.nanmean(test_ll[:,1]), 'validation': np.nanmean(valid_ll[:,1])}, writer, 'Cluster Loss' ,step = epoch)
        plot_scaler({'test':np.nanmean(test_ll[:,2]), 'validation': np.nanmean(valid_ll[:,2])}, writer, 'WNL Loss' ,step = epoch)
        plot_scaler({'test':np.nanmean(test_ll[:,3]), 'validation': np.nanmean(valid_ll[:,2])}, writer, 'High dim similarity Loss' ,step = epoch)
        plot_scaler({'test':np.nanmean(test_ll[:,4]), 'validation': np.nanmean(valid_ll[:,2])}, writer, 'High dim distance Loss' ,step = epoch)


        dur.append(time.time() - t0)
        print("Loss:", np.nanmean(test_ll, axis=0), "| Time(s) {:.4f} ".format( np.mean(dur)), sep =" " )

    os.makedirs(model_saved_path, exist_ok=True)
    path = os.path.join(model_saved_path, 'finial_' + model_saved_name)
    save_model_state_dict(models_dict, optimizer[0], path)
