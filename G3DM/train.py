import os
import sys, time, GPUtil
import dgl
import torch
import torch_optimizer as optim
import numpy as np

from .model import embedding, encoder_chain, decoder_distance, decoder_gmm, save_model_state_dict
from .loss import nllLoss, stdLoss, ClusterWassersteinLoss
from .visualize import plot_feature, plot_X, plot_cluster, plot_confusion_mat, plot_distributions
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
    de_distance_net = decoder_distance(nh, nc, 'bead', 'interacts').to(device)
    de_gmm_net = decoder_gmm(nc).to(device)

    nll = nllLoss().to(device)
    stdl = stdLoss().to(device)
    cwnl = ClusterWassersteinLoss(nc).to(device)
    # rmslel = RMSLELoss().to(device)

    parameters_list = list(em_bead.parameters()) + \
                list(en_net.parameters()) + \
                list(de_distance_net.parameters()) + \
                list(de_gmm_net.parameters())

    opt = optim.AdaBound( parameters_list, 
                        lr=1e-3, betas=(0.9, 0.999), 
                        final_lr=0.1, gamma=1e-3, 
                        eps=1e-8, weight_decay=0,
                        amsbound=False)


    # opt = optim.RAdam( parameters_list,
    #                     lr= 1e-2, betas=(0.9, 0.999),
    #                     eps=1e-8, weight_decay=0)

    # opt = optim.QHAdam( parameters_list,
    #                     lr= 1e-3, betas=(0.9, 0.999),
    #                     nus=(1.0, 1.0), weight_decay=0,
    #                     decouple_weight_decay=False,
    #                     eps=1e-8)

    # opt = torch.optim.RMSprop(list(em_bead.parameters()) + list(en_net.parameters()) 
    #                         + list(de_distance_net.parameters()) + list(de_gmm_net.parameters()))

    # opt = optim.Yogi(parameters_list,
    #                 lr= 1e-3,
    #                 betas=(0.9, 0.999),
    #                 eps=1e-3,
    #                 initial_accumulator=1e-6,
    #                 weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)

    em_networks = [em_bead]
    ae_networks = [en_net, de_distance_net, de_gmm_net]
    return em_networks, ae_networks, [nll, stdl, cwnl], [opt], scheduler


def setup_train(configuration):
    itn = int(configuration['parameter']['G3DM']['iteration'])
    # batch_size = int(configuration['parameter']['G3DM']['batchsize'])
    return itn


def fit_one_step(epoch, require_grad, graphs, features, cluster_weights, em_networks, ae_networks, loss_fc, optimizer, device):
    top_graph = graphs['top_graph'].to(device)
    top_subgraphs = graphs['top_subgraphs'].to(device)

    cw = cluster_weights
    ncluster = len(cluster_weights)

    h_feat = features

    em_bead = em_networks[0]
    en_net = ae_networks[0]
    de_dis_net = ae_networks[1]
    de_gmm_net = ae_networks[2]

    top_list = [e for e in top_subgraphs.etypes if 'interacts_c' in e]

    X1 = em_bead(h_feat)
    h_center = en_net(top_subgraphs, X1, top_list, ['w'], ['bead'])
    xp, std = de_dis_net(top_graph, h_center)
    # xt = top_graph.edges['interacts'].data['value']
    lt = top_graph.edges['interacts'].data['label']
    if xp.shape[0]==0 or xp.shape[0]!= lt.shape[0]:
        return [None]

    # idx =  (xp.flatten()).multinomial(num_samples=len(xp), replacement=False)
    # # xt = xt[idx]
    # lt = lt[idx]
    # xp = xp[idx]
    # [dis_cdf, cnt_cdf], [dis_cmpt_cdf, cnt_cmpt_cdf], [dis_cmpt_lp, cnt_cmpt_lp], [cnt_gmm, dis_gmm] = de_gmm_net(xp, xt)
    [dis_cmpt_lp], [dis_gmm, cmpt_w] = de_gmm_net(xp, torch.div(1.0, cw)**(1)) 
    # l_nll = loss_fc[0](xp, xt, cw) 
    # l_nll_noweight = loss_fc[0](xp, xt, None)
    # l_wnl = loss_fc[1](xp, xt, ncluster)
    # l_stdl = loss_fc[2](std, xt, ncluster)
    # if l_nll > 1e4 or l_wnl > 1e4 or l_stdl > 1e4:
    #     print(xp.min(), xp.max(), xp.mean())
    #     print(X1.mean(dim=0))
    #     return [None, None, None, None]

    # rmseloss_all = loss_fc[0](dis_cdf, cnt_cdf)
    # rmseloss_cmpt = loss_fc[0](dis_cmpt_cdf, cnt_cmpt_cdf)
    one_hot_lt = torch.nn.functional.one_hot(lt.long(), ncluster)
    weight_r = torch.linspace( ncluster, 1, steps=ncluster, dtype=torch.float, device=device)
    # weight_l = torch.linspace(1, ncluster, steps=ncluster, dtype=torch.float, device=device)
    # weight_r = cw**(0.5)+10.0
    # weight_r = torch.ones_like(cw)
    
    l_nll = loss_fc[0](dis_cmpt_lp, lt, cw, weight_r)
    l_wnl = loss_fc[2](dis_cmpt_lp, one_hot_lt, cw, weight_r)
    # if (epoch%10) < 7 or epoch <=20:
    #     l_nll = loss_fc[0](dis_cmpt_lp, lt, cw, weight_r**(0.5))
    #     l_wnl = loss_fc[2](dis_cmpt_lp, one_hot_lt, cw, weight_r**(0.5))
    # else:
    #     l_nll = loss_fc[0](dis_cmpt_lp, lt, cw, weight_l)
    #     l_wnl = loss_fc[2](dis_cmpt_lp, one_hot_lt, cw, weight_l)

    l_stdl = loss_fc[1](std, lt, ncluster)

    if require_grad:
        loss = l_nll**2 + l_wnl*10 # + l_wnl + l_stdl
        optimizer[0].zero_grad()
        loss.backward()  # retain_graph=False,
        optimizer[0].step()

    return [l_nll.item(), l_stdl.item(), l_wnl.item()]


def inference(graphs, features, cluster_weights, num_heads, num_clusters, em_networks, ae_networks, device):
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

        X1 = em_bead(h_feat)
        h_center = en_net( top_subgraphs, X1, top_list, ['w'], ['bead'])

        xp1, _ = de_dis_net(top_graph, h_center)

        # xt1 = top_graph.edges['interacts'].data['value']
        # [dis_cdf, cnt_cdf], [dis_cmpt_cdf, cnt_cmpt_cdf], [dis_cmpt_lp, cnt_cmpt_lp], [cnt_gmm, dis_gmm] = de_gmm_net(xp1)
        cw = cluster_weights
        [dis_cmpt_lp], [dis_gmm, cmpt_w] = de_gmm_net(xp1, torch.div(1.0, cw)**(1) )

        dp1 = torch.exp(dis_cmpt_lp).cpu().detach().numpy()
        # cp1 = torch.exp(cnt_cmpt_lp).cpu().detach().numpy() # 
        tp1 = top_graph.edges['interacts'].data['label'].cpu().detach().numpy()

        pred_X = h_center.cpu().detach().numpy()
        xs,ys = graphs['top_graph'].edges(etype='interacts', form='uv')[0], graphs['top_graph'].edges(etype='interacts', form='uv')[1]

        pred_distance_cluster_mat = np.ones((pred_X.shape[0], pred_X.shape[0]))*(num_clusters-1)
        pred_distance_cluster_mat[xs, ys] = np.argmax(dp1, axis=1)

        # pred_contact_cluster_mat = np.ones((pred_X.shape[0], pred_X.shape[0]))*(num_clusters-1)
        # pred_contact_cluster_mat[xs, ys] = np.argmax(cp1, axis=1)

        true_cluster_mat = np.ones((pred_X.shape[0], pred_X.shape[0]))*(num_clusters-1)
        true_cluster_mat[xs, ys] = tp1
        # return pred_X, pred_distance_cluster_mat, pred_contact_cluster_mat, true_cluster_mat, [cnt_gmm, dis_gmm]
        return pred_X, pred_distance_cluster_mat, true_cluster_mat, [dis_gmm, cmpt_w]


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

            ll = fit_one_step(epoch, True, graphs, h_feat, cw, em_networks, ae_networks, loss_fc, optimizer, device)
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
            cw = torch.tensor(cluster_weights['cw']).to(device)
 
            h_f, h_p = features['feat'], features['pos']
            h_f_n = torch.nn.functional.normalize(torch.tensor(h_f, dtype=torch.float), p=1.0, dim=1)*h_f.shape[1]
            h_p_n =torch.nn.functional.normalize(torch.tensor(h_p, dtype=torch.float), p=1.0, dim=1)*h_p.shape[1]
            h_feat = torch.stack( [h_f_n, h_p_n], dim=1).to(device)

            ll = fit_one_step(epoch, False, graphs, h_feat, cw, em_networks, ae_networks, loss_fc, optimizer, device)
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
                pred_distance_mat, 
                center_true_mat, [dis_gmm, cmpt_w] ] = inference(graphs, h_feat, cw, num_heads, 
                                            int(config['parameter']['graph']['num_clusters']), 
                                            em_networks, ae_networks, device)

                plot_X(center_X, writer, '1, 3D/center', step=epoch)
                plot_cluster(pred_distance_mat, writer, 
                            int(config['parameter']['graph']['num_clusters']),
                            '2,1 cluster/prediction distance', step=epoch)
                # plot_cluster(pred_contact_mat, writer, 
                #             int(config['parameter']['graph']['num_clusters']),
                #             '2,1 cluster/prediction contact', step=epoch)

                plot_cluster(center_true_mat, writer, 
                            int(config['parameter']['graph']['num_clusters']),
                            '2,1 cluster/true', step=epoch) if epoch==0 else None

                # plot_confusion_mat(pred_distance_mat, pred_contact_mat,  writer, '2,2 confusion matrix/predicted distance - predicted contact', step=epoch)
                plot_confusion_mat(pred_distance_mat, center_true_mat,  writer, '2,2 confusion matrix/normalize: ture, predicted distance - true contact', step=epoch, normalize='true')
                plot_confusion_mat(pred_distance_mat, center_true_mat,  writer, '2,2 confusion matrix/normalize: all, predicted distance - true contact', step=epoch, normalize='all')
                # plot_confusion_mat(pred_contact_mat, center_true_mat,  writer, '2,3 confusion matrix/predicted contact - true contact', step=epoch)

                mu = (dis_gmm.component_distribution.mean)
                std = (dis_gmm.component_distribution.stddev)
                # std = (dis_gmm.component_distribution.variance)
                x = torch.linspace(start=0.1, end=mu.max()*1.5, steps=150, device=device)
                log_pdfs = dis_gmm.component_distribution.log_prob(x.view(-1,1))
                log_pdfs = log_pdfs + cmpt_w.view(1, -1)
                normal_pdfs = torch.exp(log_pdfs).to('cpu').detach().numpy()
                weights = (dis_gmm.mixture_distribution.probs).to('cpu').detach().numpy()
                plot_distributions([ mu.to('cpu').detach().numpy(), 
                                    x.to('cpu').detach().numpy(), 
                                    normal_pdfs,
                                    weights], 
                                    writer, '2,3 hop_dist/Normal ln(x)~N(,)', step=epoch) 

                lognormal_pdfs = torch.empty(normal_pdfs.shape)
                lognormal_mu = torch.empty(mu.shape)
                lognormal_mode = torch.empty(mu.shape)
                x = torch.exp(torch.linspace(start=-2.0, end=mu.max()*(1+1e-4), steps=150, device=device))
                for i in np.arange(len(mu)):
                    A = torch.div( torch.ones(1, device=device), x*std[i]*torch.sqrt(2.0*torch.tensor(np.pi, device=device)))
                    B = (torch.log(x)-mu[i])**2
                    C = 2*std[i]**2
                    lognormal_pdfs[:,i] = (A * torch.exp(-1.0*torch.div(B, C)))*torch.exp(cmpt_w[i])
                    lognormal_mu[i] = torch.exp(mu[i])*torch.sqrt( torch.exp(std[i]**2.0))
                    lognormal_mode[i] = torch.exp(mu[i] - std[i]**2)
                plot_distributions( [lognormal_mode.to('cpu').detach().numpy(), 
                                    x.to('cpu').detach().numpy(), 
                                    lognormal_pdfs.to('cpu').detach().numpy(),
                                    weights], 
                                    writer, '2,3 hop_dist/LogNormal x~LogNormal(,)', step=epoch) 

            torch.cuda.empty_cache()
        scheduler.step()
        test_ll = np.array(test_loss_list)
        valid_ll = np.array(valid_loss_list)

        plot_scaler({'test':np.nanmean(test_ll[:,0]), 'validation': np.nanmean(valid_ll[:,0])}, writer, 'NL Loss' ,step = epoch)
        plot_scaler({'test':np.nanmean(test_ll[:,1]), 'validation': np.nanmean(valid_ll[:,1])}, writer, 'STD Loss' ,step = epoch)
        plot_scaler({'test':np.nanmean(test_ll[:,2]), 'validation': np.nanmean(valid_ll[:,2])}, writer, 'WNL Loss' ,step = epoch)

        dur.append(time.time() - t0)
        print("Loss:", np.nanmean(test_ll, axis=0), "| Time(s) {:.4f} ".format( np.mean(dur)), sep =" " )
        # if epoch%10==0:
        #     GPUtil.showUtilization()

    os.makedirs(model_saved_path, exist_ok=True)
    path = os.path.join(model_saved_path, 'finial_' + model_saved_name)
    save_model_state_dict(models_dict, optimizer[0], path)
