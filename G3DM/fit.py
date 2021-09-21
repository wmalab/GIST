import os
import sys, time, GPUtil
import dgl
import torch
import torch_optimizer as optim
import numpy as np

from .model import embedding, encoder_chain, decoder_distance, decoder_gmm, decoder_euclidean, decoder_similarity
from .model import save_model_state_dict
from .loss import nllLoss, WassersteinLoss, RMSLELoss #MSELoss # stdLoss, ClusterLoss,
from .visualize import plot_feature, plot_X, plot_cluster, plot_confusion_mat, plot_distributions, plot_histogram2d
from .visualize import plot_scaler

# import GPUtil
# gpuIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.05, maxMemory = 0.05, includeNan=False, excludeID=[], excludeUUID=[])
# device =  'cpu' if len(gpuIDs)==0 else 'cuda:{}'.format(gpuIDs[0])


def load_dataset(path, name):
    '''graph_dict[chromosome] = {top_graph, top_subgraphs, bottom_graph, inter_graph}
    feature_dict[chromosome] = {'feat', 'pos'} 'feat': hic features; 'pos': position features
    HiCDataset[i]: graph[i], feature[i], label[i](chromosome)'''
    HiCDataset = torch.load(os.path.join(path, name))
    return HiCDataset

def create_network(configuration, device):
    config = configuration['parameter']
    # top_graph = graph['top_graph']
    # top_subgraphs = graph['top_subgraphs']

    ind = int(config['feature']['in_dim'])
    outd = int(config['feature']['out_dim'])

    em_bead = embedding(in_dim=ind, out_dim=outd, in_num_channels=2).to(device).float()

    nh = int(config['G3DM']['num_heads'])

    chain = config['G3DM']['graph_dim']
    cin, chidden, cout = int(chain['in_dim']), int( chain['hidden_dim']), int(chain['out_dim'])
    e_list = ['interacts_c{}'.format(i) for i in np.arange( int(config['graph']['cutoff_cluster']))]
    en_net = encoder_chain( cin, chidden, cout, num_heads=nh, etypes=e_list).to(device).float()

    nc = int(config['graph']['num_clusters']) - 1
    de_distance_net = decoder_distance(nh, nc, 'bead', 'interacts').to(device).float()
    de_gmm_net = decoder_gmm(nc).to(device).float()
    de_euc_net = decoder_euclidean().to(device).float()
    de_sim_net = decoder_similarity().to(device).float()

    nll = nllLoss().to(device).float()
    wnl = WassersteinLoss(nc).to(device).float()
    msel = RMSLELoss().to(device).float()

    parameters_list = list(em_bead.parameters()) + list(en_net.parameters()) + \
                    list(de_distance_net.parameters()) + list(de_gmm_net.parameters()) + \
                    list(de_euc_net.parameters()) + list(de_sim_net.parameters())

    opt = optim.AdaBound( parameters_list, 
                        lr=1e-3, betas=(0.9, 0.999), 
                        final_lr=0.1, gamma=1e-3, 
                        eps=1e-8, weight_decay=0,
                        amsbound=False)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)

    em_networks = [em_bead]
    ae_networks = [en_net, de_distance_net, de_gmm_net, de_euc_net, de_sim_net]
    return em_networks, ae_networks, nh, nc+1, [nll, wnl, msel], [opt], scheduler

def setup_train(configuration):
    itn = int(configuration['parameter']['G3DM']['iteration'])
    num_clusters = int(configuration['parameter']['graph']['num_clusters'])
    num_heads = int(configuration['parameter']['G3DM']['num_heads'])
    # batch_size = int(configuration['parameter']['G3DM']['batchsize'])
    return itn, num_heads, num_clusters

def fit_one_step(require_grad, graphs, features, cluster_ranges, em_networks, ae_networks, loss_fc, optimizer, device):
    top_graph = graphs['top_graph'].to(device)
    top_subgraphs = graphs['top_subgraphs'].to(device)

    ncluster = len(cluster_ranges)
    h_feat = features

    em_bead = em_networks[0]
    en_net = ae_networks[0]
    de_dis_net = ae_networks[1]
    de_gmm_net = ae_networks[2]
    de_euc_net = ae_networks[3]
    de_sim_net = ae_networks[4]

    top_list = [e for e in top_subgraphs.etypes if 'interacts_c' in e]

    X = em_bead(h_feat)
    H, h_highdim = en_net(top_subgraphs, X, cluster_ranges, top_list, ['bead'])

    pred_hd_dist = de_euc_net(top_subgraphs, h_highdim, top_list[0])
    l_diff_g = loss_fc[2](pred_hd_dist, cluster_ranges[0])

    pred_similarity = de_sim_net(top_subgraphs, h_highdim, top_list[0])
    true_v = top_subgraphs.edges[top_list[0]].data['value']
    l_similarity = loss_fc[2](pred_similarity, true_v)

    xp, _ = de_dis_net(top_graph, H)
    lt = top_graph.edges['interacts'].data['label']
    if xp.shape[0]==0 or xp.shape[0]!= lt.shape[0]: return [None]
    [dis_cmpt_lp], [dis_gmm] = de_gmm_net(xp) 

    tmp = torch.div( torch.ones(ncluster), ncluster)
    ids, n = list(), (lt.shape[0])*tmp*0.8
    for i in torch.arange(ncluster):
        idx = ((lt == i).nonzero(as_tuple=True)[0]).view(-1,)
        if idx.nelement()==0: continue      
        p = torch.ones_like(idx)/idx.shape[0]
        # nidx = p.multinomial( num_samples=int( n[i]), replacement=True)
        nidx = p.multinomial(num_samples=int( torch.minimum(n[i], 16*torch.tensor(idx.shape[0])) ), replacement=True)
        # if n[i] > torch.tensor(idx.shape[0]):
        #     idx = (idx.repeat(2)).long()
        # else:
        #     nidx = p.multinomial( num_samples=int(n[i]), replacement=False)
        #     idx = idx[nidx]
        idx = idx[nidx]
        ids.append(idx.view(-1,))
    mask = torch.cat(ids, dim=0)
    mask, _ = torch.sort(mask)
    # mask = torch.unique(mask, sorted=True, return_inverse=False, return_counts=False)

    sample_dis_cmpt_lp = dis_cmpt_lp[mask, :]
    sample_lt = lt[mask]

    weight = torch.linspace(np.pi*0.1, np.pi*0.75, steps=ncluster, device=device)
    weight = torch.sin(weight) + 1.0 

    l_nll = loss_fc[0](dis_cmpt_lp, lt, weight)
    sample_l_nll = loss_fc[0](sample_dis_cmpt_lp, sample_lt, weight)
    one_hot_lt = torch.nn.functional.one_hot(sample_lt.long(), ncluster)
    l_wnl = loss_fc[1](sample_dis_cmpt_lp, one_hot_lt, weight)

    if require_grad:
        loss = 5*sample_l_nll + 5*l_wnl + l_similarity.nansum() + l_diff_g.nansum() # + l_wnl + l_stdl 
        optimizer[0].zero_grad()
        loss.backward()  # retain_graph=False, create_graph = True
        optimizer[0].step()

    return [l_nll.item(), l_wnl.item(), (l_similarity.sum()).item(), (l_diff_g.sum()).item()], dis_gmm

def inference(graphs, features, lr_ranges, num_heads, num_clusters, em_networks, ae_networks, device):
    top_graph = graphs['top_graph'].to(device)
    top_subgraphs = graphs['top_subgraphs'].to(device)

    h_feat = features

    em_bead = em_networks[0]
    en_net = ae_networks[0]
    de_dis_net = ae_networks[1]
    de_gmm_net = ae_networks[2]
    de_euc_net = ae_networks[3]
    de_sim_net = ae_networks[4]

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

def predict(graphs, features, num_heads, num_clusters, em_networks, ae_networks, device='cpu'):
    top_graph = graphs['top_graph'].to(device)
    top_subgraphs = graphs['top_subgraphs'].to(device)
    top_list = [e for e in top_subgraphs.etypes if 'interacts_c' in e]

    tp = top_graph.edges['interacts'].data['label'].cpu().detach().numpy()
    true_cluster_mat = np.ones((features.shape[0], features.shape[0]))*(num_clusters-1)
    
    xs,ys = graphs['top_graph'].edges(etype='interacts', form='uv')[0], graphs['top_graph'].edges(etype='interacts', form='uv')[1]
    true_cluster_mat[xs, ys] = tp

    h_feat = features
    em_bead = em_networks[0]
    en_net = ae_networks[0]
    de_dis_net = ae_networks[1]
    de_gmm_net = ae_networks[2]
    de_euc_net = ae_networks[3]
    de_sim_net = ae_networks[4]

    with torch.no_grad():

        _, [dis_gmm] = de_gmm_net( torch.ones(1, device=device) )
        mu = dis_gmm.component_distribution.mean.detach()
        stddev = dis_gmm.component_distribution.stddev.detach()
        lr_ranges = torch.exp(mu - stddev**2)

        Xf = em_bead(h_feat)
        h_center, h_highdim = en_net( top_subgraphs, Xf, lr_ranges, top_list, ['bead'])
        pred_X = h_center.cpu().detach().numpy()

        xp, _ = de_dis_net(top_graph, h_center)
        [dis_cmpt_lp], _ = de_gmm_net(xp)
        dp = torch.exp(dis_cmpt_lp).cpu().detach().numpy()

        pred_dist_cluster_mat = np.ones((pred_X.shape[0], pred_X.shape[0]))*(num_clusters-1)
        pred_dist_cluster_mat[xs, ys] = np.argmax(dp, axis=1)
        pred_dist_mat = np.zeros((pred_X.shape[0], pred_X.shape[0]))
        pred_dist_mat[xs, ys] = xp.view(-1,).cpu().detach().numpy()

        pdcm_list, pdm_list = list(), list()
        for i in np.arange(num_heads):
            xp = de_euc_net(top_graph, h_center[:,i,:], 'interacts')
            [dis_cmpt_lp], _ = de_gmm_net(xp)
            dp = torch.exp(dis_cmpt_lp).cpu().detach().numpy()

            pdcm = np.ones((pred_X.shape[0], pred_X.shape[0]))*(num_clusters-1)
            pdcm[xs, ys] = np.argmax(dp, axis=1)
            pdm = np.zeros((pred_X.shape[0], pred_X.shape[0]))
            pdm[xs, ys] = xp.view(-1,).cpu().detach().numpy()

            pdcm_list.append(pdcm)
            pdm_list.append(pdm)

        return pred_X, pred_dist_cluster_mat, pdcm_list, pred_dist_mat, pdm_list, [true_cluster_mat, dis_gmm]

def run_epoch(datasets, model, num_heads, num_clusters, loss_fc, optimizer, scheduler, iterations, device, writer=None, saved_model=None):
    train_dataset = datasets[0]
    valid_dataset = datasets[1] # if len(datasets) > 1 else None

    model_saved_path = saved_model[0] if saved_model is not None else None
    model_saved_name = saved_model[1] if saved_model is not None else None

    em_networks, ae_networks = model
    models_dict = {
        'embedding_model': em_networks[0],
        'encoder_model': ae_networks[0],
        'decoder_distance_model': ae_networks[1],
        'decoder_gmm_model': ae_networks[2],
        'decoder_euclidean_model': ae_networks[3],
        'decoder_similarity_model': ae_networks[4]
    }

    test_loss_list, valid_loss_list, dur = [], [], []
    best_nll_loss = 10
    lr_ranges = torch.linspace(1.0, 23.0, 
                            steps=num_clusters-1, 
                            requires_grad=False).to(device)
    for epoch in np.arange(iterations):
        print("epoch {:d} ".format(epoch), end=' ')
        t0 = time.time()
        for j, data in enumerate(train_dataset):
            graphs, features, cluster_weights, _ = data
            h_f, h_p = features['feat'], features['pos']
            h_f_n = torch.nn.functional.normalize(torch.tensor(h_f), p=1.0, dim=1)*h_f.shape[1]
            h_p_n =torch.nn.functional.normalize(torch.tensor(h_p), p=1.0, dim=1)*h_p.shape[1]
            h_feat = torch.stack([h_f_n, h_p_n], dim=1).to(device)
            h_feat = h_feat.float()

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
                        ae_networks[3].load_state_dict(checkpoint['decoder_euclidean_model_state_dict'])
                        ae_networks[4].load_state_dict(checkpoint['decoder_simlarity_model_state_dict'])
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
            graphs, features, cluster_weights, _ = data
 
            h_f, h_p = features['feat'], features['pos']
            h_f_n = torch.nn.functional.normalize(torch.tensor(h_f, dtype=torch.float), p=1.0, dim=1)*h_f.shape[1]
            h_p_n =torch.nn.functional.normalize(torch.tensor(h_p, dtype=torch.float), p=1.0, dim=1)*h_p.shape[1]
            h_feat = torch.stack( [h_f_n, h_p_n], dim=1).to(device)

            ll, _ = fit_one_step(False, graphs, h_feat, lr_ranges, em_networks, ae_networks, loss_fc, optimizer, device)
            if None in ll: continue
            valid_loss_list.append(ll)

            if epoch == 0 and j == 0 and writer is not None:
                # pass
                m = cluster_weights['mat']
                plot_feature([h_f_n, h_p_n], writer, '0, features/h')
                plot_cluster(m, writer, num_clusters,'0, cluster/bead', step=None)

            if epoch%3==0 and j == 0 and writer is not None: # and config is not None:
                # num_heads = int(config['parameter']['G3DM']['num_heads'])
                [X, pred_distance_mat, 
                center_true_mat, [dis_gmm], 
                distance_mat ] = inference(graphs, h_feat, lr_ranges, num_heads, num_clusters, 
                                            em_networks, ae_networks, device)

                plot_X(X, writer, '1, 3D/center', step=epoch)
                plot_cluster(pred_distance_mat, writer, num_clusters,
                            '2,1 cluster/prediction distance', step=epoch)

                plot_cluster(center_true_mat, writer, num_clusters,
                            '2,1 cluster/true', step=epoch) if epoch==0 else None

                plot_confusion_mat(pred_distance_mat, center_true_mat,  writer, 
                                '2,2 confusion matrix/normalize: ture, predicted distance - true contact', 
                                step=epoch, normalize='true')
                plot_confusion_mat(pred_distance_mat, center_true_mat,  writer, 
                                '2,2 confusion matrix/normalize: all, predicted distance - true contact', 
                                step=epoch, normalize='all')
                
                mu = (dis_gmm.component_distribution.mean)
                std = (dis_gmm.component_distribution.stddev)
                # std = (dis_gmm.component_distribution.variance)
                x = torch.linspace(start=0.1, end=7.0, steps=150, device=device) # mu.max()*1.5,
                log_pdfs = dis_gmm.component_distribution.log_prob(x.view(-1,1))
                # log_pdfs = log_pdfs + torch.log(dis_gmm.mixture_distribution.probs).view(1, -1)
                normal_pdfs = torch.exp(log_pdfs)*(dis_gmm.mixture_distribution.probs).view(1, -1)
                normal_pdfs = normal_pdfs.to('cpu').detach().numpy()
                weights = (dis_gmm.mixture_distribution.probs).to('cpu').detach().numpy()
                plot_distributions([ mu.to('cpu').detach().numpy(), 
                                    x.to('cpu').detach().numpy(), 
                                    normal_pdfs, weights], 
                                    writer, '2,3 hop_dist/Normal ln(x)~N(,)', step=epoch) 

                lognormal_pdfs = torch.empty(normal_pdfs.shape)
                lognormal_mu = torch.empty(mu.shape)
                lognormal_mode = torch.empty(mu.shape)
                x = torch.linspace(start=0.5, end=70.0, steps=150, device=device) # mu.max()*(1+1e-4)
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
        plot_scaler({'test':np.nanmean(test_ll[:,1]), 'validation': np.nanmean(valid_ll[:,2])}, writer, 'WNL Loss' ,step = epoch)
        plot_scaler({'test':np.nanmean(test_ll[:,2]), 'validation': np.nanmean(valid_ll[:,2])}, writer, 'High dim cos similarity Loss' ,step = epoch)
        plot_scaler({'test':np.nanmean(test_ll[:,3]), 'validation': np.nanmean(valid_ll[:,2])}, writer, 'High dim distance G Loss' ,step = epoch)

        dur.append(time.time() - t0)
        print("Loss:", np.mean(test_ll, axis=0), "| Time(s) {:.4f} ".format( np.mean(dur)), sep =" " )

    os.makedirs(model_saved_path, exist_ok=True)
    path = os.path.join(model_saved_path, 'finial_' + model_saved_name)
    save_model_state_dict(models_dict, optimizer[0], path)

def run_prediction(dataset, model, saved_parameters_model, num_heads, num_clusters, device='cpu'):
    model_saved_path = saved_parameters_model[0] if saved_parameters_model is not None else None
    model_saved_name = saved_parameters_model[1] if saved_parameters_model is not None else None

    em_networks, ae_networks = model
    models_dict = {
        'embedding_model': em_networks[0],
        'encoder_model': ae_networks[0],
        'decoder_distance_model': ae_networks[1],
        'decoder_gmm_model': ae_networks[2],
        'decoder_euclidean_model': ae_networks[3],
        'decoder_similarity_model': ae_networks[4]
    }

    path = os.path.join(model_saved_path, model_saved_name)
    checkpoint = torch.load(path, map_location=device)
    em_networks[0].load_state_dict(checkpoint['embedding_model_state_dict'])
    ae_networks[0].load_state_dict(checkpoint['encoder_model_state_dict'])
    ae_networks[1].load_state_dict(checkpoint['decoder_distance_model_state_dict'])
    ae_networks[2].load_state_dict(checkpoint['decoder_gmm_model_state_dict'])
    ae_networks[3].load_state_dict(checkpoint['decoder_euclidean_model_state_dict'])
    ae_networks[4].load_state_dict(checkpoint['decoder_simlarity_model_state_dict'])
    # optimizer[0].load_state_dict(checkpoint['optimizer_state_dict'])

    for key, m in models_dict.items():
        for param in list(m.parameters()):
            if param.isnan().any():
                print('Detected NaN in the parameters of {}, Exit'.format(key))
                return None

    prediction = dict()
    for i, data in enumerate(dataset):
        t0 = time.time()

        graphs, features, _, index = data
        h_f, h_p = features['feat'], features['pos']
        h_f_n = torch.nn.functional.normalize(torch.tensor(h_f), p=1.0, dim=1)*h_f.shape[1]
        h_p_n =torch.nn.functional.normalize(torch.tensor(h_p), p=1.0, dim=1)*h_p.shape[1]
        h_feat = torch.stack([h_f_n, h_p_n], dim=1).to(device)
        h_feat = h_feat.float()

        [pred_X, 
        pred_dist_cluster_mat, pdcm_list, 
        pred_dist_mat, pdm_list, 
        [true_cluster_mat, dis_gmm]] = predict(graphs, h_feat, num_heads, num_clusters, em_networks, ae_networks, device)
        prediction[index] = {'structures': pred_X, 
                            'predict_cluster': [pred_dist_cluster_mat, pdcm_list], 
                            'predict_distance': [pred_dist_mat, pdm_list],
                            'true_cluster': true_cluster_mat}
        prediction['mixture model'] = dis_gmm
        dur = time.time() - t0
        print( 'Time(s) {:.4f} '.format(dur))
    return prediction
