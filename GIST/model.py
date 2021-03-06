import torch
from torch import distributions as D 
import dgl
import numpy as np

class embedding(torch.nn.Module):
    '''in_dim, out_dim'''
    def __init__(self, in_dim, out_dim, in_num_channels):
        super(embedding, self).__init__()
        self.conv1d_1 = torch.nn.Conv1d(in_num_channels, 8, 3, stride=1, padding=1, padding_mode='replicate')
        self.conv1d_2 = torch.nn.Conv1d(8, 32, 5, stride=1, padding=2, padding_mode='replicate')
        self.conv1d_3 = torch.nn.Conv1d(32, 8, 7, stride=3, padding=3, padding_mode='replicate')
        self.conv1d_4 = torch.nn.Conv1d(8, 1, 7, stride=3, padding=3, padding_mode='replicate')
        self.hidden_dim = np.floor((in_dim+2)/3).astype(float)
        self.hidden_dim = np.floor((self.hidden_dim+2)/3).astype(int)
        self.fc1 = torch.nn.Linear(self.hidden_dim, out_dim, bias=True)
        self.pool = torch.nn.MaxPool1d(3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm1d(num_features=out_dim)
        self.reset()

    def reset(self):
        gain = torch.nn.init.calculate_gain('leaky_relu', 0.2)
        torch.nn.init.xavier_normal_(self.fc1.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.conv1d_1.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.conv1d_2.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.conv1d_3.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.conv1d_4.weight, gain=gain)

    def forward(self, h):
        X = self.conv1d_1(h)
        X = torch.nn.functional.leaky_relu(X)
        X = self.conv1d_2(X)
        X = torch.nn.functional.leaky_relu(X)
        X = self.pool(X)
        X = self.conv1d_3(X) # ceil( (Lin+2)/3 )
        X = torch.nn.functional.leaky_relu(X)
        X = self.conv1d_4(X) # ceil( (Lin+2)/3 )
        X = torch.nn.functional.leaky_relu(X)
        X = self.pool(X)
        X = self.fc1(X)
        X = torch.squeeze(X, dim=1)
        X = self.bn(X)
        return X


class encoder_chain(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, etypes):
        super(encoder_chain, self).__init__()
        self.num_heads = num_heads

        l1 = dict()
        for et in etypes:
            l1[et] = dgl.nn.GATConv( in_dim, hidden_dim, 
                                    num_heads=1, residual=False, 
                                    allow_zero_in_degree=True)
        self.layer1 = dgl.nn.HeteroGraphConv( l1, aggregate = self.agg_func1)

        l2 = dict()
        for et in etypes:
            l2[et] = dgl.nn.GATConv( hidden_dim, hidden_dim, 
                                    num_heads=1, residual=False, 
                                    allow_zero_in_degree=True)
        self.layer2 = dgl.nn.HeteroGraphConv( l2, aggregate = self.agg_func2)

        lMH = dict()
        for et in etypes:
            lMH[et] = dgl.nn.GATConv( hidden_dim, out_dim, 
                                    num_heads=num_heads, residual=False, 
                                    allow_zero_in_degree=True)
        self.layerMHs = dgl.nn.HeteroGraphConv( lMH, aggregate=self.agg_funcMH)

        self.fc1 = torch.nn.Linear(len(etypes)*hidden_dim, hidden_dim, bias=True)
        self.fc2 = torch.nn.Linear(len(etypes)*hidden_dim, hidden_dim, bias=True)
        # self.fc2 = torch.nn.Linear(len(etypes), len(etypes), bias=False)
        self.fcmh = torch.nn.Linear(len(etypes), len(etypes), bias=False)

        gain = torch.nn.init.calculate_gain('leaky_relu', 0.2)
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fcmh.weight, gain=gain)


    def agg_func1(self, tensors, dsttype):
        concated = torch.cat(tensors, dim=-1)
        res = self.fc1(concated)
        return res

    def agg_func2(self, tensors, dsttype):
        # stacked = torch.stack(tensors, dim=-1)
        concated = torch.cat(tensors, dim=-1)
        res = self.fc2(concated)
        return res # torch.mean(res, dim=-1)

    def agg_funcMH(self, tensors, dsttype):
        stacked = torch.stack(tensors, dim=-1)
        res = self.fcmh(stacked)
        return torch.mean(res, dim=-1)

    def norm_(self, x):
        xp = torch.cat([torch.zeros((1,3), device=x.device), x[0:-1, :]], dim=0)
        dx = x - xp
        dx = torch.nan_to_num(dx, nan=0)
        dmean = torch.median( torch.norm(dx, dim=-1))+1e-4
        x = torch.cumsum(torch.div(dx, dmean)*1.0, dim=0)
        return x

    def forward(self, g, x, lr_ranges, etypes, ntype):
        subg_interacts = g.edge_type_subgraph(etypes)
        # sub_0 = g.edge_type_subgraph([etypes[0]])
        # lr_ranges = torch.cat( ( 0.8*torch.ones((1), device=lr_ranges.device), lr_ranges.view(-1,), 
        #                         torch.tensor(float('inf'), device=lr_ranges.device).view(-1,) ), 
        #                         dim=0).float().to(lr_ranges.device)

        h = self.layer1(subg_interacts, {ntype[0]: x })
        h = torch.squeeze(h[ntype[0]], dim=1)
        h = self.layer2(subg_interacts, {ntype[0]: h })
        x = torch.squeeze(h[ntype[0]], dim=1)
        h_res = x
        h = self.layerMHs(subg_interacts, {ntype[0]: x })
        res = list()
        for i in torch.arange(self.num_heads):
            x = h[ntype[0]][:,i,:]
            x = self.norm_(x)
            x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
            # dist = torch.distributions.Normal(x, 0.1*torch.ones_like(x))
            # x = dist.rsample()
            res.append(x)
        res = torch.stack(res, dim=1)
        return res, h_res

class decoder_euclidean(torch.nn.Module):
    def __init__(self):
        super(decoder_euclidean, self).__init__()

    def edge_distance(self, edges):
        dist = torch.norm((edges.dst['h'] - edges.src['h']), dim=-1, keepdim=True)
        return {'distance_score': dist}
    
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
            graph.apply_edges(self.edge_distance, etype=etype)
            return graph.edges[etype].data.pop('distance_score') # graph.edges[etype].data['dotproduct_score'], 

class decoder_similarity(torch.nn.Module):
    def __init__(self):
        super(decoder_similarity, self).__init__()
    
    # def edge_distance(self, edges):
    #     cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    #     output = cos(edges.dst['h'], edges.src['h'])
    #     return {'cosine_score': output}
    
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
            # graph.apply_edges(self.edge_distance, etype=etype)
            graph.apply_edges(dgl.function.u_dot_v('h', 'h', 'similarity_score'), etype=etype)
            res = graph.edges[etype].data.pop('similarity_score')
            return res.clamp(min=-0.9)

class decoder_distance(torch.nn.Module):
    ''' num_heads, num_clusters, ntype, etype '''
    def __init__(self, num_heads, num_clusters, ntype, etype):
        # True: increasing, Flase: decreasing
        super(decoder_distance, self).__init__()
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.ntype = ntype
        self.etype = etype

        self.w = torch.nn.Parameter(torch.ones( (self.num_heads)), requires_grad=True)
        self.register_parameter('w', self.w)
        # torch.nn.init.uniform_(self.w, a=-10.0, b=10.0)

    def edge_distance(self, edges):
        n2 = torch.norm((edges.dst['z'] - edges.src['z']), dim=-1, keepdim=False)
        weight = torch.nn.functional.softmax(self.w.clamp(min=-3.0, max=3.0), dim=0)
        dist = torch.sum(n2*weight, dim=-1, keepdim=True)
        # std, mean = torch.std_mean(n2, dim=-1, unbiased=False, keepdim=False)
        return {'dist_pred': dist}

    def forward(self, g, h):
        with g.local_scope():
            g.nodes[self.ntype].data['z'] = h
            g.apply_edges(self.edge_distance, etype=self.etype)
            return g.edata.pop('dist_pred') #, g.edata.pop('std')

class decoder_gmm(torch.nn.Module):
    def __init__(self, num_clusters):
        super(decoder_gmm, self).__init__()
        self.num_clusters = num_clusters

        self.k = torch.nn.Parameter( torch.ones(self.num_clusters), requires_grad=True)

        ms = torch.linspace(0, 4.0, steps=self.num_clusters, dtype=torch.float, requires_grad=True)
        self.means = torch.nn.Parameter( ms, requires_grad=True)

        self.distance_stdevs = torch.nn.Parameter( 0.3*torch.ones((self.num_clusters)), requires_grad=True)

        inter = torch.linspace(start=0, end=0.1, steps=self.num_clusters)
        self.interval = torch.nn.Parameter( inter, requires_grad=False)

        self.cweight = torch.nn.Parameter( torch.zeros((self.num_clusters)), requires_grad=True)
        self.bias =  torch.nn.Parameter( torch.linspace(1e-8, 1e-5, steps=self.num_clusters, dtype=torch.float), requires_grad=False)

    # gmm
    def fc(self, stds_l, stds_r, k):
        k = torch.sigmoid(k.clamp(min=-torch.log(torch.tensor(9.0)), max=9.0))
        rate = torch.div(stds_l, stds_r)
        kr = (k*rate).clamp(min=1e-8, max=0.9) # must < 1
        return stds_l * torch.sqrt( -2.0*torch.log(kr) + 1e-8  )

    def forward(self, distance):
        cweight = torch.nn.functional.softmax(self.cweight.view(-1,), 0)
        mix = D.Categorical(cweight)

        stds = torch.relu(self.distance_stdevs) + 1e-3

        d_left = self.fc(stds, stds, self.k)
        d_left = torch.cumsum(d_left, dim=0)

        d_right = self.fc(stds[0:-1], stds[1:], self.k[1:])
        d_right = torch.cat( (torch.zeros(1, device=d_right.device), d_right), dim=0)
        # d_right = torch.cumsum(d_right, dim=0)
        means = (d_left + d_right)
        means = (means + self.interval).clamp(max=5.0)

        # activate = torch.nn.LeakyReLU(0.01)
        # means = activate(self.means).clamp(max=4.5)
        # means = torch.nan_to_num(means, nan=4.5)
        # means = means + self.interval
        # stds = torch.relu(self.distance_stdevs) + 1e-3

        mode = torch.exp(means - stds**2)
        _, idx = torch.sort(mode.view(-1,), dim=0, descending=False)
        means = means[idx]
        stds = stds[idx]

        dis_cmp = D.Normal(means.view(-1,), stds.view(-1,))
        dis_gmm = D.MixtureSameFamily(mix, dis_cmp)
        data = torch.log(distance).view(-1,1)

        unsafe_dis_cmpt_lp = dis_gmm.component_distribution.log_prob(data)
        dis_cmpt_lp = torch.nan_to_num(unsafe_dis_cmpt_lp, nan=-float('inf'))

        dis_cmpt_p = torch.exp(dis_cmpt_lp) * (dis_gmm.mixture_distribution.probs).view(1,-1) + self.bias
        dis_cmpt_p = torch.nn.functional.normalize(dis_cmpt_p, p=1, dim=1)
        dis_cmpt_lp = torch.log(dis_cmpt_p)

        return [dis_cmpt_lp.float()], [dis_gmm]

def save_model_state_dict(models, optimizer, path, epoch=None, loss=None):
    state_dict = {
        'embedding_model_state_dict': models['embedding_model'].state_dict(),
        'encoder_model_state_dict': models['encoder_model'].state_dict(),
        'decoder_distance_model_state_dict': models['decoder_distance_model'].state_dict(),
        'decoder_gmm_model_state_dict': models['decoder_gmm_model'].state_dict(),
        'decoder_euclidean_model_state_dict': models['decoder_euclidean_model'].state_dict(),
        'decoder_simlarity_model_state_dict': models['decoder_similarity_model'].state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    if epoch is not None:
        state_dict['epoch'] = epoch
    if loss is not None:
        state_dict['nll_loss'] = loss

    torch.save(state_dict, path)

def save_model_entire():
    pass


"""
# for i, et in enumerate(etypes):
#     x = self.layerConstruct(subg_interacts, x, [lr_ranges[i], lr_ranges[i+2]], et)
class ConstructLayer(torch.nn.Module):
    def __init__(self):
        super(ConstructLayer, self).__init__()
        # src: h1 -> dst: h0

    def edge_scale(self, edges):
        d = edges.dst['z'] - edges.src['z']
        z2 = torch.norm(d, dim=-1, keepdim=True) + 1e-5
        clamp_d = torch.div(d, z2)*(z2.float().clamp(min=self.le, max=self.ge))
        return {'e': clamp_d}

    def message_func(self, edges):
        return {'src_z': edges.src['z'], 'e': edges.data['e'], 'dst_z': edges.dst['z']}

    def reduce_func(self, nodes):
        h = torch.mean( nodes.mailbox['src_z'] + nodes.mailbox['e'], dim=1)
        return {'ah': h}

    def forward(self, graph, h, lr_ranges, etype):
        self.le, self.ge = lr_ranges
        with graph.local_scope():
            graph.ndata['z'] = h
            graph.apply_edges(self.edge_scale, etype=etype)
            graph.update_all(self.message_func, self.reduce_func, etype=etype)
            res = graph.ndata.pop('ah')
            return res"""

"""class constrainLayer(torch.nn.Module):
    def __init__(self, in_dim):
        super(constrainLayer, self).__init__()
        self.alpha_fc = torch.nn.Linear(in_dim, 1, bias=True)
        self.beta_fc = torch.nn.Linear(in_dim, 1, bias=True)
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_normal_(self.alpha_fc.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.beta_fc.weight, gain=gain)

    def forward(self, g, h, r):
        g.ndata['z'] = h
        message_func = dgl.function.u_sub_v('z', 'z', 'm')
        reduce_func = dgl.function.sum('m', 'h')
        g.update_all(message_func, reduce_func)
        h = g.ndata['h']
        l = torch.norm(h, p=2, dim=-1, keepdim=True) + 1e-7
        dh = (h/l) + 1e-4
        '''ha = self.alpha_fc(h)
        hb = self.beta_fc(h)
        x = r * torch.sin(ha) * torch.cos(hb)
        y = r * torch.sin(ha) * torch.sin(hb)
        z = r * torch.cos(ha)
        dh = torch.cat([x,y,z], dim=-1)'''
        return dh
        # return h"""

"""class encoder_bead(torch.nn.Module): 
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(encoder_bead, self).__init__()
        '''self.layer1 = dgl.nn.GraphConv( in_dim, hidden_dim, 
                                        norm='none', weight=True, 
                                        allow_zero_in_degree=True)
        self.layer2 = dgl.nn.GraphConv( hidden_dim, out_dim, 
                                        norm='none', weight=True, 
                                        allow_zero_in_degree=True)
        self.layer3 = dgl.nn.GraphConv( out_dim, out_dim, 
                                        norm='none', weight=True, 
                                        allow_zero_in_degree=True)'''
        self.layer1 = dgl.nn.SAGEConv( in_dim, hidden_dim, 'lstm',
                                        norm=None)
        self.layer2 = dgl.nn.SAGEConv( hidden_dim, out_dim, 'lstm',
                                        norm=None)
        self.layer3 = dgl.nn.SAGEConv( out_dim, out_dim, 'lstm',
                                        norm=None)
        self.norm = dgl.nn.EdgeWeightNorm(norm='both')

    def forward(self, blocks, x, etypes, efeat):
        edge_weights = [sub.edata[efeat[0]] for sub in blocks]
        # norm_edge_weights = [ self.norm(blocks[i], w) for i, w in enumerate(edge_weights)]
        
        num = x.shape[1]
        res = []
        for i in np.arange(num):
            h = x[:,i,:]
            block = blocks[0]
            h = self.layer1(block, h, edge_weight=edge_weights[0])

            block = blocks[1]
            h = self.layer2(block, h, edge_weight=edge_weights[1])

            block = blocks[2]
            h = self.layer3(block, h, edge_weight=edge_weights[2])
            res.append(h)
        return torch.stack(res, dim=1)"""

"""class encoder_union(torch.nn.Module):
    # src: center -> dst: bead
    def  __init__(self, in_h1_dim, in_h0_dim, out_dim, in_h1_heads, in_h0_heads, out_heads):
        super(encoder_union, self).__init__()
        self.layer_merge = dgl.nn.GATConv((in_h1_dim, in_h0_dim), out_dim, 
                                            num_heads=in_h0_heads, 
                                            allow_zero_in_degree=True)
        self.in_h1_heads = in_h1_heads

        self.wn_fc = torch.nn.utils.weight_norm( torch.nn.Linear(in_features=in_h0_heads*in_h1_heads, out_features=out_heads) )
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_normal_(self.wn_fc.weight, gain=gain)

    def normWeight(self, module): # 
        module.weight.data = torch.softmax(module.weight.data, dim=1)

    def forward(self, graph, hier_1, hier_0):
        res = []
        for i in torch.arange(self.in_h1_heads):
            k = self.layer_merge(graph, (hier_1[:,i,:], hier_0))
            res.append(k)
        res = torch.cat(res, dim=1)
        res = torch.transpose(res, 1, 2)
        self.normWeight(self.wn_fc)
        res = self.wn_fc(res)
        res = torch.transpose(res, 1, 2)
        return res"""

"""class encoder_union(torch.nn.Module):
    # src: center -> dst: bead
    def  __init__(self, in_h1_dim, in_h0_dim, out_dim, in_h1_heads, in_h0_heads, out_heads):
        super(encoder_union, self).__init__()
        '''self.layer_merge = dgl.nn.GATConv((in_h1_dim, in_h0_dim), out_dim, 
                                            num_heads=in_h0_heads, 
                                            allow_zero_in_degree=True)'''
        self.layer_merge = MultiHeadMergeLayer(in_h0_dim, in_h1_dim, out_dim, in_h0_heads, merge='stack')
        self.in_h1_heads = in_h1_heads

        self.wn_fc = torch.nn.Linear(in_features=in_h0_heads*in_h1_heads, out_features=out_heads, bias=False)
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_normal_(self.wn_fc.weight, gain=gain)

    def normWeight(self, module): # 
        w = torch.relu(module.weight.data)
        module.weight.data = w/(torch.sum(w, dim=0, keepdim=True))

    def forward(self, graph, hier_1, hier_0):
        res = []
        for i in torch.arange(self.in_h1_heads):
            k = self.layer_merge(graph, (hier_0, hier_1[:,i,:]))
            # k = self.layer_merge(graph, (hier_1[:,i,:], hier_0))
            res.append(k)
        res = torch.cat(res, dim=2)
        '''res = torch.cat(res, dim=1)
        res = torch.transpose(res, 1, 2)'''
        self.normWeight(self.wn_fc)
        res = self.wn_fc(res)
        res = torch.transpose(res, 1, 2)
        return res"""

"""class MergeLayer(torch.nn.Module):
    def __init__(self, in_h0_dim, in_h1_dim, out_dim):
        super(MergeLayer, self).__init__()
        # src: center h1 -> dst: bead h0
        # self.fcsrc = torch.nn.Linear(in_h1_dim, out_dim, bias=False) 
        self.fcdst = torch.nn.Linear(in_h0_dim, out_dim, bias=False)
        self.attn_interacts = torch.nn.Linear(2*out_dim, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        '''Reinitialize learnable parameters.'''
        gain = torch.nn.init.calculate_gain('relu')
        # torch.nn.init.xavier_normal_(self.fcsrc.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.fcdst.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.attn_interacts.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_interacts(z2)
        return {'e': torch.nn.functional.leaky_relu(a)}

    def message_func(self, edges):
        return {'src_z': edges.src['z'], 'e': edges.data['e'], 'dst_z': edges.dst['z']}

    def reduce_func(self, nodes):
        alpha = torch.nn.functional.softmax(nodes.mailbox['e'], dim=1)
        n = nodes.mailbox['src_z'].shape[1]
        h = (torch.mean(nodes.mailbox['dst_z'], dim=1) + n*torch.mean(alpha*(nodes.mailbox['src_z']), dim=1))/(n)
        return {'ah': h}

    def forward(self, graph, h0, h1):
        with graph.local_scope():
            # graph.srcdata['z'] = self.fcsrc(h1)
            graph.srcdata['z'] = h1
            graph.dstdata['z'] = self.fcdst(h0)
            graph.apply_edges(self.edge_attention)
            graph.update_all(self.message_func, self.reduce_func)
            res = graph.ndata.pop('ah')['h0_bead']
            return res"""

"""class MultiHeadMergeLayer(torch.nn.Module):
    def __init__(self, in_h0_dim, in_h1_dim, out_dim, num_heads, merge='stack'):
        super(MultiHeadMergeLayer, self).__init__()
        self.heads = torch.nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(MergeLayer(in_h0_dim, in_h1_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h[0], h[1]) for attn_head in self.heads]
        if self.merge == 'stack':
            # stack on the output feature dimension (dim=1)
            return torch.stack(head_outs, dim=-1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))"""

"""class decoder(torch.nn.Module):
    ''' num_heads, num_clusters, ntype, etype '''
    def __init__(self, num_heads, num_clusters, ntype, etype):
        # True: increasing, Flase: decreasing
        super(decoder, self).__init__()
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.ntype = ntype
        self.etype = etype

        num_seq = num_clusters

        self.w = torch.nn.Parameter(torch.empty( (self.num_heads)), requires_grad=True)
        self.register_parameter('w', self.w)
        torch.nn.init.uniform_(self.w, a=0.0, b=1.0)

        self.r_dist = torch.nn.Parameter(torch.empty((1, num_seq)), requires_grad=True)
        self.register_parameter('r_dist',self.r_dist) 
        torch.nn.init.uniform_(self.r_dist, a=0.2, b=0.3)

        upones = torch.ones((num_seq, num_seq))
        upones = torch.triu(upones)
        self.upones = torch.nn.Parameter( upones, requires_grad=False)
        
        lowones = torch.ones((num_seq, num_seq))
        lowones = torch.triu(lowones, diagonal=1)
        self.lowones = torch.nn.Parameter( lowones, requires_grad=False)

        self.v_cluster = torch.nn.Parameter( torch.arange(num_clusters, dtype=torch.int32), requires_grad=False)

    def dim2_score(self, x):
        upper = self.upper_bound - x
        lower = x - self.lower_bound
        score = 10*(upper*lower)/(self.r_dist**2 + 1)
        return score

    # def dim3_score(self, x):
    #     upper = self.upper_bound.view(1,1,-1) - torch.unsqueeze(x, dim=-1)
    #     lower = torch.unsqueeze(x, dim=-1) - self.lower_bound.view(1,1,-1)
    #     score = (upper*lower)/(self.r_dist.view(1,1,-1)**2 + 1)
    #     return score

    def edge_distance(self, edges):
        n2 = torch.norm((edges.dst['z'] - edges.src['z']), dim=-1, keepdim=False)
        weight = torch.nn.functional.softmax(self.w, dim=0)

        dist = torch.sum(n2*weight, dim=-1, keepdim=True)
        outputs_score = self.dim2_score(dist)

        # dist = torch.mean(n2, dim=-1, keepdim=True)
        # outputs_score = self.dim2_score(dist)

        # score = self.dim3_score(n2)
        # prob = torch.softmax(score, dim=-1)
        # clusters = torch.sum(prob * self.v_cluster.view(1,1,-1), dim=-1, keepdim=False)
        # mean = torch.sum(clusters*weight.view(1,-1), dim=-1, keepdim=True)
        # diff = clusters - mean
        # std = torch.sqrt(torch.sum(diff**2, dim=-1, keepdim=True))

        std = torch.std(n2, dim=-1, unbiased=True, keepdim=False)
        return {'dist_pred': outputs_score, 'std': std}

    def forward(self, g, h):
        with g.local_scope():
            g.nodes[self.ntype].data['z'] = h
            # r = 10/torch.sum(torch.abs(self.r_dist))
            r = self.r_dist.clamp(min=0.1)
            self.upper_bound = (torch.matmul(r, self.upones)).clamp(min=0.0, max=15.0) # *r
            self.lower_bound = (torch.matmul(r, self.lowones)).clamp(min=0.01, max=15.0) # *r

            g.apply_edges(self.edge_distance, etype=self.etype)
            return g.edata.pop('dist_pred'), g.edata.pop('std')"""

"""class decoder(torch.nn.Module):
    ''' num_heads, num_clusters, ntype, etype '''
    def __init__(self, num_heads, num_clusters, ntype, etype):
        # True: increasing, Flase: decreasing
        super(decoder, self).__init__()
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.ntype = ntype
        self.etype = etype

        num_seq = num_clusters

        self.w = torch.nn.Parameter(torch.empty( (self.num_heads)), requires_grad=True)
        self.register_parameter('w', self.w)
        torch.nn.init.uniform_(self.w, a=-10.0, b=10.0)

        self.bottom = torch.tensor(0.0, dtype=torch.float32)
        self.register_buffer('bottom_const', self.bottom)

        self.top = torch.tensor(10.0, dtype=torch.float32)
        self.register_buffer('top_const', self.top)

        self.drange = torch.linspace(self.bottom_const, 1.0, steps=num_seq-1, dtype=torch.float, requires_grad=True)
        self.in_dist = torch.nn.Parameter(self.drange+0.1, requires_grad=True)
        self.register_parameter('in_dist', self.in_dist)

        # self.in_dist = torch.nn.Parameter( torch.eye(num_step, num_seq-1), requires_grad=True)
        # # self.in_dist = torch.nn.Parameter( torch.empty((num_step, num_seq-1)), requires_grad=True)
        # # torch.nn.init.uniform_(self.in_dist, a=-10.0, b=10.0)
        # self.register_parameter('in_dist', self.in_dist)

        mat = torch.diag( -1*torch.ones((num_seq+1)), diagonal=0) + torch.diag( torch.ones((num_seq)), diagonal=-1)
        self.subtract_mat = torch.nn.Parameter(mat[:,:-1], requires_grad=False)



    def dim2_score(self, x):
        upper = self.upper_bound - x
        lower = x - self.lower_bound
        score = torch.clamp( (4.0*upper*lower)/(self.r_dist**2 + 1), min=-6.0, max=6.0)
        score = (torch.nn.functional.sigmoid(score)*2.0-1)*10.0
        return score

    # def dim3_score(self, x):
    #     upper = self.upper_bound.view(1,1,-1) - torch.unsqueeze(x, dim=-1)
    #     lower = torch.unsqueeze(x, dim=-1) - self.lower_bound.view(1,1,-1)
    #     score = (4.0*upper*lower)/(self.r_dist.view(1,1,-1)**2 + 1.0)
    #     score = (torch.nn.functional.sigmoid(score)*2.0-1)*10.0
    #     return score

    def edge_distance(self, edges):
        n2 = torch.norm((edges.dst['z'] - edges.src['z']), dim=-1, keepdim=False)
        weight = torch.nn.functional.softmax(self.w, dim=0)

        # score = self.dim3_score(n2)
        # outputs_score = torch.sum(score*weight.view(1,-1,1), dim=1)

        dist = torch.sum(n2*weight, dim=-1, keepdim=True)
        outputs_score = self.dim2_score(dist)
        std, mean = torch.std_mean(n2, dim=-1, unbiased=False, keepdim=False)
        return {'dist_pred': outputs_score, 'std': std/(mean+1.0)}

    def forward(self, g, h):
        with g.local_scope():
            g.nodes[self.ntype].data['z'] = h
            # sorted_in_d = self.in_dist.view(1,-1)
            # dist_mat = torch.softmax(self.in_dist, dim=0)
            dist = torch.square(self.in_dist) + torch.ones_like(self.in_dist)*0.01
            d = torch.cumsum( dist, dim=0)
            sorted_in_d = d.clamp(min=0.1, max=20.0).view(1,-1)
            # sorted_in_d, _ = torch.sort( in_d, dim=-1)

            self.lower_bound = torch.cat( (self.bottom_const.view(1,-1), 
                                        sorted_in_d), 
                                        dim=1)
            self.upper_bound = torch.cat( (sorted_in_d, 
                                        self.top_const.view(1,-1)), 
                                        dim=1)
            self.bound = torch.cat( (self.bottom_const.view(1,-1), 
                                    sorted_in_d, 
                                    self.top_const.view(1,-1)), 
                                    dim=1)
            self.r_dist = torch.relu( torch.matmul(self.bound, self.subtract_mat) )
            g.apply_edges(self.edge_distance, etype=self.etype)
            return g.edata.pop('dist_pred'), g.edata.pop('std')"""