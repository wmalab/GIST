import torch
import dgl
import numpy as np

class embedding(torch.nn.Module):
    '''in_dim, out_dim'''
    def __init__(self, in_dim, out_dim, in_num_channels):
        super(embedding, self).__init__()
        self.conv1d_1 = torch.nn.Conv1d(in_num_channels, 8, 3, stride=1, padding=1, padding_mode='replicate')
        self.conv1d_2 = torch.nn.Conv1d(8, 16, 5, stride=1, padding=2, padding_mode='replicate')
        self.conv1d_3 = torch.nn.Conv1d(16, 4, 7, stride=3, padding=3, padding_mode='replicate')
        self.conv1d_4 = torch.nn.Conv1d(4, 1, 7, stride=3, padding=3, padding_mode='replicate')
        self.hidden_dim = np.floor((in_dim+2)/3).astype(float)
        self.hidden_dim = np.floor((self.hidden_dim+2)/3).astype(int)
        self.fc1 = torch.nn.Linear(self.hidden_dim, out_dim, bias=True)
        self.fc2 = torch.nn.Linear(out_dim, out_dim, bias=True)
        self.pool = torch.nn.MaxPool1d(3, stride=1, padding=1)
        self.reset()

    def reset(self):
        gain = torch.nn.init.calculate_gain('leaky_relu', 0.2)
        torch.nn.init.xavier_normal_(self.fc1.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.fc2.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.conv1d_1.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.conv1d_2.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.conv1d_3.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.conv1d_4.weight, gain=gain)

    def forward(self, h):
        # X = torch.nn.functional.normalize(h, p=2.0, dim=-1)
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
        X = torch.nn.functional.leaky_relu(X)
        X = self.fc2(X)
        # X = torch.nn.functional.normalize(X, p=2.0, dim=-1)
        # X = torch.nn.functional.leaky_relu(X)
        X = torch.squeeze(X, dim=1)
        return X

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

class encoder_chain(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, etypes):
        super(encoder_chain, self).__init__()

        l1 = dict()
        for et in etypes:
            '''l1[et] = dgl.nn.GraphConv( in_dim, hidden_dim, 
            norm='right', weight=True, allow_zero_in_degree=True)'''
            l1[et] = dgl.nn.GATConv( in_dim, hidden_dim, 
                                    num_heads=1, residual=False, 
                                    allow_zero_in_degree=True)
        self.layer1 = dgl.nn.HeteroGraphConv( l1, aggregate = 'mean')
        
        l2 = dict()
        for et in etypes:
            '''l2[et] = dgl.nn.GraphConv( hidden_dim, out_dim, 
                                        norm='right', weight=True, 
                                        allow_zero_in_degree=True)'''
            l2[et] = dgl.nn.GATConv( hidden_dim, out_dim, 
                                    num_heads=1, residual=False, 
                                    allow_zero_in_degree=True)
        self.layer2 = dgl.nn.HeteroGraphConv( l2, aggregate = self.agg_func)
        
        lMH = dict()
        for et in etypes:
            lMH[et] = dgl.nn.GATConv( out_dim, out_dim, 
                                    num_heads=num_heads, residual=False, 
                                    allow_zero_in_degree=True)
        self.layerMHs = dgl.nn.HeteroGraphConv( lMH, aggregate='mean')

        '''self.chain = constrainLayer(out_dim)'''
        self.num_heads = num_heads

        self.fc = torch.nn.Linear(len(etypes), len(etypes), bias=False)
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_normal_(self.fc.weight, gain=gain)

        self.r = torch.nn.Parameter(torch.empty((1)), requires_grad=True)
        self.register_parameter('r', self.r)
        torch.nn.init.uniform_(self.r, a=0.1, b=0.2)

    def agg_func(self, tensors, dsttype):
        stacked = torch.stack(tensors, dim=-1)
        res = self.fc(stacked)
        return torch.mean(res, dim=-1)

    def forward(self, g, x, etypes, efeat, ntype):

        subg_interacts = g.edge_type_subgraph(etypes[:-1])
        # edge_weight = subg_interacts.edata[efeat[0]]

        '''h = self.layer1(subg_interacts, {ntype[0]: x }, {'edge_weight':edge_weight})
        h = self.layer2(subg_interacts, h, {'edge_weight':edge_weight})'''
        h = self.layer1(subg_interacts, {ntype[0]: x })
        h = self.layer2(subg_interacts, h)

        '''subg_chain = g.edge_type_subgraph([etypes[1]])
        radius = torch.clamp(self.r, min=10e-3, max=3)
        dh = self.chain(subg_chain, h[ntype[0]], radius)
        conh = torch.cumsum(dh, dim=-2)
        h = self.layerMHs(subg_interacts, {ntype[0]: conh })'''

        h = self.layerMHs(subg_interacts, h) #{ntype[0]: h })

        res = list()
        for i in torch.arange(self.num_heads):
            '''ds = self.chain(subg_chain, h[ntype[0]][:,i,:], radius)
            s = torch.cumsum(ds, dim=-2)
            res.append(s)'''
            x = h[ntype[0]][:,i,:]
            '''vmin, _ = torch.min(x, dim=0, keepdim=True)
            vmax, _ = torch.max(x, dim=0, keepdim=True)
            x = (x - vmin)/(vmax-vmin)'''
            vmean = torch.mean(x, dim=0, keepdim=True)
            x = x - vmean
            res.append(x)
        res = torch.stack(res, dim=1)
        return res

class encoder_bead(torch.nn.Module): 
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(encoder_bead, self).__init__()
        '''self.layer1 = dgl.nn.GraphConv( in_dim, hidden_dim, 
                                        norm='right', weight=True, 
                                        allow_zero_in_degree=True)
        self.layer2 = dgl.nn.GraphConv( hidden_dim, out_dim, 
                                        norm='right', weight=True, 
                                        allow_zero_in_degree=True)
        self.layer3 = dgl.nn.GraphConv( out_dim, out_dim, 
                                        norm='right', weight=True, 
                                        allow_zero_in_degree=True)'''
        self.layer1 = dgl.nn.SAGEConv( in_dim, hidden_dim, 'lstm',
                                        norm=None)
        self.layer2 = dgl.nn.SAGEConv( hidden_dim, out_dim, 'lstm',
                                        norm=None)
        self.layer3 = dgl.nn.SAGEConv( out_dim, out_dim, 'lstm',
                                        norm=None)
        # self.norm = dgl.nn.EdgeWeightNorm(norm='both')

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
        return torch.stack(res, dim=1)

'''class encoder_union(torch.nn.Module):
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
        return res'''

class encoder_union(torch.nn.Module):
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
        return res

class MergeLayer(torch.nn.Module):
    def __init__(self, in_h0_dim, in_h1_dim, out_dim):
        super(MergeLayer, self).__init__()
        # src: center h1 -> dst: bead h0
        # self.fcsrc = torch.nn.Linear(in_h1_dim, out_dim, bias=False) 
        self.fcdst = torch.nn.Linear(in_h0_dim, out_dim, bias=False)
        self.attn_interacts = torch.nn.Linear(2*out_dim, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
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
        h = (torch.mean(nodes.mailbox['dst_z'], dim=1) + n*torch.sum(alpha*(nodes.mailbox['src_z']), dim=1))/(n)
        return {'ah': h}

    def forward(self, graph, h0, h1):
        with graph.local_scope():
            # graph.srcdata['z'] = self.fcsrc(h1)
            graph.srcdata['z'] = h1
            graph.dstdata['z'] = self.fcdst(h0)
            graph.apply_edges(self.edge_attention)
            graph.update_all(self.message_func, self.reduce_func)
            res = graph.ndata.pop('ah')['h0_bead']
            return res

class MultiHeadMergeLayer(torch.nn.Module):
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
            return torch.mean(torch.stack(head_outs))

class decoder(torch.nn.Module):
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

    def aritficial_fc(self, x):
        upper = self.upper_bound - x
        lower = x - self.lower_bound
        score = (upper*lower)/(self.r_dist**2 + 1)
        return score

        '''m = torch.relu(mean)
        score = 1.0/((x - m)**2 + self.b)
        return score'''

    def edge_distance(self, edges):
        n2 = torch.norm((edges.dst['z'] - edges.src['z']), dim=-1)
        weight = torch.nn.functional.softmax(self.w, dim=0)
        dist = torch.sum(n2*weight, dim=-1, keepdim=True)
        outputs_dist = self.aritficial_fc(dist)
        std = torch.sqrt(torch.mean((n2 - dist)**2, dim=-1, keepdim=True))
        return {'dist_pred': outputs_dist, 'std': std}

    def forward(self, g, h):
        with g.local_scope():
            g.nodes[self.ntype].data['z'] = h
            self.upper_bound = torch.matmul(torch.abs(self.r_dist), self.upones)+0.1
            self.lower_bound = torch.matmul(torch.abs(self.r_dist), self.lowones)
            g.apply_edges(self.edge_distance, etype=self.etype)
            return g.edata.pop('dist_pred'), g.edata.pop('std')

