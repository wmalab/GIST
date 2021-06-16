import torch
import dgl
import numpy as np

class embedding(torch.nn.Module):
    '''in_dim, out_dim'''
    def __init__(self, in_dim, out_dim):
        super(embedding, self).__init__()
        self.conv1d_1 = torch.nn.Conv1d(2, 8, 3, stride=1, padding=1, padding_mode='replicate')
        self.conv1d_2 = torch.nn.Conv1d(8, 1, 5, stride=1, padding=2, padding_mode='replicate')
        self.fc1 = torch.nn.Linear(in_dim, out_dim, bias=True)
        self.fc2 = torch.nn.Linear(out_dim, out_dim, bias=True)
        self.pool = torch.nn.MaxPool1d(3, stride=1, padding=1)
        
        self.reset()

    def reset(self):
        gain = torch.nn.init.calculate_gain('leaky_relu', 0.2)
        torch.nn.init.xavier_normal_(self.fc1.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.fc2.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.conv1d_1.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.conv1d_2.weight, gain=gain)

    def forward(self, h):
        X = torch.nn.functional.normalize(h, p=2.0, dim=-1)
        X = self.conv1d_1(X)
        X = torch.nn.functional.leaky_relu(X)
        X = self.conv1d_2(X)
        X = torch.nn.functional.leaky_relu(X)
        X = self.pool(X)
        X = self.fc1(X)
        X = torch.nn.functional.leaky_relu(X)
        X = self.fc2(X)
        X = torch.nn.functional.normalize(X, p=2.0, dim=-1)
        # X = torch.nn.functional.leaky_relu(X)
        X = torch.squeeze(X, dim=1)
        return X

class constrainLayer(torch.nn.Module):
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
        h = (h/l)
        '''ha = self.alpha_fc(h)
        hb = self.beta_fc(h)
        x = r * torch.sin(ha) * torch.cos(hb)
        y = r * torch.sin(ha) * torch.sin(hb)
        z = r * torch.cos(ha)
        dh = torch.cat([x,y,z], dim=-1)
        return dh'''
        return h

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
        self.layer2 = dgl.nn.HeteroGraphConv( l2, aggregate = 'mean')
        
        lMH = dict()
        for et in etypes:
            lMH[et] = dgl.nn.GATConv( out_dim, out_dim, 
                                    num_heads=num_heads, residual=False, 
                                    allow_zero_in_degree=True)
        self.layerMHs = dgl.nn.HeteroGraphConv( lMH, aggregate='mean')

        self.chain = constrainLayer(out_dim)
        self.num_heads = num_heads

        self.r = torch.nn.Parameter(torch.empty((1)), requires_grad=True)
        self.register_parameter('r', self.r)
        torch.nn.init.uniform_(self.r, a=0.0, b=1.0)


    def forward(self, g, x, etypes, efeat, ntype):

        subg_interacts = g.edge_type_subgraph(etypes[:-1])
        # edge_weight = subg_interacts.edata[efeat[0]]

        '''h = self.layer1(subg_interacts, {ntype[0]: x }, {'edge_weight':edge_weight})
        h = self.layer2(subg_interacts, h, {'edge_weight':edge_weight})'''
        h = self.layer1(subg_interacts, {ntype[0]: x })
        h = self.layer2(subg_interacts, h)

        subg_chain = g.edge_type_subgraph([etypes[1]])
        radius = torch.clamp(self.r, min=10e-3, max=3)
        dh = self.chain(subg_chain, h[ntype[0]], radius)
        conh = torch.cumsum(dh, dim=-2)

        h = self.layerMHs(subg_interacts, {ntype[0]: conh })

        res = list()
        for i in torch.arange(self.num_heads):
            ds = self.chain(subg_chain, h[ntype[0]][:,i,:], radius)
            s = torch.cumsum(ds, dim=-2)
            res.append(s)
        res = torch.stack(res, dim=1)
        return res

class encoder_bead(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(encoder_bead, self).__init__()
        self.layer1 = dgl.nn.GraphConv( in_dim, hidden_dim, 
                                        norm='right', weight=True, 
                                        allow_zero_in_degree=True)
        self.layer2 = dgl.nn.GraphConv( hidden_dim, out_dim, 
                                        norm='right', weight=True, 
                                        allow_zero_in_degree=True)
        self.layer3 = dgl.nn.GraphConv( out_dim, out_dim, 
                                        norm='right', weight=True, 
                                        allow_zero_in_degree=True)
        self.norm = dgl.nn.EdgeWeightNorm(norm='both')

    def forward(self, blocks, x, etypes, efeat):
        edge_weights = [sub.edata[efeat[0]] for  sub in blocks]
        norm_edge_weights = [ self.norm(blocks[i], w) for i, w in enumerate(edge_weights)]
        
        # block = blocks[0].edge_type_subgraph([etypes[0]])
        # block = dgl.to_homogeneous(block)
        # block = dgl.to_block(block)
        block = blocks[0]
        h = self.layer1(block, x, edge_weight=norm_edge_weights[0])

        # block = blocks[1].edge_type_subgraph([etypes[0]])
        # block = dgl.to_homogeneous(block)
        # block = dgl.to_block(block)
        block = blocks[1]
        h = self.layer2(block, h, edge_weight=norm_edge_weights[1])

        # block = blocks[2].edge_type_subgraph([etypes[0]])
        # block = dgl.to_homogeneous(block)
        # lock = dgl.to_block(block)
        block = blocks[2]
        res = self.layer3(block, h, edge_weight=norm_edge_weights[2])

        return res

class encoder_union(torch.nn.Module):
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
        return res

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
        # x = lambda a : (a-1)*2+5
        # num_seq = x(x(num_clusters))
        # print('num_cluster: {}, mun_seq: {}'.format(num_clusters, num_seq))

        # num_seq = ((2*num_clusters+3)*2+3)+2
        # self.pi = torch.acos(torch.zeros(1)) * 2
        # self.register_buffer('pi_const', self.pi)

        self.w = torch.nn.Parameter(torch.empty( (self.num_heads)), requires_grad=True)
        self.register_parameter('w', self.w)
        torch.nn.init.uniform_(self.w, a=0.0, b=1.0)

        self.r_dist = torch.nn.Parameter(torch.empty((1,num_seq)), requires_grad=True)
        self.register_parameter('r_dist',self.r_dist) 
        torch.nn.init.uniform_(self.r_dist, a=0.3, b=0.5)
        '''self.mean_dist = torch.nn.Parameter(torch.cumsum(torch.abs(self.r_dist)+1e-4, dim=1))
        self.register_parameter('mean_dist',self.mean_dist)''' 

        self.b = torch.nn.Parameter(torch.empty((1)), requires_grad=True)
        self.register_parameter('b',self.b) 
        torch.nn.init.uniform_(self.b, a=1.0e-7, b=1.0e-4)

        # self.std_dist = torch.nn.Parameter(torch.empty((1,num_seq)), requires_grad=True)
        # self.register_parameter('std_dist',self.std_dist) 
        # torch.nn.init.uniform_(self.std_dist, a=0.0, b=1.0)

        '''self.conv1d_dist_0 = torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=7, stride=1, padding=0, padding_mode='replicate', bias=True)
        self.conv1d_dist_1 = torch.nn.Conv1d(in_channels=8, out_channels=1, kernel_size=5, stride=2, padding=0)
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_normal_(self.conv1d_dist_0.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.conv1d_dist_1.weight, gain=gain)

        self.avgP1d_5_2 = torch.nn.AvgPool1d(5, stride=2)
        self.maxP1d_5_2 = torch.nn.MaxPool1d(5, stride=2)'''

    def norm_prob(self, mean, std, x):
        f = (-1.0/2.0)*(((x-mean)/std)**2)
        #log norm probability
        # pdf = -torch.log(std*torch.sqrt(2*self.pi)) + f
        pdf = 1/(std*torch.sqrt(2*self.pi_const)) * torch.exp(f)
        # pdf = torch.nn.functional.normalize( pdf, p=2, dim=-1)
        return pdf

    def aritficial_fc(self, mean, x):
        m = torch.relu(mean)
        score = 1.0/((x - m)**2 + self.b)
        return score

    def edge_distance(self, edges):
        n2 = torch.norm((edges.dst['z'] - edges.src['z']), dim=-1)
        weight = torch.nn.functional.softmax(self.w, dim=0)
        dist = torch.sum(n2*weight, dim=-1, keepdim=True)

        # std = torch.min(torch.abs(self.std_dist), torch.abs(self.r_dist)/4)
        # outputs_dist = self.norm_prob(self.mean_dist, std, dist)
        mean_dist = torch.cumsum(torch.abs(self.r_dist+1e-4)+1e-4, dim=1)
        outputs_dist = self.aritficial_fc(mean_dist, dist)
        # outputs_dist = torch.unsqueeze(outputs_dist, dim=1)
        # outputs_dist = self.conv1d_dist_0(outputs_dist)
        # outputs_dist = self.conv1d_dist_1(outputs_dist)
        # outputs_dist = self.avgP1d_5_2(outputs_dist)
        # outputs_dist = self.maxP1d_5_2(outputs_dist)
        # outputs_dist = torch.squeeze(outputs_dist, dim=1)

        # return {'dist_pred': outputs_dist, 'count_pred':outputs_count}
        return {'dist_pred': outputs_dist}

    def forward(self, g, h):
        with g.local_scope():
            g.nodes[self.ntype].data['z'] = h
            g.apply_edges(self.edge_distance, etype=self.etype)
            # return g.edata.pop('dist_pred'), g.edata.pop('count_pred'), self.mean_dist, self.mean_count
            return g.edata.pop('dist_pred')

