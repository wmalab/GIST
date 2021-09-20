import dgl
import torch
import os

class HiCDataset(dgl.data.DGLDataset):
    def __init__(self, graphs_dict, features_dict, cluster_weight_dict, path=None, name=None):
        self.g_dict = graphs_dict
        self.f_dict = features_dict
        self.cw_dict = cluster_weight_dict
        # self.train = train
        # self.test = test
        if (path is not None) and (name is not None):
            save_dir = os.path.join(path, name)
        else:
            save_dir=None
        super(HiCDataset, self).__init__(name='Hi-C_dgl_dataset', save_dir=save_dir)


    def process(self):
        self.graphs = []
        self.features = []
        self.cw = []
        self.index = []
        # self.label = []
        self.list = []
        count = 0
        for i, (key, gs_list) in enumerate(self.g_dict.items()):
            feature = self.f_dict[key]
            for j, (idx, gs) in enumerate(gs_list.items()):
                nodes = gs['top_graph'].ndata['id']
                feat = feature['feat'][nodes, :]
                pos = feature['pos'][nodes, :]
                self.features.append({'feat':feat, 'pos':pos})
                self.graphs.append(gs)
                self.index.append('{}_{}'.format(key, idx))
                self.cw.append(self.cw_dict[key])
                self.list.append(count)
                # if(key in self.test):
                #     self.test_list.append(count)
                #     self.label.append('test')
                count = count+1


    def __getitem__(self, i):
        return self.graphs[i], self.features[i], self.cw[i], self.index[i] # self.label[i], 

    def __len__(self):
        return len(self.graphs)
    
    def save(self):
        torch.save(self, self.save_dir)

'''
    HD = HiCDataset(graph_dict, train_list, valid_list, test_list, os.path.join( root, 'data', cell, hyper), 'dataset.pt')
    # HD.save()
    torch.save(HD, os.path.join( root, 'data', cell, hyper, 'dataset.pt'))
    load_HD = torch.load(os.path.join( root, 'data', cell, hyper, 'dataset.pt'))
'''