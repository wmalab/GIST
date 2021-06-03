import dgl
import torch
import os

class HiCDataset(dgl.data.DGLDataset):
    def __init__(self, graphs_dict, features_dict, train, validation, test, path=None, name=None):
        self.g_dict = graphs_dict
        self.f_dict = features_dict
        self.train = train
        self.valid = validation
        self.test = test
        if (path is not None) and (name is not None):
            save_dir = os.path.join(path, name)
        else:
            save_dir=None
        super(HiCDataset, self).__init__(name='Hi-C_dgl_dataset', save_dir=save_dir)


    def process(self):
        self.graphs = []
        self.features = []
        self.labels = []
        self.train_list = []
        self.valid_list = []
        self.test_list = []
        for i, (key, gs) in enumerate(self.g_dict.items()):
            self.graphs.append(gs)
            self.features.append(self.f_dict[key])
            self.labels.append(key)
            if(key in self.train):
                self.train_list.append(i)
            if(key in self.valid):
                self.valid_list.append(i)
            if(key in self.test):
                self.test_list.append(i)


    def __getitem__(self, i):
        return self.graphs[i], self.features[i], self.labels[i]

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