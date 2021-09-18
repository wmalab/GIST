import torch
import torch.nn as nn
# from geomloss import SamplesLoss
# import torch.nn.functional as F

class WassersteinLoss(nn.Module):
    def __init__(self, num_cluster):
        super(WassersteinLoss, self).__init__()
        # self.action = torch.nn.LeakyReLU(0.3)
        self.num_cluster = num_cluster
        

    def forward(self, pred, target, weight=None):
        np = torch.nn.functional.normalize(torch.exp(pred), p=1, dim=-1)
        pred_cdf = torch.cumsum(np, dim=-1)
        target_cdf = torch.cumsum(target, dim=-1)
        ncluster = np.shape[-1]
        diff = torch.abs(pred_cdf - target_cdf)
        res = diff[~torch.isnan(diff)].mean(dim=0)
        res = (res*ncluster)**2
        if weight is None:
            w = torch.ones((np.shape[1]), device=np.device)
        else:
            w = weight
        w = torch.nn.functional.normalize(w.view(1,-1), p=1)
        res = res.view(1,-1)*w.view(1,-1)
        res = res.sum(dim=-1)
        return res

class ClusterLoss(nn.Module):
    def __init__(self, num_cluster):
        super(ClusterLoss, self).__init__()
        # self.action = torch.nn.LeakyReLU(0.3)
        self.num_cluster = num_cluster
        

    def forward(self, pred, target, weight=None):
        np = torch.nn.functional.normalize(torch.exp(pred), p=1, dim=-1)
        ncluster = np.shape[-1]
        diff = torch.abs(pred - target)
        res = diff.mean(dim=0)
        if weight is None:
            w = torch.ones((np.shape[1]), device=np.device)
        else:
            w = weight
        w = torch.nn.functional.normalize(w.view(1,-1), p=1)
        res = res.view(1,-1)*w.view(1,-1)
        res = res.sum(dim=-1)
        return res

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return torch.sqrt( self.mse(torch.log1p(pred+1), torch.log1p(target+1)) )

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred.float(), target.float())).float()

class stdLoss(nn.Module):
    def __init__(self):
        super(stdLoss, self).__init__()
        
    def forward(self, std, cluster, num_cluster):
        cutoff = torch.relu( 2*torch.ones_like(cluster)- cluster )
        weight = cutoff
        mask = (cluster!=0)
        res = torch.sum(std*weight.view(-1,))/(1+mask.sum())
        # res = torch.mean(std)
        return res

class nllLoss(torch.nn.Module):
    def __init__(self):
        super(nllLoss, self).__init__()
    
    def forward(self, pred, target, weight=None):
        logp = pred 
        if weight is  None:
            w = torch.ones((pred.shape[-1]), device=pred.device)
        else:
            w = weight 
        w = torch.nn.functional.normalize(w.view(1,-1), p=1)
        loss = torch.nn.functional.nll_loss(logp, target.long(), weight=w.float(), reduce=True, reduction='mean')
        
        return loss

# class crossNllLoss(nn.Module):
#     def __init__(self):
#         super(crossNllLoss, self).__init__()
    
#     def forward(self, pred, target):
#         class_p = torch.argmax(pred, dim=-1, keepdim=False)
#         class_t = torch.argmax(target, dim=-1, keepdim=False)

#         logp = nn.log_softmax(pred, 1)
#         logt = nn.log_softmax(target, 1)
#         loss_pt = nn.nll_loss(logp, class_t.long(), reduce=True, reduction='mean')
#         loss_pp = nn.nll_loss(logp, class_p.long(),  reduce=True, reduction='mean')
#         loss_tt = nn.nll_loss(logt, class_t.long(), reduce=True, reduction='mean')
#         loss = loss_pt + loss_pp + loss_tt
#         return loss

# class WassersteinLoss(nn.Module):
#     def __init__(self, device):
#         super(WassersteinLoss, self).__init__()
#         loss = "sinkhorn"
#         p = 1
#         blur = 0.01
#         self.loss_fc = SamplesLoss(loss, p=p, blur=blur)
#         self.device = device

#     def forward(self, pred, target, num_cluster):
#         self.ncluster = torch.arange(0, num_cluster, dtype= torch.float, device=self.device,requires_grad=False)

#         p = torch.relu(pred)
#         p = torch.nn.functional.normalize(p)
#         p = torch.sum(p * self.ncluster.view(1, -1), dim=-1, keepdim=True)
#         res = self.loss_fc(p.float(), target.view(-1,1).float())
#         return res
