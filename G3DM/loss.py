import torch
import torch.nn as nn
# from geomloss import SamplesLoss
# import torch.nn.functional as F

class ClusterWassersteinLoss(nn.Module):
    def __init__(self, num_cluster):
        super(ClusterWassersteinLoss, self).__init__()
        # self.action = torch.nn.LeakyReLU(0.3)
        self.num_cluster = num_cluster
        

    def forward(self, pred, target, balance_weight, weight=None):
        np = torch.nn.functional.normalize(torch.exp(pred), p=1, dim=-1)
        pred_cdf = torch.cumsum(np, dim=-1)
        target_cdf = torch.cumsum(target, dim=-1)

        diff = pred_cdf - target_cdf
        res = (torch.abs(diff)**2).mean(dim=0)
        bw = balance_weight/balance_weight.mean() + 1
        if weight is None:
            w = torch.ones((np.shape[1]), device=np.device)
        else:
            w = weight # *torch.sqrt(cw/cw.mean()+1.0) 
        w = bw.view(1,-1)*w.view(1,-1)
        w = torch.nn.functional.normalize(w.view(1,-1), p=1)
        res = res.view(1,-1)*w.view(1,-1)
        res = res.sum(dim=-1)
        return res


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return torch.sqrt( self.mse(torch.log1p(pred), torch.log1p(target)) )

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
    
    def forward(self, pred, target, balance_weight, weight=None):
        logp = pred # torch.nn.functional.log_softmax(pred, 1)
        # logp = torch.nn.functional.log_softmax(p, 1)
        bw = balance_weight/balance_weight.mean() + 1
        if weight is  None:
            w = torch.ones_like(bw, device=bw.device) # * (weights/weights.mean() + 10.0) # torch.sqrt(weights/weights.mean() + 1.0)
        else:
            w = weight 
        w = bw.view(1,-1)*w.view(1,-1)
        w = torch.nn.functional.normalize(w.view(1,-1), p=1)
        loss = torch.nn.functional.nll_loss(logp, target.long(), weight=w.double(), reduce=True, reduction='mean')
        
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
