import torch
import torch.nn as nn
# from geomloss import SamplesLoss
# import torch.nn.functional as F

class ClusterWassersteinLoss(nn.Module):
    def __init__(self, num_cluster):
        super(ClusterWassersteinLoss, self).__init__()
        # self.action = torch.nn.LeakyReLU(0.3)
        self.num_cluster = num_cluster
        

    def forward(self, pred, target):
        np = torch.nn.functional.normalize(torch.exp(pred), p=1, dim=-1)
        pred_cdf = torch.cumsum(np, dim=-1)
        target_cdf = torch.cumsum(target, dim=-1)
        
        weight = torch.range(start=self.num_cluster, end=1, step=-1, device=target.device)
        diff = pred_cdf - target_cdf

        up_res = torch.relu(diff).mean(dim=0)
        down_res = torch.relu(-diff).mean(dim=0)
        up_ratio = torch.div(up_res+1, down_res+1)
        down_ratio = torch.div(down_res+1, up_res+1)

        res = up_res*up_ratio + down_res*down_ratio
        res = (res * weight).sum(dim=-1)
        res = torch/(self.num_cluster-1)
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
        weight = torch.relu( torch.abs(cluster - num_cluster/2) - (num_cluster/4) )
        res = torch.sum(std*weight.view(-1,))/std.shape[0]
        return res

class nllLoss(torch.nn.Module):
    def __init__(self):
        super(nllLoss, self).__init__()
    
    def forward(self, pred, target, weights=None):
        logp = pred # torch.nn.functional.log_softmax(pred, 1)
        if weights is not None:
            w = torch.square(weights/weights.mean() + 1.0) # torch.sqrt(weights/weights.mean() + 1.0)
            loss = torch.nn.functional.nll_loss(logp, target.long(), weight=w.float(), reduce=True, reduction='mean')
        else:
            loss = torch.nn.functional.nll_loss(logp, target.long(), reduce=True, reduction='mean')
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
