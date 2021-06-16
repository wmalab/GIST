import torch
import torch.nn as nn
# import torch.nn.functional as F

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        return torch.sqrt(self.mse(torch.log1p(pred), torch.log1p(target)))

class nllLoss(torch.nn.Module):
    def __init__(self):
        super(nllLoss, self).__init__()
    
    def forward(self, pred, target, weights=None):
        logp = torch.nn.functional.log_softmax(pred, 1)
        if weights is not None:
            w = weights/weights.mean()
            print(w)
            loss = torch.nn.functional.nll_loss(logp, target.long(), weight=w.float(), reduce=True, reduction='mean')
        else:
            loss = torch.nn.functional.nll_loss(logp, target.long(), reduce=True, reduction='mean')
        return loss

class crossNllLoss(nn.Module):
    def __init__(self):
        super(crossNllLoss, self).__init__()
    
    def forward(self, pred, target):
        class_p = torch.argmax(pred, dim=-1, keepdim=False)
        class_t = torch.argmax(target, dim=-1, keepdim=False)

        logp = nn.log_softmax(pred, 1)
        logt = nn.log_softmax(target, 1)
        loss_pt = nn.nll_loss(logp, class_t.long(), reduce=True, reduction='mean')
        loss_pp = nn.nll_loss(logp, class_p.long(),  reduce=True, reduction='mean')
        loss_tt = nn.nll_loss(logt, class_t.long(), reduce=True, reduction='mean')
        loss = loss_pt + loss_pp + loss_tt
        return loss