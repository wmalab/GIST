import numpy as np
import torch
from torch import nn
from torch import optim
import torch.distributions as D


weights = torch.ones(8,requires_grad=True)
means = torch.tensor(np.random.randn(8,),requires_grad=True)
stdevs = torch.tensor(np.abs(np.random.randn(8,)),requires_grad=True)

parameters = [weights, means, stdevs]
optimizer = optim.SGD(parameters, lr=0.001, momentum=0.9)

num_iter = 10001
for i in range(num_iter):
    mix = D.Categorical(weights)
    comp = D.Normal(means, stdevs)
    gmm = D.MixtureSameFamily(mix, comp)

    optimizer.zero_grad()
    x = torch.randn(5000,1)#this can be an arbitrary x samples
    loss = -gmm.log_prob(x).mean()#-densityflow.log_prob(inputs=x).mean()
    loss.backward()
    optimizer.step()

    print(i, loss, gmm.mean, gmm.log_prob(x).shape)
