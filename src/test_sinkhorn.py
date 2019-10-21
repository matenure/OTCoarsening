#!/usr/bin/env python

from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

import Sinkhorn as spc
from torch.utils.data import DataLoader

from numpy import random

n = 1000
m = 500
N = [n,m] # Number of points per cloud

# Dimension of the cloud : 2
# x = random.rand(2,N[0])-.5
# theta = 2*np.pi*random.rand(1,N[1])
# r = .8 + .2*random.rand(1,N[1])
# y = np.vstack((np.cos(theta)*r,np.sin(theta)*r))
# plotp = lambda x,col: plt.scatter(x[0,:], x[1,:], s=50, edgecolors="k", c=col, linewidths=1)

# Uniform measure
mu = np.ones(n)/n
nu = np.ones(m)/m
mu = Variable(torch.from_numpy(mu).float(), requires_grad=False)
nu = Variable(torch.from_numpy(nu).float(), requires_grad=False)


# Sinkhorn parameters
epsilon = 0.01
niter = 100

# Wrap with torch tensors
# X = torch.FloatTensor(x.T)
# Y = torch.FloatTensor(y.T)
X = torch.rand((n,3))
Y = torch.randn((m,3))

l1 = spc.sinkhorn_loss_default(X,Y,epsilon,niter=niter)
l2 = spc.sinkhorn_normalized(X,Y,epsilon, mu, nu, n, m, 2, niter)

print("Sinkhorn loss : ", l1.data)
print("Sinkhorn loss (normalized) : ", l2.data)


X = torch.rand((3,n))
Y = torch.rand((3,2))

W = Variable(torch.rand((n,2)),requires_grad=True)
print(W)
XX = torch.matmul(X, W)

opt = spc.sinkhorn_loss_default(XX,Y,epsilon,niter=niter)
print(opt.requires_grad)
opt.backward()
print(W.grad)

