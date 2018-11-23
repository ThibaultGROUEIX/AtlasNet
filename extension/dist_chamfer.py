import math
from torch import nn
from torch.autograd import Function
import torch
import sys
from numbers import Number
from collections import Set, Mapping, deque
import chamfer


# Chamfer's distance module @thibaultgroueix
# GPU tensors only
# remember to call chamferFunction.clean() after loss.backward() to avoid memory leak

class chamferFunction(Function):
    def __enter__(self):
        """ """

    def __exit__(self, exc_type, exc_value, traceback):
        del self.xyz1
        del self.xyz2
        del self.idx1
        del self.idx2
        del self.dist1
        del self.dist2

    def clean(self):
        # print('Destructor called, vehicle deleted.')
        del self.xyz1
        del self.xyz2
        del self.idx1
        del self.idx2
        del self.dist1
        del self.dist2

    def forward(self, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        self.xyz1 = xyz1
        self.xyz2 = xyz2
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        self.idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        self.idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        self.idx1 = self.idx1.cuda()
        self.idx2 = self.idx2.cuda()
        chamfer.forward(xyz1, xyz2, dist1, dist2, self.idx1, self.idx2)

        self.dist1 = dist1
        self.dist2 = dist2
        return dist1, dist2

    def backward(self, graddist1, graddist2):
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(self.xyz1.size())
        gradxyz2 = torch.zeros(self.xyz2.size())

        gradxyz1 = gradxyz1.cuda()
        gradxyz2 = gradxyz2.cuda()
        chamfer.backward(self.xyz1, self.xyz2, gradxyz1, gradxyz2, graddist1, graddist2, self.idx1, self.idx2)
        return gradxyz1, gradxyz2

class chamferDist(nn.Module):
    def __init__(self):
        super(chamferDist, self).__init__()
        self.cham = chamferFunction()

    def forward(self, input1, input2):
        self.cham = chamferFunction()
        return self.cham(input1, input2)

