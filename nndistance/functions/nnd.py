# functions/add.py
import torch
from torch.autograd import Function
from _ext import my_lib


class NNDFunction(Function):
    def forward(self, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()   
        self.xyz1 = xyz1
        self.xyz2 = xyz2
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)
        
        self.idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        self.idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)
        
        if not xyz1.is_cuda:
            my_lib.nnd_forward(xyz1, xyz2, dist1, dist2, self.idx1, self.idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            self.idx1 = self.idx1.cuda()
            self.idx2 = self.idx2.cuda()
            my_lib.nnd_forward_cuda(xyz1, xyz2, dist1, dist2, self.idx1, self.idx2)
            
        self.dist1 = dist1
        self.dist2 = dist2
        
        #print(batchsize, n, m)

        return dist1, dist2

    def backward(self, graddist1, graddist2):
        #print(self.idx1, self.idx2)


        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(self.xyz1.size())
        gradxyz2 = torch.zeros(self.xyz2.size())
        
        if not graddist1.is_cuda:
            my_lib.nnd_backward(self.xyz1, self.xyz2, gradxyz1, gradxyz2, graddist1, graddist2, self.idx1, self.idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            my_lib.nnd_backward_cuda(self.xyz1, self.xyz2, gradxyz1, gradxyz2, graddist1, graddist2, self.idx1, self.idx2)
            
        return gradxyz1, gradxyz2