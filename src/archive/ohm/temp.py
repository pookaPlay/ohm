import torch
import torch.nn as nn
#from unet import UNet
from mnet import MNet
from wos import WOS
from linear import Linear
import QuanSynData
import BSDSData as bsds
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pdb
from morphology import Morphology
import math
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd import gradcheck

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'


def SmartSort(x, permutation):
    d1 = x.shape[0]
    d2 = x.shape[1]
    
    ret = x[
        torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
        permutation.flatten()
    ].view(d1, d2)
    return ret

##############################################################
## Basic Training Program
if __name__ == '__main__':

    torch.manual_seed(0)
    dim = 3

    
    input = torch.Tensor([[2, 3, 5], [4, 7, 1]])
    #input = input.unsqueeze(1)
    #input = input.unsqueeze(0)
    print("Input")    
    print(input)
    
    weight = nn.Parameter(torch.Tensor(1, dim), requires_grad=True)
    nn.init.ones_(weight)
    #nn.init.xavier_uniform_(weight)
    #nn.init.constant_(self.bias, 1.0)
    mask = nn.Parameter(torch.Tensor(1, dim), requires_grad=True)
    nn.init.zeros_(mask)
    #nn.init.uniform_(self.mask, -1.0, 1.0)
    
    N = input.shape[0]
    D = input.shape[1]
    #weight = weight.unsqueeze(0)
    #mask = mask.unsqueeze(0)
        
    x = mask + input
    mx = torch.cat((x, -x), 1)

    mw = torch.cat((weight, weight), 1)    
    allw = mw.repeat(N, 1)

    smx, si = torch.sort(mx, 1, descending=True)
    sw = SmartSort(allw, si)

    accw = torch.cumsum(sw, 1)
    print("Cummulative")
    print(accw)

    li = torch.sum(torch.le(accw, D), 1, keepdim=True) - 1
    y = torch.gather(smx, 1, li)
    print(li)
    print(smx)
    print(y)
    #x = torch.sum(x, 2)
    
    print("DONE")
