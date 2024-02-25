import torch
import torch.nn as nn
from wos import WOS
from wos import ClipNegatives
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pdb
#from morphology import Morphology
import math
import pickle
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd import gradcheck
from torch.optim.lr_scheduler import StepLR
import SynData
import random
from FindMinWOS import FindIntWOS
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


class ConvertWOS(object):

    def __init__(self):
        pass

    def __call__(self, module):

        if hasattr(module, 'weight'):
            weight = module.weight.data
            bias = module.bias.data
            for fi in range(weight.shape[0]):
                w = weight[fi].detach().numpy()
                t = bias[fi].detach().numpy() 
                
                #print("WOS...")                
                nw = w[w > 1.0e-6]                                
                #print(nw)
                #print(t)
                WW, TT = FindIntWOS(nw, t)
                w[w > 1.0e-6] = WW
                t = TT
                weight[fi] = torch.tensor(WW)
                bias[fi] = torch.tensor(TT)
                #print("IWOS!!!")
                #print(WW)
                #print(TT)
            print(weight)
            print(bias)
            module.weight.data = weight
            module.weight.bias = TT

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()      
        self.conv1 = WOS(2, 16, 1)        
        self.conv2 = WOS(16, 16, 1)
        self.conv3 = WOS(16, 16, 1)
        self.conv4 = WOS(16, 1, 1)

    def forward(self, x):
        #print("FWD")
        #print(x.shape)
        x = self.conv1(x)        
        x = self.conv2(x)                
        x = self.conv3(x)
        x = self.conv4(x)
        #print(x.shape)
        return x
    
    def MyPrint(self):
        self.conv1.MyPrint()
        self.conv2.MyPrint()
        self.conv3.MyPrint()
        self.conv4.MyPrint()


##############################################################
## Basic Training Program
if __name__ == '__main__':

    theSeed = 0
    np.random.seed(theSeed)
    random.seed(theSeed)
    torch.manual_seed(theSeed)

    numEpochs = 3000    
    numSamples = 100
    learningRate = 1

    (X, Y, XV, YV, XMAP) = SynData.LoadXor(numSamples)

    X = X.unsqueeze(-1)
    X = X.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    XV = XV.unsqueeze(-1)
    XV = XV.unsqueeze(-1)
    YV = YV.unsqueeze(-1)
    YV = YV.unsqueeze(-1)
    XMAP = XMAP.unsqueeze(-1)
    XMAP = XMAP.unsqueeze(-1)
   
    X = X.to(device)
    Y = Y.to(device)       
    XV = XV.to(device)
    YV = YV.to(device)       
    XMAP = XMAP.to(device)
                
    model = Net().to(device)        
    clipper = ClipNegatives()
    converter = ConvertWOS()

    model.load_state_dict(torch.load("trn_4_model_xor.pth"))
    model.MyPrint()
    model.apply(converter)
    model.MyPrint()
    #torch.save(model.state_dict(), "WOS_4_model_xor.pth")
    
    if False:
        model.eval()
        with torch.no_grad():
            vOut = model(XV)
            lossv, errorv = loss_fn(vOut, YV)
            print(f'Apply: Loss: {lossv}  Error: {errorv}')
            YMAP = model(XMAP)
            YMAP = YMAP.squeeze()
            SynData.PlotMap(YMAP)
            print("DONE")
