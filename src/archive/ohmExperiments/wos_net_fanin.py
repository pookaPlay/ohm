import torch
import torch.nn as nn
from wos import WOS
from wos import ClipNegatives
from wos import ConvertToInteger
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

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


class WOSNetFanin(nn.Module):
    def __init__(self, NET_WIDTH = 4, NET_DEPTH = 2):
        super(WOSNetFanin, self).__init__()      
        
        self.first = WOS(2, NET_WIDTH, 1)
        self.depth_path = nn.ModuleList()        

        for i in range(NET_DEPTH):
            self.depth_path.append(WOS(NET_WIDTH, NET_WIDTH, 1))
        
        self.last  = WOS(NET_WIDTH, 1, 1)
        

    def forward(self, x):

        y = self.first(x)
        
        for i, depth in enumerate(self.depth_path):
            x = depth(y)   
            y = x 

        return self.last(x)
    
    def MyPrint(self):
        self.first.MyPrint()
        for i, depth in enumerate(self.depth_path):
            depth.MyPrint()
        self.last.MyPrint()

    def FindRanks(self, x):
        stats = []

        y = self.first(x)
        stat = self.first.FindRanks(x, y)
        stats.append(stat)

        for i, depth in enumerate(self.depth_path):
            x = depth(y)   
            stat = depth.FindRanks(y, x)
            stats.append(stat)
            y = x 

        x = self.last(y)        
        stat = self.last.FindRanks(y, x)
        stats.append(stat)
        
        return stats
