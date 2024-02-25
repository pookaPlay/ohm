import progressbar
import sys
import glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import pdb
import skimage.io as skio
from scipy.signal import medfilt as med_filt
import math
import random
import skimage.transform
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from scipy.ndimage.measurements import label
from wos import ClipNegatives
from wos import WOS
import torch.nn as nn

class WosNet(nn.Module):
    def __init__(self):
        super(WosNet, self).__init__()
        #print(device)
        self.conv1 = WOS(1, 32, 3)
        self.conv2 = WOS(32, 64, 3)
        self.fc1 = WOS(9216, 32, 1)
        self.fc2 = WOS(32, 10, 1)

        #self.conv1 = nn.Conv2d(1, 32, 3, 1)
        #self.conv2 = nn.Conv2d(32, 64, 3, 1)
        #self.fc1 = nn.Conv2d(9216, 128, 1, 1)
        #self.fc2 = nn.Conv2d(128, 10, 1, 1)

        #self.fc1 = nn.Linear(9216, 128)
        #self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #print("FWD")
        #print(x.shape)
        self.conv1 = self.conv1.to(device)
        x = self.conv1(x)        
        print(x.shape)
        #x = torch.tanh(x)        
        self.conv2 = self.conv2.to(device)
        
        x = self.conv2(x)
        print(x.shape)
        #x = torch.tanh(x)
        x = F.max_pool2d(x, 2)        
        x = torch.flatten(x, 1)        
        x = x.unsqueeze(-1)        
        x = x.unsqueeze(-1)
        print(x.shape)
        self.fc1 = self.fc1.to(device)
        x = self.fc1(x)
        print(x.shape)
        #x = torch.tanh(x)
        self.fc2 = self.fc2.to(device)        
        x = self.fc2(x)        
        print(x.shape)
        x = x.squeeze()        
        output = F.log_softmax(x, dim=1)
        return output

