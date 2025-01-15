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

class WOSMnist(nn.Module):
    def __init__(self):
        super(WOSMnist, self).__init__()
        #print(device)
        self.conv1 = WOS(1, 16, 3)
        self.conv2 = WOS(16, 16, 3)
        self.fc1 = WOS(2304, 16, 1)
        self.fc2 = WOS(16, 1, 1)

        #self.conv1 = nn.Conv2d(1, 32, 3, 1)
        #self.conv2 = nn.Conv2d(32, 64, 3, 1)
        #self.fc1 = nn.Conv2d(9216, 128, 1, 1)
        #self.fc2 = nn.Conv2d(128, 10, 1, 1)

        #self.fc1 = nn.Linear(9216, 128)
        #self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #print("FWD")
        #print(x.shape)        
        x = self.conv1(x)        
        #print(x.shape)
        #x = torch.tanh(x)                        
        x = self.conv2(x)
        #print(x.shape)
        #x = torch.tanh(x)
        x = F.max_pool2d(x, 2)        
        x = torch.flatten(x, 1)        
        x = x.unsqueeze(-1)        
        x = x.unsqueeze(-1)
        #print(x.shape)        
        x = self.fc1(x)
        #print(x.shape)
        #x = torch.tanh(x)        
        x = self.fc2(x)        
        #print(output.shape)
        output = x.squeeze()
        output = output.unsqueeze(0) 
        #if len(output.shape) == 1:
        #    output = output.unsqueeze(0) 
        
        return output
