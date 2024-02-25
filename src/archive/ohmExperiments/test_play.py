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
from wos_mnist import WOSMnist

device = 'cpu'

##  0.03999999910593033
##############################################################
## Basic Training Program
if __name__ == '__main__':

    #error   = pickle.load(open("mymnist_error.pkl", 'rb'))
    #minv = np.min(error)
    #maxv = np.max(error)
    #print(f'Min error {minv} and {maxv}') 
    #stats = pickle.load(open("stats_8bit_mymnist_6580.pkl", 'rb'))
    #stats = pickle.load(open("stats_12bit_mymnist_6580.pkl", 'rb'))
    stats = pickle.load(open("stats_16bit_mymnist_6580.pkl", 'rb'))
    l2mean = 0.0
    l3mean = 0.0
    l4mean = 0.0
    for i in range(len(stats)): 
        stat = stats[i]
        if i == 0:
            mv = torch.mean(stat)
            print("Layer 0: " + str(mv.numpy())) 
        if i == 1:
            mv = torch.mean(stat)
            print("Layer 1: " + str(mv.numpy())) 
        if (i >= 2) and (i <= 9):
            mv = torch.mean(stat)
            l2mean = l2mean + mv
        if (i==11) or (i==13) or (i==15) or (i==17) or (i==19) or (i==21) or (i==23) or (i==25):
            mv = torch.mean(stat)
            l3mean = l3mean + mv
        if (i==12) or (i==14) or (i==16) or (i==18) or (i==20) or (i==22) or (i==24) or (i==26):
            mv = torch.mean(stat)
            l4mean = l4mean + mv
        if i == 27: 
            mv = torch.mean(stat)
            print("Layer 5: " + str(mv.numpy()))
        if i == 28:
            mv = torch.mean(stat)
            print("Layer 6: " + str(mv.numpy()))

    l2mean = l2mean / 8.0
    print("Layer 2: " + str(l2mean.numpy()))

    l3mean = l3mean / 8.0
    print("Layer 3: " + str(l3mean.numpy()))

    l4mean = l4mean / 8.0
    print("Layer 4: " + str(l4mean.numpy()))


    