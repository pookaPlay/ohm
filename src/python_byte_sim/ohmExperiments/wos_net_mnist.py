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


class WOSNetMnist(nn.Module):
    def __init__(self, NET_WIDTH):
        super(WOSNetMnist, self).__init__()
        #print(device)
        self.out_size = NET_WIDTH
        self.conv1 = WOS(1, self.out_size, 3)
        self.linc1 = WOS(self.out_size, self.out_size, 1)
        self.conv2set = nn.ModuleList()        
        for i in range(self.out_size):
            self.conv2set.append(WOS(1, 1, 3))            
        # max pool
        self.linc2 = WOS(self.out_size, self.out_size, 1)
        self.conv3set = nn.ModuleList()        
        for i in range(self.out_size):
            self.conv3set.append(WOS(1, 1, 3))            
        self.conv4set = nn.ModuleList()        
        for i in range(self.out_size):
            self.conv4set.append(WOS(1, 1, 3))            

        # max pool
        self.linc3 = WOS(self.out_size, self.out_size, 1)
        self.linc4 = WOS(self.out_size, 1, 1)
        # max pool

    def forward(self, x):
        print("FWD")
        #print("INPUT")
        #print(x.shape)        
        x = self.conv1(x)        
        x = self.linc1(x)
        #print(x.shape)        
        y = torch.zeros((x.shape[0], x.shape[1], x.shape[2]-2, x.shape[3]-2))
        for i in range(self.out_size):
            #print("CONV 2 : " + str(i)) 
            xx = x[:,i,:,:].unsqueeze(1)
            #print(xx.shape)
            temp = self.conv2set[i](xx)
            #print(temp.shape)
            y[:,i,:,:] = temp.squeeze()
        print(y.shape)
        x = F.max_pool2d(y, 2)
        # 8 x 12 x 12
        x = self.linc2(x)
        print(x.shape)
        y = torch.zeros((x.shape[0], x.shape[1], x.shape[2]-4, x.shape[3]-4))
        for i in range(self.out_size):
            #print("CONV 3 : " + str(i)) 
            xx = x[:,i,:,:].unsqueeze(1)
            #print(xx.shape)
            temp = self.conv3set[i](xx)
            #print(x.shape)
            #print("CONV 4 : " + str(i)) 
            temp = self.conv4set[i](temp)
            #print(temp.shape)
            y[:,i,:,:] = temp.squeeze()
        # 8 x 8 x 8        
        #print("OUTPUT")
        #print(y.shape)
        x = F.max_pool2d(y, 2)
        #print(x.shape)
        # 8 x 4 x 4
        x = self.linc3(x)
        # 8 x 4 x 4
        x = self.linc4(x)
        output = F.max_pool2d(x, 4)
        #print(output.shape)                
        return output.squeeze()


    def MyShapes(self):
        
        print("INPUT")
        self.conv1.MyShapes()
        self.linc1.MyShapes()
        print("CONV 2: " + str(len(self.conv2set)))                        
        self.conv2set[0].MyShapes()
        self.linc2.MyShapes()
        print("CONV 3: " + str(len(self.conv3set)))        
        self.conv3set[0].MyShapes()
        print("CONV 4: " + str(len(self.conv4set)))        
        self.conv4set[0].MyShapes()
        print("OUTPUT")
        self.linc3.MyShapes()
        self.linc4.MyShapes()

    def MyPrint(self):
        
        self.conv1.MyPrint()
        self.linc1.MyPrint()
        #
        for i in range(len(self.conv2set)):    #self.out_size):
            self.conv2set[i].MyPrint()
        # max pool
        self.linc2.MyPrint()
        for i in range(len(self.conv3set)):    #self.out_size):
            self.conv3set[i].MyPrint()
        #self.conv4set = nn.ModuleList()        
        for i in range(len(self.conv4set)):    #self.out_size):
            self.conv4set[i].MyPrint()

        self.linc3.MyPrint()
        self.linc4.MyPrint()

    def FindRanks(self, x):
                
        stats = []

        y = self.conv1(x)    
        print(x.shape)    
        print("--- results")
        print(y.shape)    
        stat = self.conv1.FindRanks(x, y)
        stats.append(stat)

        x = self.linc1(y)
        stat = self.linc1.FindRanks(y, x)
        stats.append(stat)        

        #print(x.shape)        
        y = torch.zeros((x.shape[0], x.shape[1], x.shape[2]-2, x.shape[3]-2))
        for i in range(len(self.conv2set)):    #self.out_size):
            
            xx = x[:,i,:,:].unsqueeze(1)
            #print(xx.shape)
            temp = self.conv2set[i](xx)
            y[:,i,:,:] = temp.squeeze()
            print("  Conv 2: " + str(i) + " finding precision")
            stat = self.conv2set[i].FindRanks(xx, temp)
            stats.append(stat)
            #print(temp.shape)
            
        #print(y.shape)
        x = F.max_pool2d(y, 2)
        # 8 x 12 x 12
        xy = self.linc2(x)
        stat = self.linc2.FindRanks(x, xy)
        stats.append(stat)        

        #print(x.shape)
        y = torch.zeros((x.shape[0], x.shape[1], x.shape[2]-4, x.shape[3]-4))
        for i in range(len(self.conv3set)):    #self.out_size):
            #print(i)
            xx = x[:,i,:,:].unsqueeze(1)
            #print(xx.shape)
            temp = self.conv3set[i](xx)
            stat = self.conv3set[i].FindRanks(xx, temp)
            stats.append(stat)

            #print(x.shape)
            temp2 = self.conv4set[i](temp)
            stat = self.conv4set[i].FindRanks(temp, temp2)
            stats.append(stat)

            #print(temp.shape)
            y[:,i,:,:] = temp2.squeeze()
        # 8 x 8 x 8        
        #print(y.shape)
        x = F.max_pool2d(y, 2)
        #print(x.shape)
        # 8 x 4 x 4
        y = self.linc3(x)
        stat = self.linc3.FindRanks(x, y)
        stats.append(stat)        

        # 8 x 4 x 4
        x = self.linc4(y)
        stat = self.linc4.FindRanks(y, x)
        stats.append(stat)        

        output = F.max_pool2d(x, 4)
        #print(output.shape)                
        return stats
