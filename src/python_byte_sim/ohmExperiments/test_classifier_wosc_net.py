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
from wos_net import WOSNet
from wos_net_fanin import WOSNetFanin

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def HingeLoss(YP, YL):    
    
    loss = 1.0 - torch.mul(YP, Y)
    loss[loss < 0.0] = 0.0
    #werror = torch.mul(hinge_loss, tweights)
    hingeLoss = torch.mean(loss)      
    myP = YP
    myP[myP < 0] = -1.0
    myP[myP >= 0] = 1.0
    same = torch.sum(myP == Y).float()
    error = Y.shape[0] - same
    error = torch.true_divide(error, Y.shape[0])

    return(hingeLoss, error)

def MAE(YP, YL):    

    loss = torch.abs(YP - YL)
    mae = torch.mean(loss)      
    return(mae)

##############################################################
## Basic Training Program
if __name__ == '__main__':

    theSeed = 100  
    numSamples = 100
    
    CONVERT_TO_INT = True
    GET_INT_STATS = False
    GET_FLOAT_STATS = False
    VIZ_2D = False

    NET_WIDTH = 2
    NET_DEPTH = 2
    
    intName = "int_xor_" + str(NET_WIDTH) + "_" + str(NET_DEPTH) + ".pkl"
    statsName = "stats_xor_" + str(NET_WIDTH) + "_" + str(NET_DEPTH) + ".pkl"
    modelLastName = "model_xor_last_" + str(NET_WIDTH) + "_" + str(NET_DEPTH) + ".pth"
    modelLossName = "model_xor_loss_" + str(NET_WIDTH) + "_" + str(NET_DEPTH) + ".pth"
    modelErrorName = "model_xor_error_" + str(NET_WIDTH) + "_" + str(NET_DEPTH) + ".pth"
    #modelLastName = "fanin_xor_last_" + str(NET_WIDTH) + "_" + str(NET_DEPTH) + ".pth"
    #modelLossName = "fanin_xor_loss_" + str(NET_WIDTH) + "_" + str(NET_DEPTH) + ".pth"
    #modelErrorName = "fanin_xor_error_" + str(NET_WIDTH) + "_" + str(NET_DEPTH) + ".pth"
    
    modelName = modelLossName
    #modelName = modelErrorName
    #modelName = modelLastName
    #modelName = "viz2_xor_loss_4_4.2348.pth"

    np.random.seed(theSeed)
    random.seed(theSeed)
    torch.manual_seed(theSeed)

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
            
    #loss_fn = nn.MSELoss()
    #loss_fn = MAE
    loss_fn = HingeLoss

    #model = WOSNetFanin(NET_WIDTH, NET_DEPTH).to(device)        
    model = WOSNet(NET_WIDTH, NET_DEPTH).to(device)        
    clipper = ClipNegatives()
    converter = ConvertToInteger()

    if CONVERT_TO_INT:  
        print("Converting to integer")
        model.load_state_dict(torch.load(modelName))         
        with torch.no_grad():
            model.apply(converter)
            torch.save(model.state_dict(), intName)
            print("==============================================")
            print("INTEGER WOS")        
            model.MyPrint()


    if GET_INT_STATS:  
        #print("Calculating integer stats")
        #model.load_state_dict(torch.load(intName))                        
        #model.MyStats()

        with torch.no_grad():
            stats = model.FindRanks(X)
            with open(statsName, 'wb') as f:
                pickle.dump(stats, f)

            for stat in stats:            
                meanStats = np.mean(stat, 0)
                print(meanStats.squeeze())

    if GET_FLOAT_STATS:  
        #print("Calculating stats for reals")
        model.load_state_dict(torch.load(modelName))                        
        with torch.no_grad():
            stats = model.FindRanks(X)
            #with open(statsName, 'wb') as f:
            #    pickle.dump(stats, f)
            #print(stats)
            #print("##############################")
            #print("And means")
            if True:
                layerStat = np.zeros((5, 1))
                layerTotal = np.zeros((5, 1))
                WOS_PRECISION = 16
                N = 400*NET_WIDTH
                for ni in range(400):
                    for ci in range(NET_WIDTH):
                        for li in range(5):
                            val = stats[li].numpy()[ni,ci]
                            layerStat[li] = layerStat[li] + val
                            layerTotal[li] = layerTotal[li] + WOS_PRECISION
                            #print(str(stats[li].numpy()[ni,ci]) + " ", end='')
                        #print("")
                for li in range(5):
                    frac = layerStat[li] / layerTotal[li]
                    print("Layer " + str(li) + " : " + " has " + str(frac))
            #for stat in stats:            
            #    print(stat.shape)
            #    meanStats = torch.mean(stat, 0)
            #    print(meanStats.squeeze())

    if VIZ_2D:  
        print("Visualizing 2D")
        model.load_state_dict(torch.load(modelName))
        with torch.no_grad():
            YMAP = model(XMAP)
            YMAP = YMAP.squeeze()
            SynData.PlotMap(YMAP)
            fname = 'marginLossNet.4.4.png'
            SynData.PlotMapWithData(YMAP, X, Y, fname)
            
    
    #print("DONE")
