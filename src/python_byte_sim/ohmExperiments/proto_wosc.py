import torch
import torch.nn as nn
from wos import WOS
from wos import ClipNegatives
from wos import ConvertToInteger
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pdb
import math
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd import gradcheck
import SynData
import random

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

##############################################################
## Basic Training Program
if __name__ == '__main__':
    theSeed = 0
    verbose = 0
    numEpochs = 1
    learningRate = 1
    learningRho = 0.9

    np.random.seed(theSeed)
    random.seed(theSeed)
    torch.manual_seed(theSeed)

    #[X, Y, XV, YV, XMAP] = SynData.LoadXor(50)
    [X, Y, XMAP] = SynData.LoadLinear(5)
    XT = X[:,0]+0.1
    XT = XT.unsqueeze(-1)
    
    X = torch.cat((X, XT), 1) 
    
    
    X = X.unsqueeze(-1)
    X = X.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    
    X = X.to(device)
    XMAP = XMAP.to(device)
    Y = Y.to(device)   
            
    X = X.to(device)
    XMAP = XMAP.to(device)
    Y = Y.to(device)   
    #print(X.shape)
    #print(Y.shape)
    #model = WOS(X.shape[1], 2, 1, test_init=True).to(device)    
    model = WOS(X.shape[1], 1, 1, test_init=True).to(device)    
    clipper = ClipNegatives()
    converter = ConvertToInteger()
    loss_fn = HingeLoss

    optimizer = torch.optim.Adadelta(model.parameters(), rho=learningRho, lr=learningRate)     

    model.MyPrint()
    
    for i in range(numEpochs):        
                
        optimizer.zero_grad()
        
        yOut = model(X)           
        loss, error = loss_fn(yOut, Y)
        loss.backward()
        optimizer.step()
        
        model.apply(clipper) 

    if False:

        with torch.no_grad():                
            Y = model(X)
            #stat = model.FindRanks(X, Y)
            #YMAP = model(XMAP)
            #YMAP = YMAP.squeeze()
            #SynData.PlotMap(YMAP)
            #print(stat)            

        model.apply(converter)
        print("INTEGER WOS")
        model.MyPrint()
        torch.save(model.state_dict(), modelName) 
        
        with torch.no_grad():                
            #Y = model(X)
            YMAP = model(XMAP)
            YMAP = YMAP.squeeze()
            SynData.PlotMap(YMAP)
        #print(Y)

        #plt.ion()
        #plt.pause(0.0001)              
    
    print("DONE")
