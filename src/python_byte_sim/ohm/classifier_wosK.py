import torch
import torch.nn as nn
from wosK import WOSK
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd import gradcheck
import SynData
import random
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def MAE(YP, YL):    

    loss = torch.abs(YP - YL)
    mae = torch.mean(loss)      
    return(mae)

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
    print(f"Device: {device}")

    theSeed = 0
    verbose = 0
    numEpochs = 500
    learningRate = 1

    random.seed(theSeed)
    torch.manual_seed(theSeed)

    #[X, Y, XV, YV, XMAP] = SynData.LoadXor(50)
    [X, Y, XMAP] = SynData.LoadLinear(100)
    X = X.unsqueeze(-1)
    X = X.unsqueeze(-1)
    XMAP = XMAP.unsqueeze(-1)
    XMAP = XMAP.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    
    X = X.to(device)
    XMAP = XMAP.to(device)
    Y = Y.to(device)   
    
    #loss_fn = MAE
    loss_fn = HingeLoss
        
    X = X.to(device)
    XMAP = XMAP.to(device)
    Y = Y.to(device)   

    model = WOS(X.shape[1], 1, 1).to(device)
    clipper = ClipNegatives()

    yOut = model(X)    

    minv = torch.min(yOut)
    maxv = torch.max(yOut)        
    print("Prediction range %f -> %f" % (minv, maxv))
    loss, error = loss_fn(yOut, Y)
    print(f'Initial Loss: {loss} and Error: {error}')
    
    YMAP = model(XMAP)
    YMAP = YMAP.squeeze()
    #PlotMap(YMAP)

    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.Adadelta(model.parameters(), rho=0.9, lr=1.0)     

    model.train()
    with torch.enable_grad():                            
        for i in range(numEpochs):        
            
            model.verbose = False      
            #model.MyPrint()
            
            optimizer.zero_grad()
            
            yOut = model(X)

            #model.eval()
            #with torch.no_grad():                        
                #YMAP = model(XMAP)
                #YMAP = YMAP.squeeze()
                #SynData.PlotMap(YMAP)

            loss, error = loss_fn(yOut, Y)
            print(f'Iter {i}: Loss: {loss}  Error: {error}')

            loss.backward()
            optimizer.step()
            
            model.apply(clipper) 
        

    model.MyPrint()
    
    model.eval()
    with torch.no_grad():                
        YMAP = model(XMAP)
        YMAP = YMAP.squeeze()
        SynData.PlotMap(YMAP)

    #plt.ion()
    #plt.pause(0.0001)              
    
    print("DONE")
