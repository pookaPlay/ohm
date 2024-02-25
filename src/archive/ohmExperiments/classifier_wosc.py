import torch
import torch.nn as nn
from wos import WOS
from wos import ClipNegatives
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
    theSeed = 0
    verbose = 0
    numEpochs = 150
    numSamples = 100
    learningRho = 0.99
    learningRate = 1
    #dataSet = 'xor'
    dataSet = 'linear'

    np.random.seed(theSeed)
    random.seed(theSeed)
    torch.manual_seed(theSeed)

    if dataSet == 'xor':
        [X, Y, XV, YV, XMAP] = SynData.LoadXor(numSamples)        
    else:
        [X, Y, XMAP] = SynData.LoadLinear(numSamples)
    
    lossName = 'model_' + dataSet + '_loss.pth'
    errorName = 'model_' + dataSet + '_error.pth'

    X = X.unsqueeze(-1)
    X = X.unsqueeze(-1)
    XMAP = XMAP.unsqueeze(-1)
    XMAP = XMAP.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    
    X = X.to(device)
    XMAP = XMAP.to(device)
    Y = Y.to(device)   
            
    X = X.to(device)
    XMAP = XMAP.to(device)
    Y = Y.to(device)   

    model = WOS(X.shape[1], 1, 1).to(device)
    clipper = ClipNegatives()
    loss_fn = HingeLoss


    optimizer = torch.optim.Adadelta(model.parameters(), rho=learningRho, lr=learningRate)     

    bestLoss = math.inf
    bestLossInd = 0
    bestError = math.inf
    bestErrorInd = 0

    train_loss = list()
    train_error = list()
    
    #model.train()
    model.train()
    with torch.enable_grad():                            

        for i in range(numEpochs):        
            
            model.verbose = False      
            optimizer.zero_grad()
            #model.MyPrint()
            yOut = model(X)           
            loss, error = loss_fn(yOut, Y)
            #model.MyPrint()        
            print(f'{i}: Loss: {loss}  Error: {error}  Best: {bestLoss} ({bestLossInd}) Error: {bestError} ({bestErrorInd})')

            if (i%5) == 0:
                YMAP = model(XMAP)
                YMAP = YMAP.squeeze()
                fname = 'linear.' + str(i) + '.png'
                SynData.PlotMapWithData(YMAP, X, Y, fname)

            if loss < bestLoss:
                bestLoss = loss                
                bestLossInd = i
                torch.save(model.state_dict(), lossName)

            if error < bestError:
                bestError = error
                bestErrorInd = i
                torch.save(model.state_dict(), errorName)

            loss.backward()
            optimizer.step()
            
            model.apply(clipper) 

    model.MyPrint()        

    
    print("DONE")
