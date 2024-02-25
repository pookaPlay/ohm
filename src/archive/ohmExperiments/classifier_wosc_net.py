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
from wos_net import WOSNet
from wos_net_fanin import WOSNetFanin
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def HingeLoss(YP, Y):    
    
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
    numEpochs = 10000
    numSamples = 100
    learningRate = 1
    learningRho = 0.99

    NET_WIDTH = 8
    NET_DEPTH = 8
    modelLastName = "model_xor_last_" + str(NET_WIDTH) + "_" + str(NET_DEPTH) + ".pth"
    modelLossName = "model_xor_loss_" + str(NET_WIDTH) + "_" + str(NET_DEPTH) + ".pth"
    modelErrorName = "model_xor_error_" + str(NET_WIDTH) + "_" + str(NET_DEPTH) + ".pth"

    np.random.seed(theSeed)
    random.seed(theSeed)
    torch.manual_seed(theSeed)

    (X, Y, XV, YV, XMAP) = SynData.LoadXor(numSamples)

    XV = X.clone()
    YV = Y.clone()
    print(XV.shape)
    myrange = 2
    mySpace = 0.05
    for xi in range(XV.shape[0]):
        for di in range(XV.shape[1]):
            if XV[xi,di] > 0.0:
                XV[xi,di] = np.floor(XV[xi,di] / mySpace)  * mySpace
            else:
                XV[xi,di] = np.ceil(XV[xi,di] / mySpace)  * mySpace

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

    model = WOSNetFanin(NET_WIDTH, NET_DEPTH).to(device)        
    clipper = ClipNegatives()
    #model.MyPrint()
    
    optimizer = torch.optim.Adadelta(model.parameters(), rho=learningRho, lr=learningRate)

    bestLoss = math.inf
    bestLossInd = 0
    bestError = math.inf
    bestErrorInd = 0

    bestValidLoss = math.inf
    bestValidInd = 0

    train_loss = list()
    train_error = list()

    imgLossCount = 1
    imgErrorCount = 1
    for i in range(numEpochs):
        optimizer.zero_grad()                                        

        yOut = model(XV)        
        loss, error = loss_fn(yOut, Y)
        train_loss.append(loss)
        train_error.append(error)

        # if (i%5) == 0:
        #     YMAP = model(XMAP)
        #     YMAP = YMAP.squeeze()
        #     fname = 'linear.' + str(i) + '.png'
        #     SynData.PlotMapWithData(YMAP, X, Y, fname)

        if loss < bestLoss:
            bestLoss = loss                
            bestLossInd = i
            torch.save(model.state_dict(), modelLossName)
            #YMAP = model(XMAP)
            #YMAP = YMAP.squeeze()
            #fname = 'xorLoss2.' + str(imgLossCount) + '.png'            
            #SynData.PlotMapWithData(YMAP, X, Y, fname)
            #imgLossCount = imgLossCount + 1
            #modelName = modelLossName + "." + str(i) + ".pth"
            #torch.save(model.state_dict(), modelName)


        if error < bestError:
            bestError = error
            bestErrorInd = i
            torch.save(model.state_dict(), modelErrorName)
            #YMAP = model(XMAP)
            #YMAP = YMAP.squeeze()
            #fname = 'xorError2.' + str(imgErrorCount) + '.png'            
            #SynData.PlotMapWithData(YMAP, X, Y, fname)
            #imgErrorCount = imgErrorCount + 1

            #modelName = modelErrorName + "." + str(i) + ".pth"
            #torch.save(model.state_dict(), modelName)

        print(f'{i}: Loss: {loss}  Error: {error}  Best: {bestLoss} ({bestLossInd}) Error: {bestError} ({bestErrorInd})')

        loss.backward()
        optimizer.step()
        model.apply(clipper)


        torch.save(model.state_dict(), modelLastName)
        #with open('trn_4_loss_xor.pkl', 'wb') as f:
        #    pickle.dump(train_loss, f)
        #with open('trn_4_error_xor.pkl', 'wb') as f:
        #    pickle.dump(train_error, f)

# 4x4   9999: Loss: 0.48003292083740234  Error: 0.20000000298023224  Best: 0.027091724798083305 (971) Error: 0.004999999888241291 (657)
# 4x8    9999: Loss: 0.6170315742492676  Error: 0.2524999976158142  Best: 0.024556003510951996 (4051) Error: 0.007499999832361937 (5401)
# 4x4 0.999 9999: Loss: 0.3058342933654785  Error: 0.03500000014901161  Best: 0.03461381793022156 (3158) Error: 0.0024999999441206455 (990)