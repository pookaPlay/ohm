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
# from wos_net import WOSNet
from wos_net_mnist import WOSNetMnist

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

    theSeed = 0  
    numSamples = 10
    
    CONVERT_TO_INT = False
    GET_INT_STATS = False
    GET_FLOAT_STATS = True
    VIZ_2D = False
    EVAL_TEST = False
    
    NET_WIDTH = 8
    batchSize = 5

    modelName = "mymnist_6580.pth"
    statsName = "stats_16bit_mymnist_6580.pkl"
    #modelName = "mymnist_error_model.pth"
    #modelName = "best_mnist_model.pth"
    intName = "int_" + modelName
    
        
    np.random.seed(theSeed)
    random.seed(theSeed)
    torch.manual_seed(theSeed)

    dataDir = "c:\\src\\dsn\\data\\mnist\\MNIST\\processed\\"
    trainName = dataDir + 'training.pt'
    with open(trainName, 'rb') as f:
        XT, YT = torch.load(f)
    
    #print(XT.shape)
    #print(YT.shape)
    train1s = (YT == 1)
    trainNot = (YT != 1)
    num1s = torch.sum(train1s)
    numNot = torch.sum(trainNot)

    X1 = XT[train1s]
    XN = XT[trainNot]

    #t1i = np.random.permutation(num1s.item()) 
    #tni = np.random.permutation(numNot.item())         
    #t1ii = t1i[0:batchSize]
    #tnii = tni[0:batchSize]

    X = torch.cat((X1[0:batchSize], XN[0:batchSize]), 0)
    Y1 = torch.ones((batchSize))
    YN = -torch.ones((batchSize))
    Y = torch.cat((Y1, YN))
    X = X.type(torch.FloatTensor)
    X = X.unsqueeze(1)                

    X = X.to(device)
    Y = Y.to(device)   

    #loss_fn = nn.MSELoss()
    #loss_fn = MAE
    loss_fn = HingeLoss

    
    model = WOSNetMnist(NET_WIDTH).to(device)        
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
        print("Calculating integer stats")
        model.load_state_dict(torch.load(intName))                        
        model.MyPrint()
        with torch.no_grad():
            stats = model.FindRanks(X)
            with open(statsName, 'wb') as f:
                pickle.dump(stats, f)

            for stat in stats:            
                meanStats = np.mean(stat, 0)
                print(meanStats.squeeze())

    if GET_FLOAT_STATS:  
        print("Calculating stats for reals")
        model.load_state_dict(torch.load(modelName))                        
        #model.MyPrint()
        model.MyShapes()
        
        with torch.no_grad():
            stats = model.FindRanks(X)
            print("######################################################")
            print(stats)
            with open(statsName, 'wb') as f:
                pickle.dump(stats, f)

            #for stat in stats:            
            #    meanStats = np.mean(stat, 0)
            #    print(meanStats.squeeze())

    if VIZ_2D:  
        print("Visualizing 2D")
        model.load_state_dict(torch.load(modelName))
        with torch.no_grad():
            YMAP = model(XMAP)
            YMAP = YMAP.squeeze()
            SynData.PlotMap(YMAP)

    if EVAL_TEST:  
        print("Evaluating")
        model.load_state_dict(torch.load(modelName))
        with torch.no_grad():
            yOut = model(X)
            #print(yOut)
            loss, error = loss_fn(yOut, Y)                
            print("Current Loss   : " + str(loss.detach().numpy().tolist()) + "   Err: " + str(error))
    
    print("DONE")
