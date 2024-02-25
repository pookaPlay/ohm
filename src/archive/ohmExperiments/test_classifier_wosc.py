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


##############################################################
## Basic Training Program
if __name__ == '__main__':
    theSeed = 0
    verbose = 0
    numEpochs = 50
    learningRate = 1
    numSamples = 2
    dataSet = 'linear'
    errorSet = 'loss'
    #dataSet = 'linear'

    np.random.seed(theSeed)
    random.seed(theSeed)
    torch.manual_seed(theSeed)

    if dataSet == 'xor':
        [X, Y, XV, YV, XMAP] = SynData.LoadXor(numSamples)        
    else:
        [X, Y, XMAP] = SynData.LoadLinear(numSamples)
    
    modelName = 'model_' + dataSet + '_' + errorSet + '.pth'
    intName = 'int_' + dataSet + '_' + errorSet + '.pth'

    X = X.unsqueeze(-1)
    X = X.unsqueeze(-1)
    XMAP = XMAP.unsqueeze(-1)
    XMAP = XMAP.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    
    X = X.to(device)
    XMAP = XMAP.to(device)
    Y = Y.to(device)   
            
    model = WOS(X.shape[1], 1, 1, test_init=True).to(device)    
    converter = ConvertToInteger()
    
    if False:
        print("Visualizing 2D")
        #model.load_state_dict(torch.load(modelName))
        
        with torch.no_grad():                
            YMAP = model(XMAP)
            YMAP = YMAP.squeeze()
            SynData.PlotMap(YMAP)

    if False:
        print("Converting to Integer")
        model.load_state_dict(torch.load(modelName))
        model.apply(converter)
        torch.save(model.state_dict(), intName) 
    
    if True:
        model.load_state_dict(torch.load(modelName))
        with torch.no_grad():
            YP  = model(X)
            stats = model.FindRanks(X, YP)            
            print(stats)
    
    print("DONE")
