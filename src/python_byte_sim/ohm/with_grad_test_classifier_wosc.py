import torch
import torch.nn as nn
#from unet import UNet
from mnet import MNet
from wos import WOS
from linear import Linear
import QuanSynData
import BSDSData as bsds
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pdb
from morphology import Morphology
import math
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd import gradcheck

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def LoadData(N):
    mymean1 = 1
    mymean2 = 1
    myvar1 = 0.1
    myvar2 = 0.1
    myrange = 10
    mySpace = 0.1

    mean1 = torch.Tensor([mymean1, mymean1])
    cov1 = torch.eye(2)*myvar1
    mean2 = torch.Tensor([-mymean2, -mymean2])
    cov2 = torch.eye(2)*myvar2

    m1 = MultivariateNormal(mean1, cov1)
    x1 = m1.sample((N, 1)).squeeze()
    y1 = torch.ones(N,1)

    m2 = MultivariateNormal(mean2, cov2)
    x2 = m2.sample((N, 1)).squeeze()
    y2 = -torch.ones(N,1)
 
    #x1netOut.detach().numpy()      
    x = torch.cat([x1, x2], 0)
    y = torch.cat([y1, y2], 0)
    plt.scatter(x1[:,0], x1[:,1], color='g', marker='o')
    plt.scatter(x2[:,0], x2[:,1], color='r', marker='x')
    plt.show()

    #   X = np.arange(-domain+mean, domain+mean, variance)
    #   Y = np.arange(-domain+mean, domain+mean, variance)
    #   X, Y = np.meshgrid(X, Y)
    xi = np.arange(-myrange, myrange, mySpace)
    yi = np.arange(-myrange, myrange, mySpace)
    xx, yy = np.meshgrid(xi, yi)
    xx = xx.reshape((xx.shape[0]*xx.shape[1], 1))
    yy = yy.reshape((yy.shape[0]*yy.shape[1], 1))
    xxt = torch.Tensor(xx)
    yyt = torch.Tensor(yy)
    
    xxyy = torch.cat([xxt, yyt], 1)
    
    return(x,y, xxyy)
    
def PlotMap(YMAP):

    ynp = YMAP.cpu().detach().numpy() 
    #print(ynp.shape)
    L_sqrt = int(math.sqrt(ynp.shape[0]))
    ynp = ynp.reshape([L_sqrt, L_sqrt])
    ynt = ynp > 0.0
    plt.imshow(ynt, cmap='gray')
    #z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
    #h = plt.contourf(x,y,z)
    plt.show()

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

    torch.manual_seed(0)

    numEpochs = 100
    testGrad = False 
    protoGrad = False

    [X, Y, XMAP] = LoadData(10)
    #print(X)
    #print(Y)
    #print(X.shape)
    #print(XMAP.shape)
    X = X.unsqueeze(-1)
    X = X.unsqueeze(-1)
    XMAP = XMAP.unsqueeze(-1)
    XMAP = XMAP.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    
    X = X.to(device)
    XMAP = XMAP.to(device)
    Y = Y.to(device)   
    
    #loss_fn = nn.MSELoss()
    #loss_fn = MAE
    loss_fn = HingeLoss
    
    if not protoGrad:
        
        X = X.to(device)
        XMAP = XMAP.to(device)
        Y = Y.to(device)   

        model = WOS(X.shape[1], 1, 1).to(device)

    if protoGrad:
        input = torch.Tensor([[2, 3, 5], [4, 7, 1]])    
        labels = torch.Tensor([[-5], [-7]]) 
        print(input)
        print(labels)
        X = input.unsqueeze(-1)
        X = X.unsqueeze(-1)
        Y = labels.unsqueeze(-1)
        Y = Y.unsqueeze(-1)
        
        model = WOS(X.shape[1], 1, 1)    
        
        loss_fn = nn.MSELoss()
                
        #optimizer = torch.optim.Adam(model.parameters())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for i in range(numEpochs):
            model.MyPrint()
            optimizer.zero_grad()
            yOut = model(X)
            print(yOut)
            loss = loss_fn(yOut, Y)
            loss.backward()
            optimizer.step()
            
            print(f'Iter {i}: Loss: {loss}')
            
        exit()

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.    
    if testGrad:
        #input = torch.randn(40,2,dtype=torch.double,requires_grad=False)
        #input = input.unsqueeze(-1)
        #input = input.unsqueeze(-1)
        #input = input.to(device)
        #print(input.shape)
        gradcheck(loss_fn(model(X), Y), model.parameters(), eps=1e-6, atol=1e-4)         
        #test = gradcheck(model, (input,), eps=1e-6, atol=1e-4)
        print(test)
        exit()


    yOut = model(X)
    verbose = 0

    minv = torch.min(yOut)
    maxv = torch.max(yOut)        
    print("Prediction range %f -> %f" % (minv, maxv))
    loss, error = loss_fn(yOut, Y)
    print(f'Initial Loss: {loss} and Error: {error}')
    
    YMAP = model(XMAP)
    YMAP = YMAP.squeeze()
    #PlotMap(YMAP)

    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for i in range(numEpochs):        

        model.verbose = False      
        if i in [66, 67, 68]:
            model.verbose = True        
            
        optimizer.zero_grad()
        yOut = model(X)

        if i in [66, 67, 68]:
            model.MyPrint()
        
        #YMAP = model(XMAP)
        #YMAP = YMAP.squeeze()
        
        loss, error = loss_fn(yOut, Y)
        print(f'Iter {i}: Loss: {loss}  Error: {error}')

        #if i == 68:
        #    exit()    
        if i < numEpochs-1:
            loss.backward()
            optimizer.step()
        #if i % 10 == 0:
        #PlotMap(YMAP)

        #print(f'loss is {loss} at iter {i}, weight: {list(mdl.parameters())[0].item()}')

    model.MyPrint()

    YMAP = model(XMAP)
    YMAP = YMAP.squeeze()
    PlotMap(YMAP)

    #plt.ion()
    #plt.pause(0.0001)              
    
    print("DONE")
