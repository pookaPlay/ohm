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
    mymean = 5
    myrange = 10
    mySpace = 0.1

    mean1 = torch.Tensor([mymean, mymean])
    cov1 = torch.eye(2)
    mean2 = torch.Tensor([-mymean, -mymean])
    cov2 = torch.eye(2)

    m1 = MultivariateNormal(mean1, cov1)
    x1 = m1.sample((N, 1)).squeeze()
    y1 = -torch.ones(N,1)

    m2 = MultivariateNormal(mean2, cov2)
    x2 = m2.sample((N, 1)).squeeze()
    y2 = torch.ones(N,1)
 
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

def HingeLoss(YP, YL):    
    loss = 1.0 - torch.mul(YP, Y)
    loss[loss < 0.0] = 0.0
    #werror = torch.mul(hinge_loss, tweights)
    hingeLoss = torch.mean(loss)      
    return(hingeLoss)

##############################################################
## Basic Training Program
if __name__ == '__main__':

    torch.manual_seed(0)

    numEpochs = 500
    testGrad = False

    [X, Y, XMAP] = LoadData(100)
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

    #model = WOS(X1.shape[1], 1, 5).to(device)
    model = Linear(X.shape[1], 1, 1).to(device)

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.    
    if testGrad:
        input = torch.randn(40,2,dtype=torch.double,requires_grad=True)
        input = input.unsqueeze(-1)
        input = input.unsqueeze(-1)
        input = input.to(device)
        #print(input.shape)
        test = gradcheck(model, (input,), eps=1e-6, atol=1e-4)
        print(test)
        exit()


    loss_fn = nn.MSELoss()
    #loss_fn = HingeLoss
    yOut = model(X)

    minv = torch.min(yOut)
    maxv = torch.max(yOut)        
    print("Prediction range %f -> %f" % (minv, maxv))
    loss = loss_fn(yOut, Y)
    print(f'Initial Loss: {loss}')
    
    YMAP = model(XMAP)
    YMAP = YMAP.squeeze()
    PlotMap(YMAP)

    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for i in range(numEpochs):
        optimizer.zero_grad()
        yOut = model(X)
        loss = loss_fn(yOut, Y)
        loss.backward()
        optimizer.step()
        print(f'Iter {i}: Loss: {loss}')
            #YMAP = model(XMAP)
            #YMAP = YMAP.squeeze()
            #PlotMap(YMAP)

        #print(f'loss is {loss} at iter {i}, weight: {list(mdl.parameters())[0].item()}')

    YMAP = model(XMAP)
    YMAP = YMAP.squeeze()
    PlotMap(YMAP)

    #plt.ion()
    #plt.pause(0.0001)              
    
    print("DONE")
