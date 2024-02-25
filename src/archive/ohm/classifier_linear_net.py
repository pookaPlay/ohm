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
    mymean = 1
    myvar = 0.1
    myrange = 2
    mySpace = 0.01

    mean1 = torch.Tensor([mymean, -mymean])
    cov1 = torch.eye(2) * myvar
    mean2 = torch.Tensor([-mymean, mymean])
    cov2 = torch.eye(2) * myvar

    mean3 = torch.Tensor([mymean, mymean])
    cov3 = torch.eye(2) * myvar
    mean4 = torch.Tensor([-mymean, -mymean])
    cov4 = torch.eye(2) * myvar

    m1 = MultivariateNormal(mean1, cov1)
    x1 = m1.sample((N, 1)).squeeze()
    y1 = torch.ones(N,1)

    m2 = MultivariateNormal(mean2, cov2)
    x2 = m2.sample((N, 1)).squeeze()
    y2 = torch.ones(N,1)

    m3 = MultivariateNormal(mean3, cov3)
    x3 = m3.sample((N, 1)).squeeze()
    y3 = -torch.ones(N,1)

    m4 = MultivariateNormal(mean4, cov4)
    x4 = m4.sample((N, 1)).squeeze()
    y4 = -torch.ones(N,1)

    #x1netOut.detach().numpy()      
    x = torch.cat((x1, x2, x3, x4), 0)    
    y = torch.cat((y1, y2, y3, y4), 0)

    mx, mi = torch.max(torch.cat((x, -x),1), 1)     
    yx = y.squeeze() * mx.squeeze()
    
    plt.scatter(x[ y[:,0] > 0 , 0], x[ y[:,0] > 0 , 1], color='g', marker='o')
    plt.scatter(x[ y[:,0] < 0 , 0], x[ y[:,0] < 0 , 1], color='r', marker='x')
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
    
    return(x,y,yx, xxyy)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()      
        self.conv1 = Linear(2, 5, 1)        
        self.conv2 = Linear(5, 5, 1)
        self.conv3 = Linear(5, 1, 1)

    def forward(self, x):
        #print("FWD")
        #print(x.shape)
        x = self.conv1(x)        
        x = torch.tanh(x)        
        x = self.conv2(x)
        x = torch.tanh(x)        
        x = self.conv3(x)
        #print(x.shape)
        return x

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
    protoGrad = False
    sqLoss = False

    [X, Y, YX, XMAP] = LoadData(10)
    #print(X)
    #print(YX)
    #exit()
    #print(X.shape)
    #print(XMAP.shape)
    X = X.unsqueeze(-1)
    X = X.unsqueeze(-1)
    XMAP = XMAP.unsqueeze(-1)
    XMAP = XMAP.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    YX = YX.unsqueeze(-1)
    YX = YX.unsqueeze(-1)
    YX = YX.unsqueeze(-1)
    
    X = X.to(device)
    XMAP = XMAP.to(device)
    Y = Y.to(device)   
    YX = YX.to(device)   

    if sqLoss:
        loss_fn = nn.MSELoss()
    else:
        loss_fn = HingeLoss

    model = Net().to(device)        
    #model = WOS(X.shape[1], 1, 1).to(device)

    yOut = model(X)

    minv = torch.min(yOut)
    maxv = torch.max(yOut)        
    print("Prediction range %f -> %f" % (minv, maxv))
    loss = loss_fn(yOut, Y)
    print(f'Initial Loss: {loss}')
    
    #YMAP = model(XMAP)
    #YMAP = YMAP.squeeze()
    #PlotMap(YMAP)

    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for i in range(numEpochs):
        #model.MyPrint()
        optimizer.zero_grad()
        yOut = model(X)
        if sqLoss:
            loss = loss_fn(yOut, YX)
        else:
            loss = loss_fn(yOut, Y)
        
        loss.backward()
        optimizer.step()
        #if i % 10 == 0:
        print(f'Iter {i}: Loss: {loss}')
            #YMAP = model(XMAP)
            #YMAP = YMAP.squeeze()
            #PlotMap(YMAP)

        #print(f'loss is {loss} at iter {i}, weight: {list(mdl.parameters())[0].item()}')

    #model.MyPrint()

    YMAP = model(XMAP)
    YMAP = YMAP.squeeze()
    PlotMap(YMAP)

    #plt.ion()
    #plt.pause(0.0001)              
    
    print("DONE")
