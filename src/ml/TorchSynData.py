import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.distributions.multivariate_normal import MultivariateNormal


def PlotMap(ynp):

    #ynp = YMAP.cpu().detach().numpy() 
    #print(ynp.shape)
    L_sqrt = int(math.sqrt(ynp.shape[0]))
    ynp = ynp.reshape([L_sqrt, L_sqrt])
    ynt = ynp > 0.0
    plt.imshow(ynt, cmap='gray')
    #z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
    #h = plt.contourf(x,y,z)
    plt.show()

def SampleData(mymean, myvar, N, distType = 'uniform'):

    mean1 = torch.Tensor([mymean, -mymean])
    cov1 = torch.eye(2) * myvar
    mean2 = torch.Tensor([-mymean, mymean])
    cov2 = torch.eye(2) * myvar

    mean3 = torch.Tensor([mymean, mymean])
    cov3 = torch.eye(2) * myvar
    mean4 = torch.Tensor([-mymean, -mymean])
    cov4 = torch.eye(2) * myvar

    D = mean1.shape[0]
    
    if distType == 'uniform':
        x1 = torch.rand((N, D)) * (2 * myvar) - myvar
        x1 = x1 + mean1
    else:        
        m1 = MultivariateNormal(mean1, cov1)
        x1 = m1.sample((N, 1)).squeeze()
    y1 = torch.ones(N,1)

    if distType == 'uniform':        
        x2 = torch.rand((N, D)) * (2 * myvar) - myvar
        x2 = x2 + mean2
    else:
        m2 = MultivariateNormal(mean2, cov2)        
        x2 = m2.sample((N, 1)).squeeze()
    y2 = torch.ones(N,1)

    if distType == 'uniform':
        x3 = torch.rand((N, D)) * (2 * myvar) - myvar  
        x3 = x3 + mean3
    else:
        m3 = MultivariateNormal(mean3, cov3)
        x3 = m3.sample((N, 1)).squeeze()

    y3 = -torch.ones(N,1)

    if distType == 'uniform':
        x4 = torch.rand((N, D)) * (2 * myvar) - myvar        
        x4 = x4 + mean4
    else:
        m4 = MultivariateNormal(mean4, cov4)
        x4 = m4.sample((N, 1)).squeeze()
    y4 = -torch.ones(N,1)

    #x1netOut.detach().numpy()      
    x = torch.cat((x1, x2, x3, x4), 0)    
    y = torch.cat((y1, y2, y3, y4), 0)

    indices = torch.randperm(x.shape[0])
    x = x[indices]
    y = y[indices]

    return(x,y)

def LoadXor(N, distType = 'uniform', show = 0):
    
    if distType == 'uniform':
        mymean = 1.5
        myvar = 1
    else:
        mymean = 1
        myvar = 0.1

    myrange = 2
    mySpace = 0.1

    (x, y) = SampleData(mymean, myvar, N, distType)
    
    indices = torch.randperm(x.shape[0])
    x = x[indices]
    y = y[indices]

    (xv, yv) = SampleData(mymean, myvar, N, distType)

    if show == 1:    
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
        
    return(x, y, xv, yv, xxyy)



def LoadLinear(N, show = 0):
    mymean1 = 1
    mymean2 = 1
    myvar1 = 0.1
    myvar2 = 0.1
    myrange = 2
    mySpace = 0.01

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
    if show == 1:    
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
    
def SampleGaussian(mymean, myvar, N):

    mean1 = torch.Tensor([mymean, mymean])
    cov1 = torch.eye(2) * myvar

    m1 = MultivariateNormal(mean1, cov1)
    x1 = m1.sample((N, 1)).squeeze()
    y1 = torch.ones(N,1)

    m2 = MultivariateNormal(mean1, cov1)
    x2 = m2.sample((N, 1)).squeeze()
    y2 = torch.ones(N,1)

    m3 = MultivariateNormal(mean1, cov1)
    x3 = m3.sample((N, 1)).squeeze()
    y3 = -torch.ones(N,1)

    m4 = MultivariateNormal(mean1, cov1)
    x4 = m4.sample((N, 1)).squeeze()
    y4 = -torch.ones(N,1)

    #x1netOut.detach().numpy()      
    x = torch.cat((x1, x2, x3, x4), 0)    
    y = torch.cat((y1, y2, y3, y4), 0)

    indices = torch.randperm(x.shape[0])
    x = x[indices]
    y = y[indices]

    return(x,y)
    
def LoadGaussian(N, show = 0):
    
    mymean = 1
    myvar = 0.1
    myrange = 2
    mySpace = 0.1

    (x, y) = SampleGaussian(mymean, myvar, N)
    
    indices = torch.randperm(x.shape[0])
    x = x[indices]
    y = y[indices]

    (xv, yv) = SampleGaussian(mymean, myvar, N)

    if show == 1:    
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
        
    return(x, y, xv, yv, xxyy)    