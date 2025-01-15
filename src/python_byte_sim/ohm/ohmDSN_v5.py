import torch
import torch.nn.functional as F
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

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def LoadData(bsdsData):
    if bsdsData == 1:    
        (img, seg) = bsds.LoadTrain(0)    
        (img1, seg1, img2, seg2) = bsds.ScaleAndCropData(img, seg)

        bsds.VizTrainTest(img1, seg1, img2, seg2)
        #plt.imshow(img1)
        #plt.ion()
        #plt.show()

        Ysyn = np.expand_dims(seg1, axis=0)
        YTsyn = np.expand_dims(seg2, axis=0)
        Xsyn = np.zeros((img1.shape[2], img1.shape[0], img1.shape[1]))        
        XTsyn = np.zeros((img2.shape[2], img2.shape[0], img2.shape[1]))        
        for c in range(3):
            Xsyn[c,:,:] = img1[:,:,c]
            XTsyn[c,:,:] = img2[:,:,c]

        Xsyn = np.expand_dims(Xsyn, axis=0)
        XTsyn = np.expand_dims(XTsyn, axis=0)

        Xsyn = Xsyn.astype(np.single)
        XTsyn = XTsyn.astype(np.single)
        Ysyn = Ysyn.astype(np.single)
        YTsyn = YTsyn.astype(np.single)
    else:
        Xsyn, Ysyn, XTsyn, YTsyn = QuanSynData.GetData(1)
        plt.figure(1)
        plt.imshow(Xsyn[0,0])

        #make ground truth binary
        Ysyn = (Ysyn != 0)
        Ysyn = np.squeeze(Ysyn.astype(np.longlong))
        if ( len(Ysyn.shape) < 3):
            Ysyn = np.expand_dims(Ysyn, axis=0)

        plt.figure(2)
        plt.imshow(Ysyn[0])        
        #make ground truth binary
        YTsyn = (YTsyn != 0)
        YTsyn = np.squeeze(YTsyn.astype(np.longlong))
        if ( len(YTsyn.shape) < 3):
            YTsyn = np.expand_dims(YTsyn, axis=0)

    XS1 = np.expand_dims(Xsyn[0][0], axis=0)
    XS1 = np.expand_dims(XS1, axis=0)
    XS3 = np.expand_dims(Xsyn[0], axis=0)    
    YL = Ysyn[0]    
    return(XS1, XS3, YL)


def GetSegImages(WG, CG, W, H):
    imgWS = np.zeros((W, H), np.single)
    imgCC = np.zeros((W, H), np.single)
    
    for u, v, d in WG.edges(data = True):
        ul = WG.nodes[u]['label']
        imgWS[u[0], u[1]] = ul
        imgCC[u[0], u[1]] = CG.nodes[ul]['label']

        vl = WG.nodes[v]['label']
        imgWS[v[0], v[1]] = vl
        imgCC[v[0], v[1]] = CG.nodes[vl]['label']

    return(imgWS, imgCC)

def ScaleAndShow(img, fignum):
    minv = np.min(img)
    maxv = np.max(img)
    print("Fig %i: Range %f -> %f" % (fignum, minv, maxv))
    plt.figure(fignum)
    simg = img - minv 
    if abs(maxv - minv) > 1e-4:
        simg = simg / (maxv - minv)
    
    plt.imshow(simg, cmap='gray')


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

    bsdsData = 0
    numEpochs = 10000

    [XS3, XS1, YL] = LoadData(bsdsData)
    
    mint = np.min(XS1)
    maxt = np.max(XS1)
    SX1 = (XS1 - mint) / (maxt - mint)
    SX1 = SX1 * 2.0 - 1.0
    X1T = SX1
    YT = YL*2.0 - 1.0
    
    mint = np.min(X1T)
    maxt = np.max(X1T)
    print("Data Range %f -> %f" % (mint, maxt))
    mint = np.min(YT)
    maxt = np.max(YT)
    print("Label Range %f -> %f" % (mint, maxt))

    #X1 = torch.tensor(XS1, requires_grad=False)    
    #X3 = torch.tensor(XS3, requires_grad=False)    
    X1 = torch.tensor(X1T)
    Y = torch.tensor(YT)
    
    X1 = X1.to(device)
    Y = Y.to(device)   
    
    model = WOS(X1.shape[1], 1, 3, True).to(device)
    #model = Linear(X1.shape[1], 1, 3, True).to(device)

    #loss_fn = nn.MSELoss()
    loss_fn = HingeLoss

    yOut = model(X1)

    minv = torch.min(yOut)
    maxv = torch.max(yOut)        
    print("Prediction range %f -> %f" % (minv, maxv))
    loss = loss_fn(yOut, Y)
    print(f'Initial Loss: {loss}')
    
    #model.MyPrint()
    #wosOut = torch.squeeze(yOut)       
    #img = wosOut.cpu().detach().numpy()
    #ScaleAndShow(img, 3)    
    #plt.show()

    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=1)

    for i in range(numEpochs):
        optimizer.zero_grad()
        yOut = model(X1)
        loss = loss_fn(yOut, Y)
        loss.backward()
        optimizer.step()
        print(f'Iter {i}: Loss: {loss}')
        #print(f'loss is {loss} at iter {i}, weight: {list(mdl.parameters())[0].item()}')

    if numEpochs > 0:
        yOut = model(X1)
        loss = loss_fn(yOut, Y)

        print(f'Final Loss: {loss}')

        minv = torch.min(yOut)
        maxv = torch.max(yOut)
        print("Prediction range %f -> %f" % (minv, maxv))

        #wosOut = torch.sum(y_pred, 0)
        wosOut = torch.squeeze(yOut)       
        img = wosOut.cpu().detach().numpy()
        ScaleAndShow(img, 4)
        plt.show()

    #plt.ion()
    #plt.pause(0.0001)              
    
    print("DONE")
