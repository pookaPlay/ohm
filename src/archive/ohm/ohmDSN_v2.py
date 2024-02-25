import torch
import torch.nn.functional as F
import torch.nn as nn
from linNet import LinNet
from wosNet import WosNet
import QuanSynData
import BSDSData as bsds
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pdb
from morphology import Morphology

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

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


def Sobel(img):
    #Black and white input image x, 1x1xHxW
    a = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
    b = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)    
    c = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32)
    #d = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.float32)
    a = a.to(device) 
    b = b.to(device) 
    c = c.to(device) 
    #d = d.to(device) 

    a = a.view((1,1,3,3))
    b = b.view((1,1,3,3))
    c = c.view((1,1,3,3))
    #d = d.view((1,1,5,5))
    #print('In sobel type')
    #print(imgIn.type())
    #print(a.type())
    
    imgIn = F.conv2d(img, c, padding=(1,1))
    #imgIn = F.conv2d(img, d, padding=(2,2))
    #imgIn = img
    G_x = F.conv2d(imgIn, a, padding=(1,1))
    G_y = F.conv2d(imgIn, b, padding=(1,1))

    G_x = -torch.pow(G_x,2) 
    G_y = -torch.pow(G_y,2) 
    GXY = -torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
    #DXY = F.max_pool2d(GXY, (2,2))
    #DY = F.max_pool2d(GY, (2,2))    
    return G_x, G_y, GXY


def ThresholdExpand(X, Y, margin, numThreshLevels, mirror):    
    # expects -1 -> +1
    if margin==0:
        print("Threshold Expand with zero margin")
        numThreshLevels = 1
        if mirror:
            NX = -X    
            XT = np.zeros((numThreshLevels, 2*X.shape[1], X.shape[2], X.shape[3]))
        else:
            XT = np.zeros((numThreshLevels, X.shape[1], X.shape[2], X.shape[3]))
        
        YT = np.zeros((numThreshLevels, 1, Y.shape[0], Y.shape[1]))

        if mirror:
            XT[0, 0:X.shape[1], :, :] = X[0, :, :, :] > 0.0
            XT[0, X.shape[1]:-1, :, :] = NX[0, :, :, :] > 0.0
        else:
            XT[0, :, :, :] = X[0, :, :, :] > 0.0
        YT[0, 0, :, :] = Y[:,:]        
    else:
        dt = 2.0 / numThreshLevels
        if mirror:
            NX = -X    
            XT = np.zeros((numThreshLevels, 2*X.shape[1], X.shape[2], X.shape[3]))
        else:
            XT = np.zeros((numThreshLevels, X.shape[1], X.shape[2], X.shape[3]))
        
        YT = np.zeros((numThreshLevels, 1, Y.shape[0], Y.shape[1]))


        i = 0
        for t in np.arange(-1.0, 1.0, dt):        
            if mirror:
                XT[i, 0:X.shape[1], :, :] = X[0, :, :, :] > t
                XT[i, X.shape[1]:-1, :, :] = NX[0, :, :, :] > t
            else:
                XT[i, :, :, :] = X[0, :, :, :] > t
            YT[i, 0, :, :] = Y[:,:]
            i = i + 1
    
    XT = XT.astype(np.single)
    XT = XT*2.0 - 1.0
    YT = YT.astype(np.single)    
    YT = YT*2.0 - 1.0
    return(XT, YT)

def SetPositive(model, mirror):
    if mirror:
        for m in model.modules():
            if hasattr(m, 'weight') and m.weight is not None:
                #print(m.weight.data)
                ind0 = m.weight.data[0][0] < 0                
                m.weight.data[0][1][ind0] = -m.weight.data[0][0][ind0]
                m.weight.data[0][0][ind0] = 0.0
                ind1 = m.weight.data[0][1] < 0
                m.weight.data[0][0][ind1] = -m.weight.data[0][1][ind1]
                m.weight.data[0][1][ind1] = 0.0
                #print(m.weight.data)
                #m.weight.data.clamp_(0)                
    else:
        for m in model.modules():
            if hasattr(m, 'weight') and m.weight is not None:                
                m.weight.data.clamp_(0.001)                
        #if hasattr(m, 'bias') and m.bias is not None:            

def HingeLoss(YP, YL):    
    loss = 1.0 - torch.mul(YP, Y)
    loss[loss < 0.0] = 0.0
    #werror = torch.mul(hinge_loss, tweights)
    hingeLoss = torch.mean(loss)      
    return(hingeLoss)

##############################################################
## Basic Training Program
if __name__ == '__main__':
    
    torch.manual_seed(1)
    
    bsdsData = 0
    numEpochs = 20000
    mirror = False
    mirrorPos = False
    setPositive = False
    margin = 0
    threshExpand = False
    useBatchNorm = True
    numThresholds = 101    
    resulti = (numThresholds-1)/2

    [XS3, XS1, YL] = LoadData(bsdsData)
    
    mint = np.min(XS1)
    maxt = np.max(XS1)
    SX1 = (XS1 - mint) / (maxt - mint)
    SX1 = SX1 * 2.0 - 1.0
    if threshExpand:
        [X1T, YT] = ThresholdExpand(SX1, YL, margin, numThresholds, mirror)
    else:
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
    #model = WosNet(X1.shape[1], 1, depth=2, wf=5, padding=True, up_mode='upsample', batch_norm=useBatchNorm).to(device) 
    model = LinNet(X1.shape[1], 1, depth=2, wf=5, padding=True, up_mode='upsample', batch_norm=useBatchNorm).to(device) 

    if setPositive:
        SetPositive(model, mirrorPos)
    
    #loss_fn = nn.MSELoss()
    loss_fn = HingeLoss

    yOut = model(X1)

    minv = torch.min(yOut)
    maxv = torch.max(yOut)        
    print("Prediction range %f -> %f" % (minv, maxv))
    loss = loss_fn(yOut, Y)
    print(f'Initial Loss: {loss}')
        
    wosOut = torch.squeeze(yOut)       
    img = wosOut.cpu().detach().numpy()
    ScaleAndShow(img, 3)    
    plt.show()

    optimizer = torch.optim.Adam(model.parameters())
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for i in range(numEpochs):
        optimizer.zero_grad()
        yOut = model(X1)
        loss = loss_fn(yOut, Y)
        loss.backward()
        optimizer.step()
        # clip non positive
        if setPositive:
            SetPositive(model, mirrorPos)
        if i % 10 == 0:
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
