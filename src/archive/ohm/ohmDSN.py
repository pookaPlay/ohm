import torch
import torch.nn.functional as F
#from unet import UNet
from mnet import MNet
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
        
        Ysyn = np.squeeze(Ysyn.astype(np.longlong))
        if ( len(Ysyn.shape) < 3):
            Ysyn = np.expand_dims(Ysyn, axis=0)

        plt.figure(2)
        plt.imshow(Ysyn[0])        

        YTsyn = np.squeeze(YTsyn.astype(np.longlong))
        if ( len(YTsyn.shape) < 3):
            YTsyn = np.expand_dims(YTsyn, axis=0)

    XS1 = np.expand_dims(Xsyn[0][0], axis=0)
    XS1 = np.expand_dims(XS1, axis=0)
    XS3 = np.expand_dims(Xsyn[0], axis=0)    
    #make ground truth binary
    Ysyn = (Ysyn != 0)
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


class WOS(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        kernel_size=3,
    ):
        super(WOS, self).__init__()
        self.ksize = kernel_size
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.ksize, padding=(1,1)),
            nn.Sigmoid(),
        )
        
    def forward(self, X):
        out = self.layer(X)
        
        return out

##############################################################
## Basic Training Program
if __name__ == '__main__':

    bsdsData = 0
    numEpochs = 10    

    [XS1, XS3, YL] = LoadData(bsdsData)

    #X1 = torch.tensor(XS1, requires_grad=False)    
    #X3 = torch.tensor(XS3, requires_grad=False)    
    X1 = torch.tensor(XS1)
    X3 = torch.tensor(XS3)
    X3 = X3.to(device)                         


#model = UNet(in_channels=3, n_classes=1, depth=5, padding=True, up_mode='upconv').to(device)
#model = MNet(in_channels=3, n_classes=1, depth=2, padding=True, up_mode='upsample').to(device)
#model = Morphology(in_channels=3, out_channels=1, kernel_size=3, soft_max=True, type='dilation2d')
model = Morphology(in_channels=3, out_channels=1, kernel_size=3, soft_max=False, type='dilation2d')
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# was
optimizer = torch.optim.Adam(model.parameters())
verbose = 1
THRESH_OFFSET = 0.75

    mdl = NN_Linear_Model()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(mdl.parameters(), lr=0.0001)

    for i in range(100):
        optimizer.zero_grad()
        y_pred = mdl(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        for p in mdl.parameters():
            p.data.clamp_(0)
    
        if i % 10 == 0:
            print(f'loss is {loss} at iter {i}, weight: {list(mdl.parameters())[0].item()}')

    list(mdl.parameters())

    mOut = model(X3)  
    squOut = torch.squeeze(mOut)
    print(squOut.shape)
    imgOut = squOut.cpu().detach().numpy()
    ScaleAndShow(imgOut, 3)        
    plt.show()
    #plt.ion()
    #plt.pause(0.0001)              
    
    print("DONE")
