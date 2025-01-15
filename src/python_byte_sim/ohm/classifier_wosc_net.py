import torch
import torch.nn as nn
from wos import WOS
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

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

#if hasattr(module, 'weight'):

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()      
        self.conv1 = WOS(2, 5, 1)
        #self.drop1 = nn.Dropout(p=0.001)
        self.conv2 = WOS(5, 5, 1)
        self.conv3 = WOS(5, 1, 1)

    def forward(self, x):
        #print("FWD")
        #print(x.shape)
        x = self.conv1(x)        
        #x = self.drop1(x)        
        x = self.conv2(x)
        x = self.conv3(x)
        #print(x.shape)
        return x
    
    def MyPrint(self):
        self.conv1.MyPrint()
        self.conv2.MyPrint()
        self.conv3.MyPrint()

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
    numEpochs = 5000    
    numSamples = 50
    learningRate = 1

    random.seed(theSeed)
    torch.manual_seed(theSeed)

    (X, Y, XV, YV, XMAP) = SynData.LoadXor(numSamples)

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

    model = Net().to(device)        
    clipper = ClipNegatives()

    if False:
        
        optimizer = torch.optim.Adadelta(model.parameters(), rho=0.9, lr=learningRate)

        bestLoss = math.inf
        bestInd = 0
        bestValidLoss = math.inf
        bestValidInd = 0

        valid_epoch = list()
        train_epoch = list()
        for i in range(numEpochs):
            #model.MyPrint()
            model.eval()
            with torch.no_grad():                
                vOut = model(XV)        
                lossv, errorv = loss_fn(vOut, YV)
                valid_epoch.append(lossv)

                if lossv < bestValidLoss:
                    bestValidLoss = lossv            
                    bestValidInd = i
                    torch.save(model.state_dict(), "best_ohm_valid.pth")

            model.train()
            with torch.enable_grad():                            
                optimizer.zero_grad()                                        
                yOut = model(X)        
                loss, error = loss_fn(yOut, Y)
                train_epoch.append(loss)

                if loss < bestLoss:
                    bestLoss = loss            
                    bestInd = i
                    torch.save(model.state_dict(), "best_ohm_train.pth")

            
                loss.backward()
                optimizer.step()
                model.apply(clipper) 
            
            #print(f'Iter {i}: Loss: {loss}  Best: {bestLoss} ({bestInd})      Valid: {lossv}  BestValid: {bestValidLoss} ({bestValidInd})')
            print(f'Train: Iter {i}: Loss: {loss}  Error: {error}')
            print(f'Apply: Iter {i}: Loss: {lossv}  Error: {errorv}')
            #scheduler.step()


        with open('trn_loss_ohm.pkl', 'wb') as f:
            pickle.dump(train_epoch, f)
        with open('val_loss_ohm.pkl', 'wb') as f:
            pickle.dump(valid_epoch, f)

    if True:
        #print('')
        #print('############################################################')
        #print('############################################################')
        #print('')
        #model.MyPrint()
        

        
        #model = Net().to(device)                
        model.load_state_dict(torch.load("best_ohm_valid.pth"))
        model.MyPrint()
        model.eval()
        with torch.no_grad():
            print('')
            print('############################################################')
            print('')
            vOut = model(XV)
            lossv, errorv = loss_fn(vOut, YV)
            print(f'Apply: Loss: {lossv}  Error: {errorv}')
            YMAP = model(XMAP)
            YMAP = YMAP.squeeze()
            SynData.PlotMap(YMAP)
            
        #plt.ion()
        #plt.pause(0.0001)              
        
        print("DONE")
