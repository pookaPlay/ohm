import progressbar
import sys
import glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import pdb
import skimage.io as skio
from scipy.signal import medfilt as med_filt
import math
import random
import skimage.transform
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from scipy.ndimage.measurements import label
from wos import ClipNegatives
from wos import WOS
import torch.nn as nn
from wos_mnist import WOSMnist

device = 'cpu'

def HingeLoss(YP, Y):    
    
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
    verbose = 0
    theSeed = 0

    numEpochs = 10000

    learningRate = 1
    rhoMemory = 0.99
        
    batchSize = 100
    numValid = 10
    validOffset = 30000
     
    random.seed(theSeed)
    torch.manual_seed(theSeed)

    dataDir = "c:\\src\\dsn\\data\\mnist\\MNIST\\processed\\"
    trainName = dataDir + 'training.pt'
    with open(trainName, 'rb') as f:
        XT, YT = torch.load(f)
    
    print(XT.shape)
    print(YT.shape)
    YT[YT < 5] = 1.0
    YT[YT >= 5] = -1.0
    exit()
    
    NNX = XT.shape[0]
    
    device = 'cpu'

    loss_fn = HingeLoss
    model = WOSMnist().to(device)        
    clipper = ClipNegatives()

    modelName = "best_mnist_model_2.pth" 
    model.load_state_dict(torch.load(modelName)) 
    #torch.save(model.state_dict(), modelName)      

    optimizer = optim.Adadelta(model.parameters(), rho=rhoMemory, lr=learningRate)

    # Initializing training and validation lists to be empty
    trn_loss   = []
    trn_error = []

    # Initializing best loss and random error
    trn_best_loss   = math.inf
    trn_best_error = math.inf

    for cepoch in range(0,numEpochs):
        print("Epoch :    ",str(cepoch))
        
        ti = np.random.permutation(NNX-batchSize-1)
        tii = ti[1:batchSize]
        #print(tii)
        X1 = XT[ tii ]
        Y1 = YT[ tii ]

        X1 = X1.type(torch.FloatTensor)
        X1 = X1.unsqueeze(1)                

        optimizer.zero_grad()
        yOut = model(X1)
            
        loss, error = loss_fn(yOut, Y1)                

        trn_loss = trn_loss + [loss.detach().numpy().tolist()]
        trn_error  = trn_error + [error]

        avg_loss = sum(trn_loss)/len(trn_loss)
        avg_error  = sum(trn_error)/len(trn_error)

        print("Current Loss   : " + str(loss.detach().numpy().tolist()) + "   Err: " + str(error))
        print("AVG Loss   : " + str(avg_loss) + "   Err: " + str(avg_error))        

        if error < trn_best_error:
            trn_best_error = error            
            trn_best_error_model = model
            modelName = "mnist_best_error_model.pth" 
            torch.save(trn_best_error_model.state_dict(), modelName)

        if ((cepoch % 10) == 0): 
            print("Saving model, loss and errors")
            
            modelName = "mnist_model_" + str(cepoch) + ".pth" 
            torch.save(model.state_dict(), modelName)      

            vname = 'mnist_loss.pkl'
            with open(vname, 'wb') as f:
                pickle.dump(trn_loss, f)
            tname = 'mnist_error.pkl'
            with open(tname, 'wb') as f:
                pickle.dump(trn_error, f)            

            
        loss.backward()
        optimizer.step()
        model.apply(clipper) 

# # Validation every epoch        
# loss_lst_epoch   = []
# rerror_lst_epoch = []
# # Bar
# model.eval()
# with torch.no_grad():                

#     bar = progressbar.ProgressBar(maxval=numValid, widgets=[progressbar.Bar('-', '    Val[', ']'), ' ', progressbar.Percentage()])
#     bar.start()

#     for val_idx in range(0, numValid):
#         bar.update(val_idx+1)
#         # print("\t Validating on ",str(val_idx)," image")

        
#         XV1 = XVT[ ti[val_idx*batchSize:((val_idx+1)*batchSize)] ]
#         YV1 = YVT[ ti[val_idx*batchSize:((val_idx+1)*batchSize)] ]

#         XV1 = XV1.type(torch.FloatTensor)
#         YV1 = YV1.type(torch.LongTensor)
#         XV1 = XV1.unsqueeze(1)
#         XV1 = XV1.to(device)                                       

#         output = model(XV1)
#         loss = F.nll_loss(output, YV1)                

#         smx, si = torch.sort(output, 1, descending=True)                
#         mypred = si[:,0]                
#         errors = torch.sum(mypred != Y1).type(torch.FloatTensor).detach().numpy()
#         myerror = errors / mypred.shape[0]

#         rerror_lst_epoch  = rerror_lst_epoch + [myerror]
#         loss_lst_epoch    = loss_lst_epoch       + [loss.detach().numpy().tolist()]

#     # Finish bar
#     bar.finish()            

#     val_cepoch_loss    = sum(loss_lst_epoch)/len(loss_lst_epoch)
#     val_cepoch_rerror  = sum(rerror_lst_epoch)/len(rerror_lst_epoch)
#     val_rerror_lst     = val_rerror_lst + [val_cepoch_rerror]
#     val_loss_lst       = val_loss_lst   + [val_cepoch_loss]

#     if val_cepoch_loss < val_best_loss:
#         # Saving best loss model
#         val_best_loss       = val_cepoch_loss
#         val_best_loss_epoch = cepoch
#         val_best_loss_model = model
#         #print("\t\tBest val loss ", str(val_best_loss))
#         #print("\t\tCurrent val error ", str(val_best_rerror))
#         #torch.save(val_best_loss_model.state_dict(), "val_loss_model_1.pth")

#     if val_cepoch_rerror < val_best_rerror:
#         # Saving best rerror model
#         val_best_rerror       = val_cepoch_rerror
#         val_best_rerror_epoch = cepoch
#         val_best_rerror_model = model
#         print("Saving model with error ", str(val_best_rerror))
#         modelName = "val_error_model_" + str(cepoch) + ".pth"
#         torch.save(val_best_rerror_model.state_dict(), modelName)

#     if ((cepoch % 500) == 0): 
#         print("Saving valid loss and error")
#         vname = 'valid_loss_' + str(cepoch) + '.pkl'
#         with open(vname, 'wb') as f:
#             pickle.dump(val_loss_lst, f)
#         tname = 'valid_error_' + str(cepoch) + '.pkl'                    
#         with open(tname, 'wb') as f:
#             pickle.dump(val_rerror_lst, f)            
    
# Loading model
# loaded_model = UNet(in_channels=1, n_classes=1, depth=5, padding=True, up_mode='upsample').to(device)
# loaded_model.load_state_dict(torch.load("best_val_model.pth"))
# uOut1 = model(X3)
# uOut2 = loaded_model(X3)
