import torch
import urllib
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle 



def GetData(NUM_SAMPLES = 30, MAX_IDX = 64, REDUCTION = 1):
    # Training set
    im = list()
    gt = list()
    f = open('../../data/train.p', 'rb')
    data = pickle.load(f)
    f.close()  
    #print(data[0][0].shape)
    for i in range(NUM_SAMPLES):
        myX = data[0][i][0:MAX_IDX:REDUCTION,0:MAX_IDX:REDUCTION,:]
        newX = np.zeros((myX.shape[2], myX.shape[0], myX.shape[1]))        
        for c in range(3):
            newX[c,:,:] = myX[:,:,c]
            
        im.append(newX) 
        gt.append(data[1][i][0:MAX_IDX:REDUCTION,0:MAX_IDX:REDUCTION,:])
    
    X = np.array(im).astype(np.single)
    Y = np.array(gt).astype(np.single)
    # Validation set
    im = list()
    gt = list()
    f = open('../../data/test.p', 'rb')
    data = pickle.load(f)
    f.close() 
    for i in range(NUM_SAMPLES):
        myX = data[0][i][0:MAX_IDX:REDUCTION,0:MAX_IDX:REDUCTION,:]
        newX = np.zeros((myX.shape[2], myX.shape[0], myX.shape[1]))        
        for c in range(3):
            newX[c,:,:] = myX[:,:,c]        
        im.append(newX)
        gt.append(data[1][i][0:MAX_IDX:REDUCTION,0:MAX_IDX:REDUCTION,:])
    XT = np.array(im).astype(np.single)
    YT = np.array(gt).astype(np.single)

    #print('Training')
    #print(X.shape)
    #print(Y.shape)
    #print('Testing')
    #print(XT.shape)
    #print(YT.shape)
    
    return X, Y, XT, YT


if __name__ == '__main__':
    X, Y, XT, YT = GetData()
    plt.imshow(X[0].squeeze())
    plt.figure()
    plt.imshow(Y[0].squeeze())
    T = Y[0].squeeze()
    TT = (T == 0)
    plt.figure()        
    plt.imshow(TT) 

    print(np.min(Y[0]))
    plt.show()