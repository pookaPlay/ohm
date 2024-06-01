from ml.TorchSynData import LoadXor
from ml.TorchSynData import LoadLinear
from ml.TorchSynData import PlotMap
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys 

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

smallest_int = -sys.maxsize - 1
largest_int = sys.maxsize - 1

def test_MAM():

    nx = torch.tensor([[0, 0, 0], [0, -2, -4], [0, -3, 0]])
    y = torch.tensor([[0, 1, 0], [-1, -1, 0], [0, -2, 0]])    
    
    D = nx.shape[-1]
    N = len(nx)

    print(f"Processing {N} samples with D={D}")
        
    W = torch.ones([D, D]) * largest_int
    M = torch.ones([D, D]) * smallest_int        
    
    for ni in range(N):            
        #input = nx[ni].view(1, D)        
        input = nx[ni]
        #print(f"        input: {input.tolist()}")
        
        #sinput, sindex = torch.sort(input, dim=0, descending=False)
        #sinput = sinput.view(D, 1)
        sinput = y[ni].view(D, 1)  
        #print(f"sorted input: {sinput.tolist()}")
       
        imgMat = sinput - input
        #print(f"   mat: {imgMat.tolist()}")
        W = torch.min(W, imgMat)                
        M = torch.max(M, imgMat)    
        #imgMatList.append(input)
    
    #imgMatTensor = torch.stack(imgMatList)        
    #print(f"imgMatTensor: {imgMatTensor.shape}")
    
    #self.MED, self.IND = torch.median(imgMatTensor, dim=0)
    #kval = int(D/2)    # smallest        
    #self.MED, self.IND = torch.kthvalue(imgMatTensor, k=kval, dim=0)
    
    print(f"W: {W}")
    print(f"M: {M}")


    outputW = torch.zeros([D])
    outputM = torch.zeros([D])
    
    for ni in range(N):
        input = nx[ni, :]        
        for di in range(D):            
            
            diffW = input + W[di, :]                
            outputW[di] = torch.max(diffW)
            
            diffM = input + M[di,:]                
            outputM[di] = torch.min(diffM)

        outputW = outputW.int()
        outputM = outputM.int()

        print(f"W: {input.tolist()} -> {outputW.tolist()}")
        print(f"M: {input.tolist()} -> {outputM.tolist()}")
    
test_MAM()