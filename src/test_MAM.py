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

def run_MAM(expName, nx, y):

    D = nx.shape[-1]
    N = len(nx)
    
    #print("DATA")
    #for ni in range(N):
    #    print(f"{nx[ni].tolist()} -> {y[ni].tolist()}")

    #print(f"Processing {N} samples with D={D}")
        
    W = torch.ones([D, D]) * largest_int
    M = torch.ones([D, D]) * smallest_int        
    imgMatList = []
    
    
    imgMatList = []

    for ni in range(N):            

        imgMat = nx[ni].view(D, 1) - nx[ni].view(1, D)            
        W = torch.min(W, imgMat)                
        M = torch.max(M, imgMat)    
        imgMatList.append(imgMat)

    imgMatTensor = torch.stack(imgMatList)        
    print(f"imgMatTensor: {imgMatTensor.shape}")
            
    kval = int(N/2)    # 1 is smallest
    print(f"Kth value: {kval}")
    
    MED, IND = torch.kthvalue(imgMatTensor, k=kval, dim=0)
    #print(f"MED: {self.MED.shape} and IND: {self.IND.shape}")
    
    print(f"W: {W}")
    print(f"MED: {MED}")



    # Now test
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

        #print(f"W: {input.tolist()} -> {outputW.tolist()}")
        #print(f"M: {input.tolist()} -> {outputM.tolist()}")
        if (not torch.all(torch.eq(outputW, outputM))):
            print(f"{expName} result difference")
            print(f"{y[ni].tolist()}: {outputW.tolist()} != {outputM.tolist()}")
    

def test_MAM():
    exp = "paper"
    nx = torch.tensor([[0, 0, 0], [0, -2, -4], [0, -3, 0]])
    y = torch.tensor([[0, 1, 0], [-1, -1, 0], [0, -2, 0]])    
    run_MAM(exp, nx, y)

    exp = "inc"
    nx = torch.tensor([[2, 1, 3], [0, -2, -4], [-1, 1, 0]])
    y = torch.tensor([[1, 2, 3], [-4, -2, 0], [-1, 0, 1]])    
    run_MAM(exp, nx, y)
    
    exp = "dec"
    nx = torch.tensor([[2, 1, 3], [0, -2, -4], [-1, 1, 0]])
    y = torch.tensor([[3, 2, 1], [0, -2, -4], [1, 0, -1]])    
    run_MAM(exp, nx, y)
    
    exp = "combo"
    nx = torch.tensor([[2, 1, 3], [0, -2, -4], [-1, 1, 0]])
    y = torch.tensor([[1, 2, 3], [0, -2, -4], [-1, 0, 1]])    
    run_MAM(exp, nx, y)

    exp = "mag"
    nx = torch.tensor([[2, 1, 3], [0, -2, -4], [-3, 1, 0]])
    y = torch.tensor([[1, 2, 3], [-4, -2, 0], [-3, 0, 1]])    
    run_MAM(exp, nx, y)



test_MAM()