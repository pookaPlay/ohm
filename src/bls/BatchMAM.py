import matplotlib.pyplot as plt
import math
import pickle
import torch
import sys

smallest_int = -sys.maxsize - 1
largest_int = sys.maxsize - 1

class BatchMAM:

    def __init__(self, nx, nxxyy, param):
    
        self.NN = param['numNodes']      # number of parallel nodes        
        self.numNodes = param['numNodes']
        self.memD = param['memD']
        self.K = param['memK']
        self.param = param
        self.plotResults = dict()
        #self.dataMax = 2**self.K - 1.0        
        self.input = self.ScaleData(nx, param['scaleTo'], param['clipAt'])
        self.xxyy = self.ScaleData(nxxyy, param['scaleTo'], param['clipAt'])

        self.first = self.input[0].tolist()        
        
    
    def BatchTrainMAM(self) -> None:
        D = self.input.shape[-1]
        
        print(f"Processing {len(self.input)} samples with D={D}")
        
        self.W = torch.ones([D, D]) * smallest_int                
        self.M = torch.ones([D, D]) * largest_int
        
        for ni in range(len(self.input)):            
            imgMat = self.input[ni].view(D, 1) - self.input[ni].view(1, D)            
            self.W = torch.max(self.W, imgMat)
            self.M = torch.min(self.M, imgMat)

            #print(f"{self.input[ni].tolist()}")        
                

    def BatchTestMAM(self) -> None:
        D = self.input.shape[-1]
        N = len(self.input)
        print(f"Testing  {N} samples with D={D}")
                
        self.output = torch.ones([D]) * smallest_int

        for ni in range(N):
            for di in range(D):            
                diff = self.input[ni, :] - self.W[:, di]
                self.output[di] = torch.max(diff)
            
            self.output = self.output.int()
            print(f"{self.input[ni].tolist()} -> {self.output.tolist()}")

        input = torch.tensor([-47, -43, 47, 43])
        for di in range(D):            
            diff = input - self.W[:, di]
            self.output[di] = torch.max(diff)
        
        self.output = self.output.int()
        print(f"T {input.tolist()} -> {self.output.tolist()}")


    def ScaleData(self, data, maxValOut, clipValOut) -> None:
        #self.min_value = torch.min(data)
        #self.max_value = torch.max(data)        
        
        self.minScale = -3.0
        self.maxScale = 3.0
        #print(f"Scaling from: {self.min_value}->{self.max_value} to {self.minScale}->{self.maxScale}")
        
        # scale 0 -> 1
        data = (data - self.minScale) / (self.maxScale - self.minScale)
        data[data < 0.0] = 0.0
        data[data > 1.0] = 1.0
        # scale -1 -> 1
        data = (data - 0.5) * 2.0
        
        data = data * maxValOut
        
        data[data > clipValOut] = clipValOut
        data[data < -clipValOut] = -clipValOut
                
        data = torch.round(data)
        data = data.int()        

        return data

    def ReverseScaleData(self, data, maxValOut, clipValOut) -> None:
        #self.min_value = torch.min(data)
        #self.max_value = torch.max(data)        
        
        #self.minScale = -3.0
        #self.maxScale = 3.0
        # maxValOut = 127
        data = (data / maxValOut) * self.maxScale        

        return data

    def SaveScale(self, filename) -> None:
        # Save self.minScale, self.maxScale
        with open(filename, 'wb') as f:
            pickle.dump((self.min_value, self.max_value), f)
    
    def LoadScale(self, filename) -> None:
        # Load self.minScale, self.maxScale
        with open(filename, 'rb') as f:
            self.min_value, self.max_value = pickle.load(f)
       
       
