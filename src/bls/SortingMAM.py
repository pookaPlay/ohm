import matplotlib.pyplot as plt
import math
import pickle
import torch
import sys

smallest_int = -sys.maxsize - 1
largest_int = sys.maxsize - 1

class SortingMAM:

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
        
    
    def BatchTrain(self, nx, y) -> None:
        D = self.input.shape[-1]
        N = len(self.input)
        #D = nx.shape[-1]
        #N = len(nx)
     
        print(f"Processing {N} samples with D={D}")
        
        self.W = torch.ones([D, D]) * largest_int
        self.M = torch.ones([D, D]) * smallest_int        

        imgMatList = []
        for ni in range(N):            
            input = self.input[ni].view(1, D)
            #input = nx[ni].view(1, D)
            
            sinput, sindex = torch.sort(input, dim=1, descending=True)            
            sinput = sinput.view(D, 1)
            #print(sinput)
            #sinput = y[ni].view(D, 1)
            
            imgMat = sinput - input            
            #imgMat = y[ni].view(D, 1) - nx[ni].view(1, D)            

            self.W = torch.min(self.W, imgMat)                
            self.M = torch.max(self.M, imgMat)    
            #imgMatList.append(input)
        
        #imgMatTensor = torch.stack(imgMatList)        
        #print(f"imgMatTensor: {imgMatTensor.shape}")
        
        #self.MED, self.IND = torch.median(imgMatTensor, dim=0)
        #kval = int(D/2)    # smallest        
        #self.MED, self.IND = torch.kthvalue(imgMatTensor, k=kval, dim=0)
        
        print(f"W: {self.W}")
        print(f"M: {self.M}")

    def BatchTest(self, nx, y) -> None:
        D = self.input.shape[-1]
        N = len(self.input)
        #D = nx.shape[-1]
        #N = len(nx)
   
        print(f"Testing  {N} samples with D={D}")

        self.outputW = torch.zeros([D])
        self.outputM = torch.zeros([D])
        self.outputMED = torch.zeros([D])
        self.outputIND = torch.zeros([D])

        for ni in range(N):
            input = self.input[ni, :]
            #input = nx[ni, :]
            
            for di in range(D):            
                
                diffW = input + self.W[di, :]                
                self.outputW[di] = torch.max(diffW)
                
                diffM = input + self.M[di,:]                
                self.outputM[di] = torch.min(diffM)
            
                #diffMED = self.input[ni, :] + self.MED[di,:]
                #self.outputMED[di] = torch.median(diffMED)
                #kval = int(D/2)
                #kval = D-2
                #self.outputMED[di], self.outputIND[di] = torch.kthvalue(diffMED, k=kval)



            self.outputW = self.outputW.int()
            self.outputM = self.outputM.int()
            self.outputMED = self.outputMED.int()

            print(f"W: {input.tolist()} -> {self.outputW.tolist()}")
            print(f"M: {input.tolist()} -> {self.outputM.tolist()}")
            #print(f"W: {self.input[ni].tolist()} -> {self.outputW.tolist()}")
            #print(f"M: {self.input[ni].tolist()} -> {self.outputM.tolist()}")
            #print(f"MED: {self.input[ni].tolist()} -> {self.outputMED.tolist()}")
        
    
        #input = torch.tensor([-110, 119, 110, -119])
        tpl = [[23, 35], [99, 99], [47, -53]]
        tpl = []
        for ni in range(len(tpl)):
            tp = torch.tensor(tpl[ni])
            input = torch.cat((tp, -tp), dim=0)            

            self.outputW = torch.zeros([D])
            self.outputM = torch.zeros([D])
            for di in range(D):            
                diffW = input + self.W[di, :]
                self.outputW[di] = torch.max(diffW)
                diffM= input + self.M[di, :]
                self.outputM[di] = torch.min(diffM)
            
            self.outputW = self.outputW.int()
            self.outputM = self.outputM.int()
            
            print(f"Test W: {input.tolist()} -> {self.outputW.tolist()}")
            print(f"Test M: {input.tolist()} -> {self.outputM.tolist()}")


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
       
       
