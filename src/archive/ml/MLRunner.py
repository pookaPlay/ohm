from bls.OHM_NETWORK import OHM_NETWORK
import matplotlib.pyplot as plt
import math
import pickle
import torch
import sys

smallest_int = -sys.maxsize - 1
largest_int = sys.maxsize - 1

class MLRunner:

    def __init__(self, nx, nxxyy, param):
    
        self.numInputs = param['numInputs']
        self.depth = param['numLayers']     # number of layers in time              
        self.memD = param['memD']
        self.K = param['memK']
        self.param = param
        self.plotResults = dict()
        #self.dataMax = 2**self.K - 1.0        
        self.input = self.ScaleData(nx, param['scaleTo'], param['clipAt'])        

        self.xxyy = self.ScaleData(nxxyy, param['scaleTo'], param['clipAt'])
        
        self.output = torch.zeros_like(self.input)
        self.first = self.input[0].tolist()        
        self.ohm = OHM_NETWORK(self.first, param)   

    def Run(self, param) -> None:    

        ticks = list()
        
        self.posStackStats = 0
        self.negStackStats = 0
        self.sampleCount = 0
        self.posStack = 0
        self.negStack = 0
        
        numSamples = len(self.input)
        
        # Run the OHM
        sample = self.input[0].tolist()                
        print(f"Input: {sample}")
        
        results = self.ohm.Run(sample, 0, param)            
        print(f"Output: {results}")
        
        return self.output       
        
    def WeightAdjust(self) -> None:
        weights = self.ohm.paramBiasMem[0].GetLSBIntsHack()        
        for i in range(len(self.weights)):                    
            if (self.weights[i] > 1):
                self.weights[i] = int(math.floor(self.weights[i] / 2))
            
        self.ohm.paramStackMem[0].SetLSBIntsHack(self.weights)

    def ApplyToMap(self, param) -> None:
        print(f"Running on {len(self.xxyy)} samples")
                
        ticks = 0
        results = torch.zeros(len(self.xxyy))
        for ni in range(len(self.xxyy)):            
            sample = self.xxyy[ni].tolist()
            #print(f"Sample {ni}\n{sample}")
            # Run the OHM
            atick = self.ohm.Run(sample, ni, param)

            results[ni] = self.ohm.results[0]
                                
        return(results)

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
       
       
