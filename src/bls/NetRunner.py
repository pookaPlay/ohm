from bls.RunNetwork import RunNetwork
import matplotlib.pyplot as plt
import math
import pickle
import torch
import sys

smallest_int = -sys.maxsize - 1
largest_int = sys.maxsize - 1

class NetRunner:

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
        self.ohm = RunNetwork(self.first, param)   

    def Run(self, param) -> None:
        print(f"Running on {len(self.input)} samples")
        ticks = list()
        self.posStackStats = 0
        self.negStackStats = 0
        self.sampleCount = 0
        self.posStack = 0
        self.negStack = 0
        
        numSamples = len(self.input)
        
        #self.ohm.SetAdaptWeights(adaptWeights)        
        self.plotResults['thresh'] = list()

        # if param['printParameters'] == 1:            
        #     for i in range(len(self.ohm.paramStackMem)):                    
        #             weights = self.ohm.paramStackMem[i].GetLSBIntsHack()
        #             thresh = self.ohm.paramThreshMem[i].GetLSBIntsHack()                    
        #             biases = self.ohm.paramBiasMem[i].GetLSBInts()                                                        
        #             print(f"{i}          Bias: {biases}")                                                                  
        #             print(f"{i}       Weights: {weights}")                                       
        #             print(f"{i}        Thresh: {thresh}")


        for ni in range(numSamples):                                                            
            #########################################################
            #########################################################
            # Run the OHM
            sample = self.input[ni].tolist()
            if param['printSample'] == 1:
                print(f"Sample {ni} ------------------")
                print(f"Input: {sample}")


            results = self.ohm.Run(sample, ni, param)
            
            self.output[ni] = torch.tensor(results)
                                    
            if param['printSample'] == 1:
                print(f"Output: {results}")
        
        return self.output
            # if param['printSample'] == 1:
            #     for si in range(len(self.ohm.biasMem)):
            #         stackInputs = self.ohm.biasMem[si].GetLSBInts()
            #         print(f"{stackInputs} -> {results[si]}[{outIndex[si]}] in {self.ohm.doneOut[si]}")

            #     if param['printParameters'] == 1:            
            #         for i in range(len(self.ohm.paramStackMem)):                    
            #                 weights = self.ohm.paramStackMem[i].GetLSBIntsHack()
            #                 thresh = self.ohm.paramThreshMem[i].GetLSBIntsHack()                    
            #                 biases = self.ohm.paramBiasMem[i].GetLSBInts()                                                        
            #                 print(f"               Bias{i}: {biases}")                                                                  
            #                 print(f"            Weights{i}: {weights}")                                       
            #                 print(f"             Thresh{i}: {thresh}")

            
            # if (param['adaptBias'] > 0):
            #     biases = self.ohm.paramBiasMem[0].GetLSBInts()                                                        
                
            #     #sample = self.input[ni].tolist()                
            #     for i in range(len(sample)):
            #         if sample[i] > biases[i]:
            #             biases[i] = biases[i] + 1
            #         else:
            #             biases[i] = biases[i] - 1                    
                
            #     self.ohm.paramBiasMem[0].LoadList(biases)
                

        # if param['printIteration'] == 1:
        #     #avg = sum(ticks) / len(ticks)
        #     #print(f"Avg Ticks: {avg}")

        #     if param['printParameters'] == 1:                                
        #         for i in range(len(self.ohm.paramStackMem)):                    
        #             biases = self.ohm.paramBiasMem[i].GetLSBInts()
        #             print(f"{i}          Bias: {biases}")                                    
        #             weights = self.ohm.paramStackMem[i].GetLSBIntsHack()
        #             thresh = self.ohm.paramThreshMem[i].GetLSBIntsHack()                    
        #             print(f"{i}       Weights: {weights}")                                       
        #             print(f"{i}        Thresh: {thresh}")

        
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
       
       
