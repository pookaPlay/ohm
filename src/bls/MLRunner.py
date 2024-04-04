from bls.RunOHMS import RunOHMS
import math
import pickle
import torch

class MLRunner:

    def __init__(self, nx, nxxyy, param):
    
        self.NN = param['numNodes']      # number of parallel nodes        
        self.numNodes = param['numNodes']
        self.memD = param['memD']
        self.K = param['memK']
        self.param = param
        self.plotResults = dict()
        #self.dataMax = 2**self.K - 1.0        
        self.input = self.ScaleData(nx, param['scaleTo'])
        self.xxyy = self.ScaleData(nxxyy, param['scaleTo'])

        first = self.input[0].tolist()        

        self.ohm = RunOHMS(first, param)


    def Run(self, param) -> None:
        print(f"Running on {len(self.input)} samples")
        ticks = list()
        #self.ohm.SetAdaptWeights(adaptWeights)        
        self.plotResults['thresh'] = list()
        
        for ni in range(len(self.input)):                                                
            if param['printSample'] == 1:
                print(f"Sample {ni} ------------------")

            biases = self.ohm.paramBiasMem[0].GetLSBInts()                                                        
            if param['printBias'] == 1:
                print(f"       Bias: {biases}")                                  
            weights = self.ohm.paramStackMem[0].GetLSBIntsHack()                                                        
            if param['printWeights'] == 1:
                print(f"       Weights: {weights}")                                       
            thresh = self.ohm.paramThreshMem[0].GetLSBIntsHack()                                                        
            if param['printThresh'] == 1:
                print(f"       Thresh: {thresh}")                                       
            
            # Save stuff for plots
            self.plotResults['thresh'].append(thresh[0])

            sample = self.input[ni].tolist()
            atick = self.ohm.Run(sample, ni, param)

            if atick < 0: 
                atick = self.K
            ticks.append(atick)            

            outIndex = self.ohm.doneIndexOut[0]
            stackInputs = self.ohm.biasMem[0].GetLSBInts()

            if param['printSample'] == 1:
                print(f"{stackInputs} -> {self.ohm.results[0]}[{self.ohm.doneIndexOut[0]}] in {self.ohm.doneOut[0]}")
            
            if param['adaptBias'] == 1:
                biases = self.ohm.paramBiasMem[0].GetLSBInts()                                        
                biases[outIndex] = biases[outIndex] + 1 
                self.ohm.paramBiasMem[0].LoadList(biases)
                #print(f"     Bias OUT: {weights}")                                  

        avg = sum(ticks) / len(ticks)
        print(f"Avg Ticks: {avg}")
        
    def WeightAdjust(self) -> None:
        weights = self.ohm.paramBiasMem[0].GetLSBIntsHack()
        #self.weights = self.ohm.paramStackMem[0].GetLSBIntsHack()
        for i in range(len(self.weights)):                    
            if (self.weights[i] > 1):
                self.weights[i] = int(math.floor(self.weights[i] / 2))
            
        self.ohm.paramStackMem[0].SetLSBIntsHack(self.weights)

    def ApplyToMap(self, param) -> None:
        print(f"Running on {len(self.xxyy)} samples")
                
        #self.ohm.SetAdaptWeights(adaptWeights)        

        ticks = 0
        results = torch.zeros(len(self.xxyy))
        for ni in range(len(self.xxyy)):            
            sample = self.xxyy[ni].tolist()
            #print(f"Sample {ni}\n{sample}")
            atick = self.ohm.Run(sample, ni, param)
            results[ni] = self.ohm.results[0]
            
            if atick < 0: 
                atick = self.K
            ticks += atick

        avg = ticks / len(self.xxyy)
        print(f"Ticks: {avg}")
        return(results)

    def ScaleData(self, data, maxValOut) -> None:
        self.min_value = torch.min(data)
        self.max_value = torch.max(data)        
        
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
        
        data = torch.round(data)
        data = data.int()        
        return data
    
    def SaveScale(self, filename) -> None:
        # Save self.minScale, self.maxScale
        with open(filename, 'wb') as f:
            pickle.dump((self.min_value, self.max_value), f)
    
    def LoadScale(self, filename) -> None:
        # Load self.minScale, self.maxScale
        with open(filename, 'rb') as f:
            self.min_value, self.max_value = pickle.load(f)
       
       
