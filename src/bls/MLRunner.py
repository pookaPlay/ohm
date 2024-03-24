from bls.RunOHMS import RunOHMS
import torch
import pickle

class MLRunner:

    def __init__(self, memD, memK, numNodes,               
                 biasWeights, ptfWeights, 
                 nx, nxxyy, adaptWeights):
    
        self.NN = numNodes      # number of parallel nodes        
        self.numNodes = numNodes        
        self.memD = memD        
        self.K = memK        
        self.biasWeights = biasWeights
        self.ptfWeights = ptfWeights
        
        #self.dataMax = 2**self.K - 1.0
        self.dataMax = 127.0
        self.input = self.ScaleData(nx, self.dataMax)
        self.xxyy = self.ScaleData(nxxyy, self.dataMax)

        first = self.input[0].tolist()        

        self.ohm = RunOHMS(memD, memK, numNodes, first, biasWeights, ptfWeights, adaptWeights)


    def Run(self) -> None:
        print(f"Running on {len(self.input)} samples")
        
        ticks = list()
        for ni in range(1):
        #for ni in range(len(self.input)):                        
            #sample = self.input[ni].tolist()
            sample = self.input[0].tolist()
            print(f"------------------------------")
            self.weights = self.ohm.paramStackMem[0].GetLSBIntsHack()
            print(f"     PTF IN: {self.weights}")                  
            atick = self.ohm.Run(sample)

            print(f"Sample {ni}: {sample} -> {self.ohm.results[0]}")                        
            #self.ohm.paramStackMem[0].Print(f"PTF step {ni}")
            self.weights = self.ohm.paramStackMem[0].GetLSBIntsHack()
            print(f"     PTF OUT: {self.weights}")                  

            if atick < 0: 
                atick = self.K
            ticks.append(atick)            

        avg = sum(ticks) / len(ticks)
        print(f"Avg Ticks: {avg}")
        

    def ApplyToMap(self) -> None:
        print(f"Running on {len(self.xxyy)} samples")
        ticks = 0
        results = torch.zeros(len(self.xxyy))
        for ni in range(len(self.xxyy)):            
            sample = self.xxyy[ni].tolist()
            #print(f"Sample {ni}\n{sample}")
            atick = self.ohm.Run(sample)
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
       
       
