from bls.RunOHMS import RunOHMS
import torch
import pickle

class MLRunner:

    def __init__(self, memD, memK, numNodes,               
                 biasWeights, ptfWeights, 
                 nx, nxxyy):
    
        self.NN = numNodes      # number of parallel nodes        
        self.numNodes = numNodes        
        self.memD = memD        
        self.K = memK        
        self.biasWeights = biasWeights
        self.ptfWeights = ptfWeights
        
        self.input = self.ScaleData(nx)        
        self.xxyy = self.ScaleData(nxxyy)

        first = self.input[0].tolist()        

        self.ohm = RunOHMS(memD, memK, numNodes, first, biasWeights, ptfWeights)


    def Run(self) -> None:
        print(f"Running for {len(self.input)} samples")

        for ni in range(len(self.input)):            
            sample = self.input[ni].tolist()
            print(f"Sample {ni}: {sample}")
            self.ohm.Run(sample)


    def ScaleData(self, data) -> None:
        min_value = torch.min(data)
        max_value = torch.max(data)        
        
        self.minScale = -3.0
        self.maxScale = 3.0
        print(f"Scaling from: {min_value}->{max_value} to {self.minScale}->{self.maxScale}")
        
        # scale 0 -> 1
        data = (data - self.minScale) / (self.maxScale - self.minScale)
        # scale -1 -> 1
        data = (data - 0.5) * 2.0
        data = data * 127.0
        data = torch.round(data)
        data = data.int()        
        return data
    
    def SaveScale(self, filename) -> None:
        # Save self.minScale, self.maxScale
        with open(filename, 'wb') as f:
            pickle.dump((self.minScale, self.maxScale), f)
    
    def LoadScale(self, filename) -> None:
        # Load self.minScale, self.maxScale
        with open(filename, 'rb') as f:
            self.minScale, self.maxScale = pickle.load(f)
       
       
