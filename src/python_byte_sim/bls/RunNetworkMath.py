def GetNegativeIndex(din, N):
    if din < N/2:
        dout = int(din + N/2)
    else:
        dout = int(din - N/2)
    return dout

class RunNetworkMath:

    def __init__(self, param):
    
        self.param = param        

        self.numStack = param['numStack']      # number of parallel nodes        
        self.numInputs = param['numInputs']
        self.numLayers = param['numLayers']
        
        self.memD = param['memD']        
        self.K = param['memK']        

        self.dataMem = [0] * self.memD 
        self.stackMem = [[0] * self.numStack for _ in range(self.numLayers)]        
        self.biasMem = [[[0] * self.numInputs for _ in range(self.numStack)] for _ in range(self.numLayers)]

        self.paramBias = [[[0] * self.numInputs for _ in range(self.numStack)] for _ in range(self.numLayers)]
        self.paramMasks = [[[0] * self.numInputs for _ in range(self.numStack)] for _ in range(self.numLayers)]
        self.paramWeights = [[[0] * (self.numInputs * 2) for _ in range(self.numStack)] for _ in range(self.numLayers)]
        self.paramThresh = [[0 for _ in range(self.numStack)] for _ in range(self.numLayers)]
                             
        
    def Run(self, input, sampleIndex, param) -> None:      
                
        D = len(input)
        assert(D > 0)
        
        self.input = input
        self.dataMem = self.input

        for layerIndex in range(self.numLayers):

            if layerIndex == 0:
                inputMem = self.dataMem
            else:
                inputMem = self.stackMem[layerIndex-1]
            
            print(f"   >> LSB PASS ")
            for bi in range(D):
                self.biasMem[layerIndex][bi] = inputMem[bi] + self.paramBias[layerIndex][bi]

            print(f"   >> MSB PASS ")
            for si in range(self.numStack):
                self.stackMem[layerIndex][si] = self.biasMem[layerIndex][si]           
                        
            self.results = self.stackMem[layerIndex]

        
        if param['adaptThresh'] > 0:           
            pass
        

        return self.results

