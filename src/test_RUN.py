from ml.TorchSynData import LoadXor
from ml.TorchSynData import LoadLinear
from ml.TorchSynData import PlotMap
from bls.RunOHMS import RunOHMS
from bls.MLRunner import MLRunner
import torch
import numpy as np


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def ThreshExpand(x, thresh):
    
    N = x.shape[0]
    D = x.shape[1]
    ND = 2 * D * len(thresh)
    #print(x)
    #print(f"Expanding {N} samples to {ND} dimensions")
    nx = torch.zeros([N, ND])
    for t in range(len(thresh)):
        nx[:, t*D:(t+1)*D] = x - thresh[t]
    
    for t in range(len(thresh)):        
        nx[:, (t+len(thresh))*D:(t+len(thresh)+1)*D] = thresh[t] - x
    
    return nx

def test_RUN():

    adaptWeights = 0
    display = 0
    thresholds = [1.0, -1.0]
    numPoints = 5
    
    x, y, xv, yv, xxyy = LoadXor(numPoints, display)
    #x, y, xxyy = LoadLinear(numPoints, display)
        
    print(x.shape)
    print(y.shape)
    print(xxyy.shape)

    nx = ThreshExpand(x, thresholds)        
    nxxyy = ThreshExpand(xxyy, thresholds)        
        
    
    memK = 8        
    #input = [8, -8, 4, -4, 2, -2, 1, -1]
    #input += [-x for x in input]  # Add the negative values
    dataN = nx.shape[0]
    memD = len(nx[0])
    numNodes = memD
    biasWeights = numNodes * [0]
    ptfWeights = numNodes * [1]
    #ptfWeights[0] = numNodes
    #ptfWeights = [19, 0, 976, 0, 926, 11201, 6, 0]
    
    runner = MLRunner(memD, memK, numNodes, 
                      biasWeights, ptfWeights, 
                      nx, nxxyy, adaptWeights)
    runner.Run()            
    #result = runner.ApplyToMap()
    #PlotMap(result)


test_RUN()