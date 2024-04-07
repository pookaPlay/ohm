from ml.TorchSynData import LoadXor
from ml.TorchSynData import LoadLinear
from ml.TorchSynData import PlotMap
from bls.RunOHMS import RunOHMS
from bls.MLRunner import MLRunner
import torch
import numpy as np
import matplotlib.pyplot as plt

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

    numIterations = 1
    
    display = 0
    threshSpac = 0.25   # 64
    threshSpac = 1.0   # 16
    thresholds = np.arange(-2.0, 2.0, threshSpac).tolist()     
    print(thresholds)
    numPoints = 25
        
    x, y, xv, yv, xxyy = LoadXor(numPoints, display)
    #x, y, xxyy = LoadLinear(numPoints, display)
        
    nx = ThreshExpand(x, thresholds)        
    nxxyy = ThreshExpand(xxyy, thresholds)        

    print(f"Thresh expand: {x.shape} -> {nx.shape}")

    memK = 8       
    #input = [8, -8, 4, -4, 2, -2, 1, -1]
    #input += [-x for x in input]  # Add the negative values
    dataN = nx.shape[0]
    memD = len(nx[0])
    numNodes = memD
    halfD = int(memD/2)

    param = {
    'memD': memD,
    'memK': memK,
    'numNodes': numNodes,
    'numStack': 1,
    'biasWeights': numNodes * [0],
    'ptfWeights': numNodes * [1],
    'ptfThresh': [0],    
    'adaptBias': 0,
    'adaptWeights': 0,
    'adaptThresh': 0,
    'scaleTo': 127,
    'printWeights': 0,
    'printThresh': 0,
    'printSample':0,
    'printBias': 0,
    'printIteration': 1, 
    'postAdaptRun': 0,
    }

    #for i in range(halfD):
    #    param['ptfWeights'][i] = 1
    #    param['ptfWeights'][halfD+i] = -1
    
    param['ptfWeights'] = numNodes * [1]
    param['ptfThresh'] = [int(sum(param['ptfWeights'])/2)]
    param['ptfThresh'] = [numNodes]
    param['ptfThresh'] = [1]
    #print(f"Here: {param['ptfWeights']} and {param['ptfThresh']}")
    #return        
    #ptfThresh = [numNodes]    
    
    
    #ptfWeights[0] = numNodes
    #biasWeights = [125, 125, 100, 103, 98, 94, 38, 17]
    #biasWeights[0] = numNodes

    runner = MLRunner(nx, nxxyy, param)        
    posStatsSample = param['numNodes'] * [0.0]

    for iter in range(numIterations):
        print("##################################################")
        print(f"ITERATION {iter}")

        weights = runner.ohm.paramStackMem[0].GetLSBIntsHack()                                                                    
        print(f"       Weights In: {weights}")                                       
        thresh = runner.ohm.paramThreshMem[0].GetLSBIntsHack()                                                                
        print(f"       Thresh In: {thresh}")                                       
        
        runner.Run(param)
        
        weights = runner.ohm.paramStackMem[0].GetLSBIntsHack()                                                                    
        print(f"       Weights Out: {weights}")                                       
        thresh = runner.ohm.paramThreshMem[0].GetLSBIntsHack()                                                                
        print(f"       Thresh Out: {thresh} out of {sum(weights)}")                                       

        thresh = runner.plotResults['thresh']
        
        plt.plot(thresh)
        plt.xlabel('Threshold Index')
        plt.ylabel('Threshold Value')
        plt.title('Threshold Values')
        plt.show()

        #adaptWeights = 0
        #result = runner.ApplyToMap(adaptWeights)
        #PlotMap(result)
        if param['postAdaptRun'] == 1:
            param['adaptThresh'] = 0
            runner.Run(param)
            
            weights = runner.ohm.paramStackMem[0].GetLSBIntsHack()                                                                    
            print(f"       Weights Out: {weights}")                                       
            thresh = runner.ohm.paramThreshMem[0].GetLSBIntsHack()                                                                
            print(f"       Thresh Out: {thresh} out of {sum(weights)}")                                       





test_RUN()