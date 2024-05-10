from ml.TorchSynData import LoadXor, LoadGaussian
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
    
    display = 1
    #threshSpac = 0.1   # 240
    threshSpac = 0.25   # 64
    #threshSpac = 1.0   # 16
    #threshSpac = 2.0   # 8    
    thresholds = np.arange(-2.0, 2.0, threshSpac).tolist()     
    #thresholds = [0.0]
    print(f"Thresholds @ dim: {thresholds}")
    
    numPoints = 5
        
    x, y, xv, yv, xxyy = LoadXor(numPoints, display)
    #x, y, xv, yv, xxyy = LoadGaussian(numPoints, display)
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
    numStack = 1
    halfD = int(memD/2)    
    
    param = {
    'memD': memD,
    'memK': memK,
    'numNodes': numNodes,
    'numStack': numStack,
    'biasWeights': numNodes * [0],
    'ptfWeights': numStack * [numNodes * [1]],    
    'ptfThresh': numStack * [ 1 * [halfD]],   
    'ptfDeltas': np.zeros([numNodes, numNodes]),     
    'adaptBias': 0,
    'adaptWeights': 0,
    'adaptThresh': 1,
    'adaptThreshCrazy': 0,
    'scaleTo': 127,
    'clipAt': 127,
    'printSample': 0,
    'printParameters': 1,    
    'printIteration': 1, 
    'printMem': -1,  # set this to the sample index to print
    'postAdaptRun': 0,
    'plotThresh': 0,    
    'printTicks' : 0,
    }
   
    #param['ptfWeights'] = numNodes * [1]    
    #param['ptfThresh'] = [int(sum(param['ptfWeights'])/2)]
    #param['ptfThresh'] = [numNodes]
    #param['ptfThresh'] = [1]
    
    #print(f"PTF Weights: {param['ptfWeights']}")
    #print(f"PTF  Thresh: {param['ptfThresh']}")     
    

    runner = MLRunner(nx, nxxyy, param)        
    posStatsSample = param['numNodes'] * [0.0]

    for iter in range(numIterations):
        print("##################################################")
        print(f"ITERATION {iter}")
        
        runner.Run(param)
        
        if param['postAdaptRun'] == 1:
            was1 = param['adaptWeights']
            was2 = param['adaptThresh']
            was3 = param['adaptBias']

            param['adaptWeights'] = 0
            param['adaptThresh'] = 0
            param['adaptBias'] = 0
            runner.Run(param)            

            param['adaptWeights'] = was1
            param['adaptThresh'] = was2
            param['adaptBias'] = was3



"""         thresh = runner.plotResults['thresh']        
        if param['plotThresh'] == 1:
            plt.plot(thresh)
            plt.xlabel('Threshold Index')
            plt.ylabel('Threshold Value')
            plt.title('Threshold Values')
            plt.show()
 """
test_RUN()