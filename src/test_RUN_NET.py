from ml.TorchSynData import LoadXor, LoadGaussian
from ml.TorchSynData import LoadLinear
from ml.TorchSynData import PlotMap
from bls.NetRunner import NetRunner
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
        
    #threshSpac = 0.1   # 240
    threshSpac = 0.25   # 64
    #threshSpac = 1.0   # 16
    #threshSpac = 2.0   # 8    
    #thresholds = np.arange(-2.0, 2.0, threshSpac).tolist()     
    thresholds = [0.0]
    print(f"Thresholds @ dim: {thresholds}")
    
    numPoints = 5
    
    x, y, xv, yv, xxyy = LoadXor(numPoints, 'uniform')
    #x, y, xv, yv, xxyy = LoadGaussian(numPoints)
    #x, y, xxyy = LoadLinear(numPoints)    
    
    nx = ThreshExpand(x, thresholds)        
    nxxyy = ThreshExpand(xxyy, thresholds)        
    
    print(f"Thresh expand: {x.shape} -> {nx.shape}")

    memK = 8
    dataN = nx.shape[0]
    memD = len(nx[0])
    numNodes = memD
    numLayers = 1
    numStack = 4
    halfD = int(memD/2)    
    
    print(f"Input  : {memD} wide (D) -> {memK} deep (K)")
    print(f"Network: {numStack} wide (F) -> {numLayers} deep (L)")

    param = {
    'memD': memD,
    'memK': memK,
    'numNodes': numNodes,
    'numLayers': numLayers,
    'numStack': numStack,
    'biasWeights': numNodes * [ numNodes * [0] ],
    'ptfWeights': numStack * [numNodes * [1]],
    'ptfThresh': numStack * [ [ 1 ] ],     
    'ptfDeltas': np.zeros([numNodes, numNodes]),     
    'adaptBias': 0,
    'adaptWeights': 0,
    'adaptThresh': 0,
    'adaptThreshCrazy': 0,
    'scaleTo': 127,
    'clipAt': 127,
    'printSample': 0,
    'printParameters': 1,    
    'printIteration': 1, 
    'printMem': -1,  # set this to the sample 1index to print
    'postAdaptRun': 0,
    'preAdaptInit': 0,    
    'plotResults': 1,    
    'printTicks' : 0,
    'applyToMap': 0,
    }    

    #param['ptfWeights'] = numNodes * [1]    
    #param['ptfThresh'] = [int(sum(param['ptfWeights'])/2)]
    #param['ptfThresh'] = [numNodes]
    #param['ptfThresh'] = [1]
    
    #print(f"PTF Weights: {param['ptfWeights']}")
    #print(f"PTF  Thresh: {param['ptfThresh']}")     
    if param['preAdaptInit'] == 1:
        #mam = BatchMAM(nx, nxxyy, param)    
        #param['biasWeights'] = mam.BatchTrainMAM()        
        print(f"param: {param['biasWeights']}")    
    

    runner = NetRunner(nx, nxxyy, param)            

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



    if param['plotResults'] == 1:
        
        #min_value = torch.min(runner.input)
        #max_value = torch.max(runner.input)        
        minScale = -150
        maxScale = 150

        offsets = runner.ohm.paramBiasMem[0].GetLSBInts()        
        off = torch.tensor(offsets)        

        plt.scatter(runner.input[ y[:,0] > 0 , 0], runner.input[ y[:,0] > 0 , 1], color='g', marker='o')
        plt.scatter(runner.input[ y[:,0] < 0 , 0], runner.input[ y[:,0] < 0 , 1], color='r', marker='x')

        plt.axvline(x=off[0], color='b', linestyle='--')
        plt.axhline(y=off[1], color='b', linestyle='--')        
        plt.axvline(x=-off[2], color='g', linestyle='--')
        plt.axhline(y=-off[3], color='g', linestyle='--')

        plt.xlim(minScale, maxScale)
        plt.ylim(minScale, maxScale)
        plt.show()
                

test_RUN()