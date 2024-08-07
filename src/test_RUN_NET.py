from ml.TorchSynData import LoadXor, LoadGaussian
from ml.TorchSynData import LoadLinear
from ml.TorchSynData import PlotMap
from ml.MLRunner import MLRunner
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
   
    #mirrored: ND = 2 * D * len(thresh)
    ND = D * len(thresh)
    #print(x)
    #print(f"Expanding {N} samples to {ND} dimensions")
    nx = torch.zeros([N, ND])
    for t in range(len(thresh)):
        nx[:, t*D:(t+1)*D] = x - thresh[t]
    
    #mirrored
    #for t in range(len(thresh)):        
    #    nx[:, (t+len(thresh))*D:(t+len(thresh)+1)*D] = thresh[t] - x
    
    return nx

def test_RUN():

    numIterations = 1
        
    #threshSpac = 0.1   # 240
    threshSpac = 0.25   # 64
    #threshSpac = 1.0   # 16
    #threshSpac = 2.0   # 8    
    #thresholds = np.arange(-2.0, 2.0, threshSpac).tolist()     
    thresholds = [0.0]
    thresholds = [-0.5, 0.5]
    print(f"Thresholds @ dim: {thresholds}")
    
    numPoints = 5
    
    x, y, xv, yv, xxyy = LoadXor(numPoints, 'uniform', 1)
    #x, y, xv, yv, xxyy = LoadGaussian(numPoints)
    #x, y, xxyy = LoadLinear(numPoints)    
    
    nx = ThreshExpand(x, thresholds)        
    nxxyy = ThreshExpand(xxyy, thresholds)        
    
    print(f"Thresh expand: {x.shape} -> {nx.shape}")

    memK = 8
    dataN = nx.shape[0]
    memD = len(nx[0])
    
    numLayers = 1
    numInputs = 3
    numStack = 4

    halfD = int(memD/2)    
    
    print(f"Input  : {memD} wide (D) -> {memK} deep (K)")
    print(f"Network: {numStack} wide (W) -> {numLayers} deep (L)")
    print(f"Fanin (F): {numInputs}")

    param = {
    'memD': memD,
    'memK': memK,    
    'numLayers': numLayers,
    'numInputs': numInputs,
    'numStack': numStack,
    'biasWeights': numStack * [ (numInputs*2) * [0] ],
    'ptfWeights': numStack * [(numInputs*2) * [1]],
    'ptfThresh': numStack * [ [ numInputs ] ],         
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
    'plotResults': 0,    
    'printTicks' : 0,
    'applyToMap': 0,
    }    

    for i in range(numStack):
        param['ptfThresh'][i] = [i+1]
        for ni in range(numInputs):
            param['biasWeights'][i][numInputs + ni] = 1        
        #param['ptfThresh'][i] = [halfD]
        #param['ptfThresh'][i] = [numNodes]  # min
    
    #print(f"BIAS: {param['biasWeights']}")    
    #print(f"PTF Weights: {param['ptfWeights']}")
    #print(f"PTF  Thresh: {param['ptfThresh']}")     
    
    if param['preAdaptInit'] == 1:
        #mam = BatchMAM(nx, nxxyy, param)    
        #param['biasWeights'] = mam.BatchTrainMAM()        
        print(f"param: {param['biasWeights']}")    
    

    runner = MLRunner(nx, nxxyy, param)            
    print(runner.input)
    
    for iter in range(numIterations):
        print("##################################################")
        print(f"ITERATION {iter}")
        
        results = runner.Run(param)
        
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

    
    print(results)

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