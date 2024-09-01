from ml.TorchSynData import LoadXor, LoadGaussian
from ml.TorchSynData import LoadLinear
from ml.TorchSynData import PlotMap
from ml.SortRunner import SortRunner
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def test_RUN_NETWORK():

    numIterations = 1
    numPoints = 1
    numPermutations = 1

    memK = 8
    scaleTo = 127
    clipAt = 127

    inputDim = 100
    numLayers = 20
    numInputs = 10
    numStack = inputDim
    #numStack = 1

    nx = torch.randn(numPoints, inputDim)    
    #nx = torch.zeros(numPoints, inputDim)    
    #nx = torch.ones(numPoints, inputDim)*127    
    dataN = nx.shape[0]
    #print(f"DataN: {dataN}: {nx}")
    
    memD = len(nx[0])        
    halfD = int(memD/2)    
    
    print(f"Input  : {memD} dimension (D) -> {memK} precision (K)")
    print(f"Network: {numStack} wide (W) -> {numLayers} long (L)")
    print(f"Fanin (F): {numInputs}")
    
    # need to track precision and bounds! 
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
    'adaptWeights': 1,
    'adaptThresh': 1,
    'adaptThreshType': 'pc',        # 'pc' or 'ss'
    'adaptThreshCrazy': 0,
    'scaleTo': scaleTo,
    'clipAt': clipAt,    
    'printSample': 1,
    'printParameters': 0,    
    'printIteration': 1, 
    'numPermutations': numPermutations, # set to 0 to keep permutation constant
    'printMem': -1,  # set this to the sample 1index to print
    'postAdaptRun': 0,
    'preAdaptInit': 0,    
    'plotResults': 0,    
    'printTicks' : 0,
    'applyToMap': 0,
    'runMovie': 0,
    }    

    for i in range(numStack):
        #param['ptfThresh'][i] = [((i)%(numInputs*2))+1]        
        for ni in range(numInputs):
            param['biasWeights'][i][numInputs + ni] = 1        
    
    #print(f"BIAS: {param['biasWeights']}")    
    #print(f"PTF Weights: {param['ptfWeights']}")
    #print(f"PTF  Thresh: {param['ptfThresh']}")     
    
    if param['preAdaptInit'] == 1:
        #mam = BatchMAM(nx, nxxyy, param)    
        #param['biasWeights'] = mam.BatchTrainMAM()        
        print(f"param: {param['biasWeights']}")    
    
    print("##################################################")
    print("##################################################")
    runner = SortRunner(nx, param)                    
    
    for iter in range(numIterations):
        print("##################################################")
        print(f"ITERATION {iter}")
        
        results = runner.Run(param)
        
        #runner.ohm.PrintParameters()
        
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
                

test_RUN_NETWORK()