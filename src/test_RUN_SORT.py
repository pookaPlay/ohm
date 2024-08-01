from ml.TorchSynData import LoadXor, LoadGaussian
from ml.TorchSynData import LoadLinear
from ml.TorchSynData import PlotMap
from ml.SortRunner import SortRunner
import torch
import numpy as np
import matplotlib.pyplot as plt

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def test_RUN_SORT():

    numIterations = 1    
    numPoints = 5
    numPermutations = 1

    memK = 8
    scaleTo = 127
    clipAt = 127

    inputDim = 10
    numLayers = 10
    numInputs = 3
    numStack = inputDim

    nx = torch.randn(numPoints, inputDim)    
    dataN = nx.shape[0]
    
    memD = len(nx[0])        
    halfD = int(memD/2)    
    
    print(f"Input  : {memD} wide (D) -> {memK} deep (K)")
    print(f"Network: {numStack} wide (W) -> {numLayers} deep (L)")
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
    'adaptWeights': 0,
    'adaptThresh': 0,
    'adaptThreshCrazy': 0,
    'scaleTo': scaleTo,
    'clipAt': clipAt,
    'printSample': 1,
    'printParameters': 0,    
    'printIteration': 1, 
    'numPermutations': numPermutations,
    'printMem': -1,  # set this to the sample 1index to print
    'postAdaptRun': 0,
    'preAdaptInit': 0,    
    'plotResults': 0,    
    'printTicks' : 0,
    'applyToMap': 0,
    }    

    for i in range(numStack):
        #param['ptfThresh'][i] = [((2*i)%(numInputs*2))+1]        

        for ni in range(numInputs):
            param['biasWeights'][i][numInputs + ni] = 1        
    
    #print(f"BIAS: {param['biasWeights']}")    
    #print(f"PTF Weights: {param['ptfWeights']}")
    #print(f"PTF  Thresh: {param['ptfThresh']}")     
    
    if param['preAdaptInit'] == 1:
        #mam = BatchMAM(nx, nxxyy, param)    
        #param['biasWeights'] = mam.BatchTrainMAM()        
        print(f"param: {param['biasWeights']}")    
    

    runner = SortRunner(nx, param)                    
    
    for iter in range(numIterations):
        print("##################################################")
        print(f"ITERATION {iter}")
        
        results = runner.Run(param)
        
        
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
                

test_RUN_SORT()