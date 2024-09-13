from ml.TorchSynData import LoadXor, LoadGaussian
from ml.TorchSynData import LoadLinear
from ml.TorchSynData import PlotMap
from ml.ScaleData import ScaleData
from bls.OHM_NETWORK import OHM_NETWORK
from bls.OHM_PROBE import OHM_PROBE
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
    
    #first = [0, 0, 0, 0, 0, 0, 0]
    #first = [1, 2, 3, 0, -1, -2, -3]    
    #first = [1, 2, 3, 4, -1, -2, -3]    
    first = [-65, -32, -28, -68, -4, -26, -41, -68, -30, 13, -33, -11, -9, 71, 10, 20, -30, -49, 30, 8, 37, 10, -28, 34, 47, -7, -95, -61, 3, -26, -34, -6, 80, -3, 7, -33, 8, 2, 6, -20, -5, 12, -7, -1, 100, -44, 67, -27, -39, 23, 3, -19, 32, 19, 49, 85, 6, 40, -8, -27, 65, -37, -127, -32, -1, -43, -25, -43, -9, 1, 71, 1, -30, -8, -42, -35, -20, -24, 17, -42, -21, 4, -28, 31, 59, -34, -39, 47, 26, 8, -109, 36, -89, -26, -63, -48, -4, 15, -45, -37]    
    memD = len(first)        
    halfD = int(memD/2)    

    memK = 8
    scaleTo = 127
    clipAt = 127

    inputDim = memD
    numLayers = 1
    numInputs = 10
    numStack = inputDim    
    
    print(f"Input  : {memD} dimension (D) -> {memK} precision (K)")
    print(f"Network: {numStack} wide (W) -> {numLayers} long (L)")
    print(f"Fanin (F): {numInputs}")
    
    # need to track precision and bounds! 
    param = {
    'memK': memK,
    'memD': memD,        
    'numLayers': numLayers,
    'numInputs': numInputs,
    'numStack': numStack,
    'biasWeights': numStack * [ (numInputs*2) * [0] ],
    'ptfWeights': numStack * [(numInputs*2) * [1]],
    'ptfThresh': numStack * [ [ numInputs ] ],         
    'adaptBias': 0,
    'adaptWeights': 0,
    'adaptThresh': 0,
    'adaptThreshType': 'pc',        # 'pc' or 'ss'
    'adaptThreshCrazy': 0,
    'scaleTo': 127,
    'clipAt': 127,    
    'printSample': 0,
    'printParameters': 0,    
    'printIteration': 1,     
    'numPermutations' : 0,
    'printMem': -1,  # set this to the sample 1index to print
    'postAdaptRun': 0,
    'preAdaptInit': 0,    
    'plotResults': 0,    
    'printTicks' : 0,
    'applyToMap': 0,
    'runMovie': 1,
    'doneClip' : 0,
    'doneClipValue' : 0,
    }    

    for i in range(numStack):
        #param['ptfThresh'][i] = [((i)%(numInputs*2))+1]        
        for ni in range(numInputs):
            param['biasWeights'][i][numInputs + ni] = 1        
            #pass

    ohm = OHM_NETWORK(first, param)
    probe = OHM_PROBE(param, ohm)

    print(f"IN : {first}")
    results = ohm.Run(first, ni, param)
    print(f"OUT: {results}")
    print(param)
    #probe.AnalyzeRun(0, 0)    
                

test_RUN_NETWORK()