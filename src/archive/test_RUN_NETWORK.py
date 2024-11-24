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
from ml.ExperimentRunner import UpdateParam

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def test_RUN_NETWORK(first, config):

    numIterations = 1
    numPoints = 1
    numPermutations = 1
    
    memD = len(first)        
    halfD = int(memD/2)    

    memK = 8
    scaleTo = 127
    clipAt = 127

    inputDim = memD
    numLayers = 1
    numInputs = 3
    numStack = 1    
    
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
    'printSample': 1,
    'printParameters': 1,    
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

    param = UpdateParam(param, config)  

    ohm = OHM_NETWORK(first, param)
    probe = OHM_PROBE(param, ohm)

    print(f"IN : {first}")
    ohm.PrintParameters()
    results = ohm.Run(first, ni, param)
    print(f"OUT: {results}")
    ohm.PrintParameters()
    
    #probe.AnalyzeRun(0, 0)    
                

def test_RUN_ALL():
    first = [1, 2, 3]
    config = {
        'expId': 0,                
        'adaptWeights': 1, 
        'adaptThresh' : 1,     
        'adaptBias': 1,
    }

    test_RUN_NETWORK(first, config)



test_RUN_ALL()