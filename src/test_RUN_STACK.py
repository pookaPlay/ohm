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

def test_RUN():

    nx = [[0, 0, 0]]
    
    
    memK = 8
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
    'printSample': 1,
    'printParameters': 1,    
    'printIteration': 1, 
    'printMem': -1,  # set this to the sample index to print
    'postAdaptRun': 0,
    'plotResults': 1,    
    'printTicks' : 1,
    }
   
    #param['ptfWeights'] = numNodes * [1]    
    #param['ptfThresh'] = [int(sum(param['ptfWeights'])/2)]
    #param['ptfThresh'] = [numNodes]
    #param['ptfThresh'] = [1]
    
    #print(f"PTF Weights: {param['ptfWeights']}")
    #print(f"PTF  Thresh: {param['ptfThresh']}")     
    

    runner = MLRunner(nx, nx, param)        
    runner.Run(param)
    

test_RUN()