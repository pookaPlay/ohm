from ml.TorchSynData import LoadXor, LoadGaussian
from ml.TorchSynData import LoadLinear
from ml.TorchSynData import PlotMap
from ml.SortRunner import SortRunner
from ml.MLRunner import MLRunner
import torch
import numpy as np
import matplotlib.pyplot as plt


def SetupExperiment(param):

    assert 'numPoints' in param
    assert 'inputDim' in param
    assert 'expType' in param

    if param['expType'] == 'sort':
        nx = torch.randn(param['numPoints'], param['inputDim'])
        ny = torch.zeros(param['numPoints'], 1)
    
    elif param['expType'] == 'xor':
        #param['numPoints']
        nx, ny, vx, vy, mapx = LoadXor(1, 'uniform', 1)
    
    assert(nx.shape[0] == param['numPoints'])

    if 'numLayers' not in param:
        param['numLayers'] = 20
    if 'numInputs' not in param:
        param['numInputs'] = 10
    if 'numIterations' not in param:
        param['numIterations'] = 1
    if 'numPermutations' not in param:
        param['numPermutations'] = 1  # set to 0 to keep permutation constant
    if 'memK' not in param:
        param['memK'] = 8
    if 'memD' not in param:
        param['memD'] = param['inputDim']
    if 'numStack' not in param:
        param['numStack'] = param['inputDim']
    if 'biasWeights' not in param:
        param['biasWeights'] = param['numStack'] * [(param['numInputs'] * 2) * [0]]
    if 'ptfWeights' not in param:
        param['ptfWeights'] = param['numStack'] * [(param['numInputs'] * 2) * [1]]
    if 'ptfThresh' not in param:
        param['ptfThresh'] = param['numStack'] * [[param['numInputs']]]
    if 'adaptBias' not in param:
        param['adaptBias'] = 0
    if 'adaptWeights' not in param:
        param['adaptWeights'] = 1
    if 'adaptThresh' not in param:
        param['adaptThresh'] = 1
    if 'adaptThreshType' not in param:
        param['adaptThreshType'] = 'pc'  # 'pc' or 'ss'
    if 'adaptThreshCrazy' not in param:
        param['adaptThreshCrazy'] = 0
    if 'scaleTo' not in param:
        param['scaleTo'] = 127
    if 'clipAt' not in param:
        param['clipAt'] = 127
    if 'printSample' not in param:
        param['printSample'] = 1
    if 'printParameters' not in param:
        param['printParameters'] = 0
    if 'printIteration' not in param:
        param['printIteration'] = 1
    if 'printMem' not in param:
        param['printMem'] = -1  # set this to the sample 1index to print
    if 'postAdaptRun' not in param:
        param['postAdaptRun'] = 0
    if 'preAdaptInit' not in param:
        param['preAdaptInit'] = 0
    if 'plotResults' not in param:
        param['plotResults'] = 0
    if 'printTicks' not in param:
        param['printTicks'] = 0
    if 'applyToMap' not in param:
        param['applyToMap'] = 0
    if 'runMovie' not in param:
        param['runMovie'] = 1
    if 'doneClip' not in param:
        param['doneClip'] = 0        
    if 'doneClipValue' not in param:
        param['doneClipValue'] = 0        
        
    for i in range(param['numStack']):
        #param['ptfThresh'][i] = [((i)%(numInputs*2))+1]        
        for ni in range(param['numInputs']):
            param['biasWeights'][i][param['numInputs'] + ni] = 1        
    
    return nx, ny, param

def UpdateParam(param, config):
    for key, value in config.items():
        param[key] = value
    
    param['biasWeights'] = param['numStack'] * [(param['numInputs'] * 2) * [0]]
    param['ptfWeights'] = param['numStack'] * [(param['numInputs'] * 2) * [1]]
    param['ptfThresh'] = param['numStack'] * [[param['numInputs']]]

    for i in range(param['numStack']):
        #param['ptfThresh'][i] = [((i)%(numInputs*2))+1]        
        for ni in range(param['numInputs']):
            param['biasWeights'][i][param['numInputs'] + ni] = 1        

    return param


def RunNetwork(nx, ny, param):

    print(f"Input  : {param['memD']} dimension (D) -> {param['memK']} precision (K)")
    print(f"Network: {param['numStack']} wide (W) -> {param['numLayers']} long (L)")
    print(f"Fanin (F): {param['numInputs']} ({param['numInputs']*2} with mirroring)")
    
    
    #print(f"BIAS: {param['biasWeights']}")    
    #print(f"PTF Weights: {param['ptfWeights']}")
    #print(f"PTF  Thresh: {param['ptfThresh']}")     
    
    if param['preAdaptInit'] == 1:
        #mam = BatchMAM(nx, nxxyy, param)    
        #param['biasWeights'] = mam.BatchTrainMAM()        
        print(f"param: {param['biasWeights']}")    
    
    print("##################################################")
    print("##################################################")
    if param['expType'] == 'sort':
        runner = SortRunner(nx, param)                    
    elif param['expType'] == 'xor':
        runner = MLRunner(nx, ny, param)
    
    for iter in range(param['numIterations']):
        print("##################################################")
        print(f"ITERATION {iter}")
        
        runner.Run(param)
        
        runner.ohm.PrintParameters()
        

####################################################################################################
# if param['plotResults'] == 1:
    
#     #min_value = torch.min(runner.input)
#     #max_value = torch.max(runner.input)        
#     minScale = -150
#     maxScale = 150

#     offsets = runner.ohm.paramBiasMem[0].GetLSBInts()        
#     off = torch.tensor(offsets)        

#     plt.scatter(runner.input[ y[:,0] > 0 , 0], runner.input[ y[:,0] > 0 , 1], color='g', marker='o')
#     plt.scatter(runner.input[ y[:,0] < 0 , 0], runner.input[ y[:,0] < 0 , 1], color='r', marker='x')

#     plt.axvline(x=off[0], color='b', linestyle='--')
#     plt.axhline(y=off[1], color='b', linestyle='--')        
#     plt.axvline(x=-off[2], color='g', linestyle='--')
#     plt.axhline(y=-off[3], color='g', linestyle='--')

#     plt.xlim(minScale, maxScale)
#     plt.ylim(minScale, maxScale)
#     plt.show()
                
