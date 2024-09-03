from ml.TorchSynData import LoadXor, LoadGaussian
from ml.TorchSynData import LoadLinear
from ml.TorchSynData import PlotMap
from ml.SortRunner import SortRunner
import torch
import numpy as np
import matplotlib.pyplot as plt


def SetupExperiment(param):

    assert 'inputDim'  in param
    assert 'numPoints' in param

    nx = torch.randn(param['numPoints'], param['inputDim'])    
    assert(nx.shape[0] == param['numPoints'])

    param['numIterations'] = 1    
    param['numPermutations'] = 1  # set to 0 to keep permutation constant    
    param['memK'] = 8
    param['memD'] = param['inputDim']
    param['numLayers'] = 20
    param['numInputs'] = 10
    param['numStack'] = param['inputDim']
    param['biasWeights'] = param['numStack'] * [(param['numInputs'] * 2) * [0]]
    param['ptfWeights'] = param['numStack'] * [(param['numInputs'] * 2) * [1]]
    param['ptfThresh'] = param['numStack'] * [[param['numInputs']]]
    param['adaptBias'] = 0
    param['adaptWeights'] = 1
    param['adaptThresh'] = 1
    param['adaptThreshType'] = 'pc'  # 'pc' or 'ss'
    param['adaptThreshCrazy'] = 0
    param['scaleTo'] = 127
    param['clipAt'] = 127
    param['printSample'] = 1
    param['printParameters'] = 0
    param['printIteration'] = 1
    param['printMem'] = -1  # set this to the sample 1index to print
    param['postAdaptRun'] = 0
    param['preAdaptInit'] = 0
    param['plotResults'] = 0
    param['printTicks'] = 0
    param['applyToMap'] = 0
    param['runMovie'] = 1

    for i in range(param['numStack']):
        #param['ptfThresh'][i] = [((i)%(numInputs*2))+1]        
        for ni in range(param['numInputs']):
            param['biasWeights'][i][param['numInputs'] + ni] = 1        
    
    return nx, param

def UpdateParam(param, config):
    for key, value in config.items():
        param[key] = value
    return param


def RunNetwork(nx, param):

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
    runner = SortRunner(nx, param)                    
    
    for iter in range(param['numIterations']):
        print("##################################################")
        print(f"ITERATION {iter}")
        
        runner.Run(param)
        
        #runner.ohm.PrintParameters()
        

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
                
