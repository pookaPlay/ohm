from ml.TorchSynData import LoadXor
from ml.TorchSynData import LoadLinear
from ml.TorchSynData import PlotMap
from bls.RunOHMS import RunOHMS
from bls.MLRunner import MLRunner
from bls.BatchMAM import BatchMAM
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

def test_RUN_AM():

    numIterations = 1
    
    display = 0
    threshSpac = 0.25   # 64
    threshSpac = 1.0   # 16
    thresholds = [0.0]
    print(f"Thresholds @ dim: {thresholds}")
    numPoints = 5
        
    #x, y, xxyy = LoadLinear(numPoints, display)
    x, y, xv, yv, xxyy = LoadXor(numPoints, 'uniform', 1)
            
    nx = ThreshExpand(x, thresholds)        
    nxxyy = ThreshExpand(xxyy, thresholds)        

    print(f"Thresh expand: {x.shape} -> {nx.shape}")

    memK = 8
    #input = [8, -8, 4, -4, 2, -2, 1, -1]
    #input += [-x for x in input]  # Add the negative values
    dataN = nx.shape[0]
    memD = len(nx[0])
    numNodes = memD
    numStack = memD
    halfD = int(memD/2)

    param = {
    'memD': memD,
    'memK': memK,
    'numNodes': numNodes,
    'numStack': numStack,
    'biasWeights': numNodes * [0],
    'ptfWeights': numStack * [numNodes * [1]],
    'ptfThresh': numStack * [ [ 1 ] ],    
    'adaptBias': 0,
    'adaptWeights': 0,
    'adaptThresh': 0,
    'adaptThreshCrazy': 0,
    'scaleTo': 127,
    'clipAt': 127,
    'printSample':0,
    'printParameters': 1,    
    'printIteration': 1, 
    'printMem': 0,
    'printTicks': 0,
    'postAdaptRun': 0,
    'applyToMap': 1,
    'plotThresh': 0,
    }   

    for i in range(len(param['ptfThresh'])):
        param['ptfThresh'][i] = [i+1]

    #runner = MLRunner(nx, nxxyy, param)        
    posStatsSample = param['numNodes'] * [0.0]

    mam = BatchMAM(nx, nxxyy, param)    
    mam.BatchTrainMAM()
    mam.BatchTestMAM()

    return
    for iter in range(numIterations):
        print("##################################################")
        print(f"ITERATION {iter}")

        #weights = runner.ohm.paramStackMem[0].GetLSBIntsHack()                                                                    
        #print(f"       Weights In: {weights}")                                       
        #thresh = runner.ohm.paramThreshMem[0].GetLSBIntsHack()                                                                
        #print(f"       Thresh In: {thresh}")                                       
        
        runner.Run(param)
        
        # weights = runner.ohm.paramStackMem[0].GetLSBIntsHack()                                                                    
        # print(f"       Weights Out: {weights}")                                       
        # thresh = runner.ohm.paramThreshMem[0].GetLSBIntsHack()                                                                
        # print(f"       Thresh Out: {thresh} out of {sum(weights)}")                                       

        thresh = runner.plotResults['thresh']
        
        if param['plotThresh'] == 1:
            plt.plot(thresh)
            plt.xlabel('Threshold Index')
            plt.ylabel('Threshold Value')
            plt.title('Threshold Values')
            plt.show()

        if param['applyToMap'] == 1:
            adaptWeights = 0
            result = runner.ApplyToMap(adaptWeights)
            PlotMap(result)

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




test_RUN_AM()