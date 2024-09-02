from ml.ExperimentRunner import SetupExperiment, RunNetwork
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

def RUN_EXPERIMENT_test():

    param = {}
    param['inputDim'] = 100
    param['numPoints'] = 1

    nx, param = SetupExperiment(param)

    param['runMovie'] = 1
    #param['adaptBias'] = 0
    #param['adaptWeights'] = 1
    #param['adaptThresh'] = 1
    #param['adaptThreshType'] = 'pc',        # 'pc' or 'ss'    
    
    RunNetwork(nx, param)
    
                

RUN_EXPERIMENT_test()