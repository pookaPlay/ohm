from ml.ExperimentRunner import SetupExperiment, UpdateParam, RunNetwork
import torch
import numpy as np
import random
import keyboard

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def RUN_EXPERIMENT_test():

    param = {
        'numPoints': 1,
        'inputDim': 10,
        'numIterations' : 1,
        'runMovie': 1,
        'adaptWeights': 1, 
        'adaptThresh' : 1,
    }

    nx, param = SetupExperiment(param)
    
    config1 = {
        'expId': 0,                
        'numInputs': 100,
    }
    param = UpdateParam(param, config1)    
    RunNetwork(nx, param)
    
    #config2 = {        
    #    'expId': 1,                
    #    'numInputs': 5,
    #}
    ##param = UpdateParam(param, config2)    
    RunNetwork(nx, param)

    print("Press any key to continue...")    
    keyboard.read_event()

                

RUN_EXPERIMENT_test()