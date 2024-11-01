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
        'numInputs': 3,
        'numLayers': 1,
        'numIterations' : 2,        
        'numPermutations' : 0,
        'adaptWeights': 0, 
        'adaptThresh' : 0,     
        'adaptBias': 0,
        'adaptThreshType': 'ss',        # 'pc' or 'ss'
        'scaleTo': 127,
        'clipAt': 127,    
        'printSample': 1,
        'printSampleLayer': 1,
        'printIteration': 0,                
        'printParameters': 0,            
        'printTicks' : 0,
        'applyToMap': 0,
        'runMovie': 0,
        'doneClip' : 0,
        'doneClipValue' : 0,               
        }

    nx, param = SetupExperiment(param)
    
    config1 = {
        'expId': 0,                
        'adaptWeights': 0, 
        'adaptThresh' : 0,     
        'adaptBias': 0,
    }
    param1 = UpdateParam(param, config1)    
    RunNetwork(nx, param1)
    
    config2 = {        
        'expId': 1,                
        'adaptWeights': 0, 
        'adaptThresh' : 0,     
        'adaptBias': 1,
    }
    param2 = UpdateParam(param, config2)    
    RunNetwork(nx, param2)
    
    print("Press 'q' to quit...")
    while True:
        if keyboard.is_pressed('q'):
            break

                

RUN_EXPERIMENT_test()