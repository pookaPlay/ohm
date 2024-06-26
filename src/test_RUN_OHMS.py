from bls.RunOHMS import RunOHMS
from ml.TorchSynData import LoadXor
from ml.TorchSynData import LoadLinear
from ml.TorchSynData import PlotMap
#from ../ml/TorchSynData import LoadXor, LoadLinear, PlotMap

def test_RUN_OHM():

    
    memK = 8
        
    input = [8, -8, 4, -4, 2, -2, 1, -1]
    #input += [-x for x in input]  # Add the negative values

    memD = len(input)
    numNodes = memD
    biasWeights = numNodes * [0]
    ptfWeights = numNodes * [1]
    
        
    ohm = RunOHMS(memD, memK, numNodes, input, biasWeights, ptfWeights)    
    ohm.Run()
    
    #ohm.PrintMem()
    #result = ohm.lsbMem.GetLSBInts()
    #print(result)
    
    
    return

test_RUN_OHM()