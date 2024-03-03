from SingleDataReader import SingleDataReader
from DataWriter import DataWriter
from OHM_WORD import OHM_WORD
from BSMEM import BSMEM

def RunOhmNet():
    
    verbose = 2
    showInputs = 0
    
    numNodes = 3
    nodeDim = 3
    memK = 8
    memD = 8

    input = [1, 2, 3]
    weights = numNodes * [5]
    ptf = "max"
    #input = [[7, -2, -6], [7, 0, -3], [1, 3, 5], [-6, 1, 2]]
    #input = [[7, 7, 1, -6], [-2, 0, 3, 1], [-6, -3, 5, 2]]
    ohm = OHM_WORD(memD, memK, numNodes, nodeDim, input, weights, ptf)
    
    ohm.RunNStep(3)
    result = ohm.GetLSBRead()
    print(result)
    return

RunOhmNet()
