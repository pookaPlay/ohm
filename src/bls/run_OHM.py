from SingleDataReader import SingleDataReader
from DataWriter import DataWriter
from OHM_WORD import OHM_WORD
from BSMEM import BSMEM

def RunOhmNet():
    
    verbose = 2
    showInputs = 0

    NSteps = 7
    numNodes = 2      
    nodeDim = 2 
    memK = 8
    memD = 8
    
    ohm = OHM_WORD(memD, memK, numNodes, nodeDim)
    
    ohm.RunNSteps(NSteps)

    return

RunOhmNet()
