from SingleDataReader import SingleDataReader
from DataWriter import DataWriter
from OHM_WORD import OHM_WORD
from BSMEM import BSMEM

def RunOhmNet():
    
    verbose = 2
    showInputs = 0

    NSteps = 7
    NN = 2      # number of parallel nodes
    K = 8       # target precision
    DK = 8      # input data precision
    WK = 8      # input weight precision
    MD = 8      # memory dimension for lsbmem and msbmem

    
    ohm = OHM_WORD(MD, K, NN)
    
    ohm.RunNSteps(NSteps)

    return

RunOhmNet()
