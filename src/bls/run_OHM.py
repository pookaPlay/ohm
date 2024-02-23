from DataReader import DataReader
from DataWriter import DataWriter
from OHM_LSB import OHM_LSB
from OHM_MSB import OHM_MSB
from BSMEM import BSMEM

def RunOhmNet():
    
    verbose = 1

    NN = 2      # number of nodes
    K = 8       # target precision
    DK = 7      # input data precision
    WK = 7      # input weight precision

    input = [[7, -2, -6], [7, 0, -3], [1, 3, 5], [-6, 1, 2]]    
    dataMem = DataReader(input, DK, K)

    weights = [[0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1]]
    
    assert len(weights) == NN
    paramMem = [DataReader([weights[ni]], DK, K) for ni in range(len(weights))]
    
    N = len(input)   
    D = len(input[0])

    
    msbMem = BSMEM(D, K)    
    lsbMem = BSMEM(D, K)
        
    
    lsbOHM = OHM_LSB(dataMem, paramMem, msbMem)
    lsbOHM.Print()
    #msbOHM = OHM_MSB(lsbMem, msbMem, paramMem)
    
    return

RunOhmNet()
