from DataReader import DataReader
from DataWriter import DataWriter
from OHM_LSB import OHM_LSB
from OHM_MSB import OHM_MSB
from BSMEM import BSMEM

def RunOhmNet():
    
    verbose = 1

    NN = 2      # number of parallel nodes
    K = 8       # target precision
    DK = 7      # input data precision
    WK = 7      # input weight precision

    input = [[7, -2, -6], [7, 0, -3], [1, 3, 5], [-6, 1, 2]]    
    dataMem = DataReader(input, DK, K)
    N = len(input)   
    D = len(input[0])

    weights = [[0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1]]
    
    assert len(weights) == NN
    assert len(weights[0]) == 2*D

    paramMem = [DataReader([weights[ni]], DK, K) for ni in range(len(weights))]  
    
    msbMem = BSMEM(D, K)    
    lsbMem = BSMEM(D, K)
            
    lsbOHM = OHM_LSB(dataMem, paramMem, msbMem)

    msbOHM = OHM_MSB(msbMem, lsbMem)

    dataMem.Print()
    [paramMem[p].Print() for p in range(len(paramMem))]

    lsbOHM.Print()
    
    
    return

RunOhmNet()
