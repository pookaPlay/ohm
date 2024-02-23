from DataReader import DataReader
from DataWriter import DataWriter
from OHM_NET import OHM_NET
from BSMEM import BSMEM

def RunOhmNet():
    
    verbose = 1

    NSteps = 4
    NN = 2      # number of parallel nodes
    K = 8       # target precision
    DK = 7      # input data precision
    WK = 7      # input weight precision

    input = [[7, -2, -6], [7, 0, -3], [1, 3, 5], [-6, 1, 2]]    
    N = len(input)   
    D = len(input[0])
    
    # separate inputs 
    inputs = [[input[ni][di] for ni in range(len(input))] for di in range(len(input[0]))]

    print(inputs)    
    dataMem = [DataReader([inputs[ni]], DK, K) for ni in range(len(inputs))]       
    [dataMem[p].Print() for p in range(len(dataMem))]

    weights = [[0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1]]
    
    assert len(weights) == NN
    assert len(weights[0]) == 2*D

    paramMem = [DataReader([weights[ni]], DK, K) for ni in range(len(weights))]  
    [paramMem[p].Print() for p in range(len(paramMem))]
    
    msbMem = BSMEM(D, K)    
    lsbMem = BSMEM(D, K)
            
    ohm = OHM_NET(dataMem, paramMem, lsbMem, msbMem)
    ohm.Print()

    return

    ti = 0
    print(f"== {ti} ============================")
    lsbOHM.Calc()
        
    for ti in range(NSteps):
        print(f"== {ti+1} ============================")
        
        [dataMem[p].Step() for p in range(len(dataMem))]
        [paramMem[p].Step() for p in range(len(paramMem))]

        [dataMem[p].Print() for p in range(len(dataMem))]
        [paramMem[p].Print() for p in range(len(paramMem))]

        lsbOHM.Calc()
        lsbOHM.Step()

        #output.Step(ohm.Output(), ohm.lsbOut(), ohm.msbOut())            
        #output.Print()
    return

RunOhmNet()
