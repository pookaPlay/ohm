from SingleDataReader import SingleDataReader
from DataWriter import DataWriter
from OHM_NET import OHM_NET
from BSMEM import BSMEM

def RunOhmNet():
    
    verbose = 2

    NSteps = 4
    NN = 2      # number of parallel nodes
    K = 8       # target precision
    DK = 7      # input data precision
    WK = 7      # input weight precision

    #input = [[7, -2, -6], [7, 0, -3], [1, 3, 5], [-6, 1, 2]]    
    input = [[7, 7, 1, -6], [-2, 0, 3, 1], [-6, -3, 5, 2]]
    D = len(input)   
    
    dataMem = [SingleDataReader(input[ni], DK, K) for ni in range(D)]
    
    [dataMem[p].Print() for p in range(len(dataMem))]

    weights = NN * [0]
    paramMem = [SingleDataReader(weights[ni], DK, K) for ni in range(len(weights))]
    [paramMem[p].Print() for p in range(len(paramMem))]
    
    msbMem = BSMEM(D, K)    
    lsbMem = BSMEM(D, K)

    ohm = OHM_NET(dataMem, paramMem, lsbMem, msbMem)

    ti = 0
    print(f"== {ti} ============================")

    ohm.Calc()
    ohm.Print("", 2)
        
    for ti in range(NSteps):
        print(f"== {ti+1} ============================")

        [dataMem[p].Step() for p in range(len(dataMem))]
        [paramMem[p].Step() for p in range(len(paramMem))]

        print(f"DATA")
        [dataMem[p].Print() for p in range(len(dataMem))]
        print(f"PARAMS")
        [paramMem[p].Print() for p in range(len(paramMem))]

        ohm.Calc()
        ohm.Print("", 2)    
        ohm.Step()

        #output.Step(ohm.Output(), ohm.lsbOut(), ohm.msbOut())            
        #output.Print()
    return

RunOhmNet()
