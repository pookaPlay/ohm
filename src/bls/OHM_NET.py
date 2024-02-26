##############################
## This has the connectivity 

from BSMEM import BSMEM
from RDMEM import RDMEM
from OHM_LSB import OHM_LSB
from OHM_MSB import OHM_MSB

class OHM_NET:

    def __init__(self, memD, memK, numNodes):
    
        self.NN = numNodes      # number of parallel nodes
        self.D = memD
        self.K = memK
        self.lsbMem = [BSMEM(self.D, self.K), BSMEM(self.D, self.K)]
        self.msbMem = [BSMEM(self.D, self.K), BSMEM(self.D, self.K)]

        #input = [[7, -2, -6], [7, 0, -3], [1, 3, 5], [-6, 1, 2]]    
        input = [[7, 7, 1, -6], [-2, 0, 3, 1], [-6, -3, 5, 2]]
        self.dataMem = RDMEM(input, self.K, self.K)

        weights = self.NN * [1]
        self.paramMem = RDMEM(weights, self.K, self.K)

        self.ohmLSB = OHM_LSB(self.NN, self.D)        
        self.ohmMSB = OHM_MSB(self.NN, self.D)

        self.Reset()

    def Reset(self) -> None:

        [mem.Reset() for mem in self.lsbMem]
        [mem.Reset() for mem in self.msbMem]
        self.dataMem.Reset()
        self.paramMem.Reset()
        self.ohmLSB.Reset()        
        self.ohmMSB.Reset()        
        

    def Calc(self) -> None:
        print(f"OHM_NET: Calc")
        self.ohmLSB.Calc(self.dataMem, self.paramMem)        
        #self.msbInputs = [self.msbMem.GetOutput(self.msbIndex[ni]) for ni in range(self.N)]
        #self.ohmMSB.Calc(self.msbInputs)
            
    def Step(self) -> None:        
        self.ohmLSB.Step()        
        #self.ohmMSB.Step()

    def LSBOutputPass(self):
        
        self.msbDenseOut = list(self.msbMem.D * [0])
        sparseOutput = self.ohmLSB.Output()        
        
        for ni in range(len(sparseOutput)):
            self.msbDenseOut[self.msbIndex[ni]] = sparseOutput[ni]
        #print(f"OHM_NET: Output({self.denseOutput})")
        return self.msbDenseOut

    def MSBOutputPass(self):
        
        self.denseOutput = list(self.msbMem.D * [0])
        sparseOutput = self.ohmMSB.Output()        
        
        for ni in range(len(sparseOutput)):
            self.lsbDenseOut[self.lsbIndex[ni]] = sparseOutput[ni]
        #print(f"OHM_NET: Output({self.denseOutput})")
        return self.lsbDenseOut

    def Print(self, prefix="", showInput=1) -> None:        
        if showInput > 0:
            print(prefix + f"OHM_NET:")
            if showInput > 1:
                print(prefix + f"  DataMemIndex({self.dataIndex}) ParamMemIndex({self.paramIndex}) Output({self.msbIndex})")
                print(prefix + f"  DataMemValue({self.dataInputs}) ParamMemValue({self.paramInputs}) Output({self.msbIndex})")

        self.ohmLSB.Print(prefix + "  ", showInput)
        self.ohmMSB.Print(prefix + "  ", showInput)

""" 
    ohm.Calc()
    ohm.Print("", 2)
            
    msbMem.Step(ohm.LSBOutputPass())
    lsbMem.Step(ohm.MSBOutputPass())        

    ohm.Step()

    for ti in range(NSteps):
        print(f"== {ti+1} ============================")

        [dataMem[p].Step() for p in range(len(dataMem))]
        [paramMem[p].Step() for p in range(len(paramMem))]

        if showInputs > 1: 
            print(f"DATA")
            [dataMem[p].Print() for p in range(len(dataMem))]
        if showInputs > 1:
            print(f"PARAMS")
            [paramMem[p].Print() for p in range(len(paramMem))]

        ohm.Calc()
        ohm.Print("", 2)                    
        
        msbMem.Step(ohm.LSBOutputPass())
        lsbMem.Step(ohm.MSBOutputPass())        
        ohm.Step()

        #msbMem.Print("  ")
        #result = msbMem.GetInts()
        #print(f"RESULT: {result}")

        

    
    msbMem.Print("MSB")
    msbResult = msbMem.GetInts()
    print(f"RESULT: {msbResult}")

    lsbMem.Print("LSB")
    lsbResult = lsbMem.GetInts()
    print(f"RESULT: {lsbResult}") """