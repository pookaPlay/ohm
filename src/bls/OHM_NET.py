##############################
## This has the connectivity 

from BSMEM import BSMEM
from OHM_LSB import OHM_LSB
from OHM_MSB import OHM_MSB
from DataReader import DataReader

class OHM_NET:

    def __init__(self, dataMem: list[DataReader], 
                 paramMem : list[DataReader],
                 lsbMem : BSMEM,
                 msbMem : BSMEM,
                 N = 2):
    
        self.N = N      # number of parallel nodes
        self.lsbMem = lsbMem
        self.msbMem = msbMem
        self.dataMem = dataMem
        self.paramMem = paramMem

        self.ohmLSB = OHM_LSB(self.N)        
        self.ohmMSB = OHM_MSB(self.N)

        self.Reset()

    def Reset(self) -> None:
            
        # Connectivity
        self.dataIndex = list(range(self.N))
        self.paramIndex = list(range(self.N))
        self.dataInputs = list([0] * self.N)
        self.paramInputs = list([0] * self.N)

        self.msbIndex = list(range(self.N))        
        self.lsbIndex = list(range(self.N))              
        self.msbDenseOut = list(self.msbMem.D * [0])
        self.lsbDenseOut = list(self.lsbMem.D * [0])        
        
        self.ohmLSB.Reset()        
        self.ohmMSB.Reset()        
        

    def Calc(self) -> None:

        self.dataInputs = [self.dataMem[self.dataIndex[ni]].Output() for ni in range(self.N)]
        self.paramInputs = [self.paramMem[self.paramIndex[ni]].Output() for ni in range(self.N)]
        self.ohmLSB.Calc(self.dataInputs, self.paramInputs)        

        self.msbInputs = [self.msbMem.GetOutput(self.msbIndex[ni]) for ni in range(self.N)]
        self.ohmMSB.Calc(self.msbInputs)
            
    def Step(self) -> None:        
        self.ohmLSB.Step()        
        self.ohmMSB.Step()

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
