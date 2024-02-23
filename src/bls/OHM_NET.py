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
        self.outputIndex = list(range(self.N))        

        #self.lsbIndex = list([0] * self.N)
        #self.msbIndex = list([0] * self.N)        

        self.dataInputs = [self.dataMem[self.dataIndex[ni]] for ni in range(self.N)]
        self.paramInputs = [self.paramMem[self.paramIndex[ni]] for ni in range(self.N)]
        
        self.ohmLSB.Reset()        
        
        #lsbOHM = OHM_LSB(dataMem, paramMem, msbMem)
        #msbOHM = OHM_MSB(msbMem, lsbMem)                        
    

    def Calc(self) -> None:

        self.dataInputs = [self.dataMem[self.dataIndex[ni]] for ni in range(self.N)]
        self.paramInputs = [self.paramMem[self.paramIndex[ni]] for ni in range(self.N)]
        self.ohmLSB.Calc(self.dataInputs, self.paramInputs)        
            
    def Step(self) -> None:        
        pass

    def Output(self) -> int:
        return self.net.Output()
                
    def Print(self, prefix="", showInput=1) -> None:        
        print(prefix + f"OHM_NET:")
        