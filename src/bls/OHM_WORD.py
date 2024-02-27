##############################
## This has the connectivity 

from BSMEM import BSMEM
from RDMEM import RDMEM
from OHM_LSB import OHM_LSB
from OHM_MSB import OHM_MSB

class OHM_WORD:

    def __init__(self, memD, memK, numNodes):
    
        self.NN = numNodes      # number of parallel nodes
        self.D = memD
        self.K = memK
        self.lsbMem = [BSMEM(self.D, self.K), BSMEM(self.D, self.K)]
        self.msbMem = [BSMEM(self.D, self.K), BSMEM(self.D, self.K)]        
        self.memState = 0 

        input = [7, -2, -6]        
        self.dataMem = RDMEM(input, self.K, self.K)

        weights = self.NN * [1]
        self.paramMem = RDMEM(weights, self.K, self.K)

        self.ohmLSB = OHM_LSB(self.NN, self.D)        
        self.denseLSBOut = list(self.D * [0])
        self.lsbInIndex = list(range(self.D))
        self.lsbOutIndex = list(range(self.D))

        self.ohmMSB = OHM_MSB(self.NN, self.D)        
        self.denseMSBOut = list(self.D * [0])
        self.msbInIndex = list(range(self.D))
        self.msbOutIndex = list(range(self.D))
        

        self.Reset()

    def Reset(self) -> None:

        [mem.Reset() for mem in self.lsbMem]
        [mem.Reset() for mem in self.msbMem]
        self.memState = 0 

        self.dataMem.Reset()
        self.paramMem.Reset()
        self.ohmLSB.Reset()        
        self.ohmMSB.Reset()        
        self.denseLSBOut = list(self.D * [0])
        self.denseMSBOut = list(self.D * [0])

        
    def RunNSteps(self, NSteps) -> None:
            
            # Assumes all memories are ready 
            ## T = 0 
            ti = 0
            print(f"===============================================")
            print(f"== {ti} =======================================")
            self.ohmLSB.Calc(self.dataMem, self.paramMem)
            self.denseLSBOut = self.ohmLSB.Output()                                              

            #self.Print("", 2)                    

            self.lsbMem[self.memState].Step(self.denseLSBOut)            
            #[mem.Step() for mem in self.msbMem]

            [mem.Print() for mem in self.lsbMem]
            #[mem.Print() for mem in self.msbMem]

            for ti in range(NSteps):
                print(f"== {ti+1} ============================")
    
                self.dataMem.Step()
                self.paramMem.Step()
            
                self.ohmLSB.Calc(self.dataMem, self.paramMem)
                self.denseLSBOut = self.ohmLSB.Output()                      
                #self.ohmMSB.Calc(self.dataMem, self.paramMem)                                

                self.lsbMem[self.memState].Step(self.denseLSBOut)
                
            
    def Step(self) -> None:
        self.ohmLSB.Step()        
        #self.ohmMSB.Step()
            
        return 

    def MSBOutputPass(self):
        
        self.denseOutput = list(self.msbMem.D * [0])
        sparseOutput = self.ohmMSB.Output()        
        
        for ni in range(len(sparseOutput)):
            self.lsbDenseOut[self.lsbIndex[ni]] = sparseOutput[ni]
        #print(f"OHM_NET: Output({self.denseOutput})")
        return self.lsbDenseOut

    def Print(self, prefix="", showInput=1) -> None:        
        print(prefix + f"OHM_WORD:")
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