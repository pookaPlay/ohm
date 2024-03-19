from OHM import OHM
from BSMEM import BSMEM
from DataReader import DataReader
from ADD import ADD

class OHM_ADDER_CHAN:

    def __init__(self,  numInputs, memD) -> None:        
        self.numInputs = numInputs
        self.numOutputs = numInputs
        self.memD = memD
        
        self.adders = [ADD() for _ in range(self.numInputs)]        

        self.inIndexA = list(range(self.numInputs))
        self.inIndexB = list(range(self.numInputs))
        self.outIndex = list(range(self.numOutputs))
            

    def Reset(self) -> None:        
        for ai in range(len(self.adders)):
            self.adders[ai].Reset()
                        
    def Output(self):        
        return self.denseOut

    def Calc(self, memA, memB, lsb=0) -> None:
    
        self.aInputs = [memA.Output(aIndex) for aIndex in self.inIndexA]
        self.bInputs = [memB.Output(bIndex) for bIndex in self.inIndexB]

        for ai in range(len(self.adders)):
            self.adders[ai].Calc(self.aInputs[ai], self.bInputs[ai], lsb)
            #self.adders[ai].Print()
        self.denseOut = list(self.memD * [0])                

        if self.numOutputs == self.numInputs:
            self.sparseOut = [ad.Output() for ad in self.adders]
                
            for ni in range(len(self.sparseOut)):
                self.denseOut[self.outIndex[ni]] = self.sparseOut[ni]

    def Step(self) -> None:
        for ai in range(len(self.adders)):
            self.adders[ai].Step()
        
    def Print(self, prefix="", verbose=1) -> None:        
        print(f"{prefix}OHM_ADDER_TREE: {len(self.adders)} adders")
        for ai in range(len(self.adders)):
            self.adders[ai].Print(prefix + "  ", verbose)

