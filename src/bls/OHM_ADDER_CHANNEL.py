from bls.ADD import ADD

def NOT(x):
    return 1 - x

class OHM_ADDER_CHANNEL:

    def __init__(self,  numInputsMirrored, memD) -> None:        
        # numInputsMirrored is 2*actual (+ve/-ve)

        self.numInputsMirrored = numInputsMirrored
        self.numInputs = int(self.numInputsMirrored/2)        
        self.numOutputs = self.numInputsMirrored
        self.memD = memD
        
        self.adders = [ADD() for _ in range(self.numInputsMirrored)]        
        # mirror inputs
        self.inIndexA = list(range(self.numInputs))        
        self.inIndexA.extend(range(self.numInputs))
        # parameter memory (B) should be 2*A 
        self.inIndexB = list(range(self.numInputsMirrored))
        self.outIndex = list(range(self.numOutputs))
            

    def Reset(self) -> None:        
        for ai in range(len(self.adders)):
            self.adders[ai].Reset()
                        
    def Output(self):        
        return self.denseOut

    def Calc(self, memA, memB, lsb=0) -> None:
    
        self.aInputs = [memA.Output(aIndex) for aIndex in self.inIndexA]
        # mirror inputs
        for ai in range(self.numInputs):
            self.aInputs[ai+self.numInputs] = NOT(self.aInputs[ai])

        self.bInputs = [memB.Output(bIndex) for bIndex in self.inIndexB]

        for ai in range(len(self.adders)):
            self.adders[ai].Calc(self.aInputs[ai], self.bInputs[ai], lsb)
        
        self.denseOut = [ad.Output() for ad in self.adders]        

    def Step(self) -> None:
        for ai in range(len(self.adders)):
            self.adders[ai].Step()
        
    def Print(self, prefix="", verbose=1) -> None:        
        print(f"{prefix}OHM_ADDER_CHANNEL: {len(self.adders)} adders")
        for ai in range(len(self.adders)):
            self.adders[ai].Print(prefix + "  ", verbose)

