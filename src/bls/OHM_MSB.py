from OHM import OHM
from BSMEM import BSMEM
from STACK import STACK

class OHM_MSB:

    def __init__(self, numNodes, memD, nodeD, ptf="max") -> None:    
        self.N = numNodes         
        self.nodeD = nodeD
        self.memD = memD
        
        self.stacks = [STACK(self.nodeD, ptf) for _ in range(self.N)]        
                
        self.Reset()                   
            
    def Reset(self) -> None:
        for ai in range(self.N):
            self.stacks[ai].Reset()
        
        self.inIndex = [list(range(self.nodeD)) for _ in range(self.N)]
        self.outIndex = list(range(self.N))              
        self.denseOut = list(self.memD * [0])                    

    def Output(self):        
        return self.denseOut                    

    def Calc(self, mem, msb=0) -> None:
        
        self.denseOut = list(self.memD * [0])        
        
        for ai in range(self.N):
            inIndex = self.inIndex[ai] 
            inputs = [mem.OutputMSB(inIndex[ni]) for ni in range(len(inIndex))]
            self.stacks[ai].Calc(inputs, msb)                    
            self.denseOut[self.outIndex[ai]] = self.stacks[ai].Output()                    
        
            
    def Step(self) -> None:                
        for ai in range(len(self.stacks)):
            self.stacks[ai].Step()
            
                
    def Print(self, prefix="", verbose=1) -> None:        
        print(f"{prefix}OHM_MSB: {self.N} stacks")
        for ai in range(len(self.stacks)):
            self.stacks[ai].Print(prefix + "  ", verbose)
            