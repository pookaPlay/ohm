from OHM import OHM
from BSMEM import BSMEM
from STACK import STACK

class OHM_MSB:

    def __init__(self, N, D) -> None:    
        self.N = N         
        self.D = D      
        self.stacks = [STACK() for _ in range(self.N)]        
        
        self.inIndex = list(range(self.D))        
        self.outIndex = list(range(self.D))              

        self.Reset()                   
            
    def Reset(self) -> None:
        for ai in range(self.N):
            self.stacks[ai].Reset()
        pass

    def Output(self):

        result = [self.stacks[ai].Output() for ai in range(self.N)]
        return result                    

    def Calc(self, x, msb=0) -> None:
        for ai in range(self.N):
            self.stacks[ai].Calc(x[ai], msb)        
        
            
    def Step(self) -> None:                
        for ai in range(len(self.stacks)):
            self.stacks[ai].Step()
            
                
    def Print(self, prefix="", verbose=1) -> None:        
        print(f"{prefix}OHM_MSB: {self.N} stacks")
        for ai in range(len(self.stacks)):
            self.stacks[ai].Print(prefix + "  ", verbose)
            