from OHM import OHM
from BSMEM import BSMEM
from DataReader import DataReader
from ADD import ADD

class OHM_ADDER_TREE:


    def __init__(self,  numInputs, memD) -> None:        
                
        self.N = numInputs
        self.memD = memD
        
        self.adders = [ADD() for _ in range(self.N)]        

        self.inIndexA = list(range(self.N))
        self.inIndexB = list(range(self.N))
        self.outIndex = list(range(self.N))

        numStart = int(self.N/2)        
        self.tree = list()
        if numStart > 1:
            self.tree.append([ADD() for _ in range(numStart)])
            numStart = int(numStart / 2)
            while numStart > 1:
                self.tree.append([ADD() for _ in range(numStart)])
                numStart = int(numStart / 2)            
                
        self.tree.append([ADD()])            
        
    
    def Reset(self) -> None:
        # Connectivity
        for ai in range(self.N):
            self.adders[ai].Reset()
        for ti in range(len(self.tree)):
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Reset()

        
    def Output(self):        
        return self.denseOut

    def Calc(self, memA, memB, lsb=0) -> None:
    
        self.aInputs = [memA.Output(self.inIndexA[ni]) for ni in range(self.N)]
        self.bInputs = [memB.Output(self.inIndexB[ni]) for ni in range(self.N)]

        for ai in range(self.N):
            self.adders[ai].Calc(self.aInputs[ai], self.bInputs[ai], lsb)
            #self.adders[ai].Print()
        if len(self.tree) > 0:
            for ai in range(len(self.tree[0])):
                self.tree[0][ai].Calc(self.adders[ai*2].Output(), self.adders[ai*2+1].Output(), lsb)
        if len(self.tree) > 1:
            for ti in range(1, len(self.tree)):
                for ai in range(len(self.tree[ti])):
                    self.tree[ti][ai].Calc(self.tree[ti-1][ai*2].Output(), self.tree[ti-1][ai*2+1].Output(), lsb)
            
        # this should be the final node 
        self.tree[-1][0].Print()
    
        self.sparseOut = [self.adders[ai].Output() for ai in range(self.N)]
        
        self.denseOut = list(self.memD * [0])                
        for ni in range(len(self.sparseOut)):
            self.denseOut[self.outIndex[ni]] = self.sparseOut[ni]
        

    def Step(self) -> None:
        for ai in range(len(self.adders)):
            self.adders[ai].Step()
        
        for ti in range(len(self.tree)):
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Step()

                
    def Print(self, prefix="", verbose=1) -> None:        
        print(f"{prefix}OHM_ADDER_TREE: {self.N} adders")
        for ai in range(len(self.adders)):
            self.adders[ai].Print(prefix + "  ", verbose)

        for ti in range(len(self.tree)):
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Print(prefix + "  ", verbose)        
