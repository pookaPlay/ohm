from OHM import OHM
from BSMEM import BSMEM
from DataReader import DataReader
from ADD import ADD

class OHM_ADDER_TREE:


    def __init__(self,  numInputs, numOutputs, memD) -> None:        
                
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.memD = memD
        
        self.adders = [ADD() for _ in range(self.numInputs)]        

        self.inIndexA = list(range(self.numInputs))
        self.inIndexB = list(range(self.numInputs))
        self.outIndex = list(range(self.numOutputs))

        self.tree = list()        
        if self.numOutputs == 1:            
            numStart = int(self.numInputs/2)                            
            if numStart > 1:
                self.tree.append([ADD() for _ in range(numStart)])
                numStart = int(numStart / 2)
                while numStart > 1:
                    self.tree.append([ADD() for _ in range(numStart)])
                    numStart = int(numStart / 2)            
                    
            self.tree.append([ADD()])                
    

    def Reset(self) -> None:
        # Connectivity
        for ai in range(len(self.adders)):
            self.adders[ai].Reset()
        for ti in range(len(self.tree)):
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Reset()

        
    def Output(self):        
        return self.denseOut

    def Calc(self, memA, memB, lsb=0) -> None:
    
        self.aInputs = [memA.Output(aIndex) for aIndex in self.inIndexA]
        self.bInputs = [memB.Output(bIndex) for bIndex in self.inIndexB]

        for ai in range(len(self.adders)):
            self.adders[ai].Calc(self.aInputs[ai], self.bInputs[ai], lsb)
            #self.adders[ai].Print()
        
        if len(self.tree) > 0:
            for ai in range(len(self.tree[0])):
                self.tree[0][ai].Calc(self.adders[ai*2].Output(), self.adders[ai*2+1].Output(), lsb)
        if len(self.tree) > 1:
            for ti in range(1, len(self.tree)):
                for ai in range(len(self.tree[ti])):
                    self.tree[ti][ai].Calc(self.tree[ti-1][ai*2].Output(), self.tree[ti-1][ai*2+1].Output(), lsb)
            
        self.denseOut = list(self.memD * [0])                

        if self.numOutputs == self.numInputs:
            self.sparseOut = [ad.Output() for ad in self.adders]
                
            for ni in range(len(self.sparseOut)):
                self.denseOut[self.outIndex[ni]] = self.sparseOut[ni]
        elif self.numOutputs == 1:            
            # self.tree[-1][0].Print()            
            self.denseOut[0] = self.tree[-1][0].Output()
        

    def Step(self) -> None:
        for ai in range(len(self.adders)):
            self.adders[ai].Step()
        
        for ti in range(len(self.tree)):
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Step()

                
    def Print(self, prefix="", verbose=1) -> None:        
        print(f"{prefix}OHM_ADDER_TREE: {len(self.adders)} adders")
        for ai in range(len(self.adders)):
            self.adders[ai].Print(prefix + "  ", verbose)

        for ti in range(len(self.tree)):
            print(prefix + f"---")
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Print(prefix + "  ", verbose)        
