from OHM import OHM
from BSMEM import BSMEM
from DataReader import DataReader
from ADD import ADD
from OHM_ADDER_TREE import OHM_ADDER_TREE

class PTF_ADDER_TREE(OHM_ADDER_TREE):

    def __init__(self,  numInputs, memD) -> None:        
        super().__init__(numInputs, 1, memD)
            

    def Reset(self) -> None:        
        super().Reset()
                
    def Calc(self, memA, memB, lsb=0) -> None:    
        # Called for each MSB
        self.aInputs = [memA.Output(aIndex) for aIndex in self.inIndexA]
        self.bInputs = [memB.Output(bIndex) for bIndex in self.inIndexB]

        ti = 0                        
        #print(f"     == {stepi}:{ti} ")
        firstBit = 1            

        self.CalcStep(firstBit)            
        #self.denseOut = self.biases.Output()
        #self.outMem.Step(self.denseOut)   don't save down here!                    
        for ti in range(1, self.K):
            #print(f"     == {stepi}:{ti} ")                
            firstBit = 0                
            #self.dataMem.Step()                
            #self.paramMem.Step()
            #self.CalcStep(lsb)        
        

    def CalcStep(self, lsb=0) -> None:    

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
        
"""         
        inputs = x.copy()
        # Negate if msb to convert to offset
        if msb == 1:            
            inputs = [1-x for x in inputs]
            self.flags = list(self.D * [0])
            self.latchInput = list(self.D * [0])
            self.done = 0
            #print(f">>>>>>>>>>>>>>>>>>>>>>>>>> Negated inputs: {inputs}")
        else:
            for i in range(self.D):    
                if self.flags[i] == 1:
                    inputs[i] = self.latchInput[i]

        self.pbf.Calc(inputs)

        for i in range(self.D):
            if self.flags[i] == 0:
                if inputs[i] != self.pbf.Output():
                    self.flags[i] = 1
                    self.latchInput[i] = inputs[i]
                                        
        if (sum(self.flags) == (self.D-1)):            
            self.done = 1            
        
        if msb == 1:
            self.pbfOut = 1 - self.pbf.Output()
        else:
            self.pbfOut = self.pbf.Output()
 """