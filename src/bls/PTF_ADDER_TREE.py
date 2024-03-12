from OHM import OHM
from BSMEM import BSMEM
from DataReader import DataReader
from ADD import ADD
from OHM_ADDER_TREE import OHM_ADDER_TREE

class PTF_ADDER_TREE(OHM_ADDER_TREE):

    def __init__(self,  numInputs, memD, memK) -> None:        
        super().__init__(numInputs, 1, memD)
        self.memK = memK    
        self.flags = list(self.numInputs * [0])

    def Reset(self) -> None:        
        super().Reset()
        self.flags = list(self.numInputs * [0])

                
    def Calc(self, memInputs, memParamPos, memParamNeg, msb=0) -> None:    

        self.inputs = [memInputs.Output(aIndex) for aIndex in self.inIndexA]        
        print(f"STACK has {len(self.inputs)}")

        # Called for each MSB
        if msb == 1:            
            self.inputs = [1-x for x in self.inputs]
            self.flags = list(len(self.inputs) * [0])
            self.latchInput = list(len(self.inputs) * [0])
            self.done = 0
            #print(f">>>>>>>>>>>>>>>>>>>>>>>>>> Negated inputs: {inputs}")
        else:
            for i in range(len(self.inputs)):    
                if self.flags[i] == 1:
                    self.inputs[i] = self.latchInput[i]

        # Now run the LSB loop
        self.treeInputs = list()                    
        for i in range(len(self.inputs)):            
            if self.inputs[i] == 1:
                self.treeInputs.append( memParamPos.Output(self.inIndexB[i]) )
            else:
                self.treeInputs.append( memParamNeg.Output(self.inIndexB[i]) )

        ti = 0
        lsb = 1        
        pbfout = self.CalcPBFStep(self.treeInputs, lsb)            
        #print(f"     == LSB PBF Step 0: {pbfout}")                        
        lsb = 0
        for ti in range(1, self.memK):
            memParamPos.Step()
            memParamNeg.Step()
            
            self.treeInputs = list()                    
            for i in range(len(self.inputs)):            
                if self.inputs[i] == 1:
                    self.treeInputs.append( memParamPos.Output(self.inIndexB[i]) )
                else:
                    self.treeInputs.append( memParamNeg.Output(self.inIndexB[i]) )
            
            self.CalcPBFStep(self.treeInputs, lsb)            
            #print(f"     == LSB PBF Step {ti}: {self.pbfOut}")

        print(f"  == PBF Out: {self.pbfOut}")                        
        
        for i in range(len(self.inputs)):
            if self.flags[i] == 0:
                if self.inputs[i] != pbfout:
                    self.flags[i] = 1
                    self.latchInput[i] = self.inputs[i]
                                        
        if (sum(self.flags) == (len(self.inputs)-1)):            
            self.done = 1            
        
        if msb == 1:
            self.pbfOut = 1 - self.pbfOut
        
        self.denseOut = list(self.memD * [0])
        self.denseOut[0] = self.pbfOut

        return self.denseOut

    def CalcPBFStep(self, inputs, lsb=0):    
        
        if len(self.tree) > 0:
            for ai in range(len(self.tree[0])):
                self.tree[0][ai].Calc(inputs[ai*2], inputs[ai*2+1], lsb)
        if len(self.tree) > 1:
            for ti in range(1, len(self.tree)):
                for ai in range(len(self.tree[ti])):
                    self.tree[ti][ai].Calc(self.tree[ti-1][ai*2].Output(), self.tree[ti-1][ai*2+1].Output(), lsb)
            
        self.pbfOut = self.tree[-1][0].Output()
        
        return self.pbfOut
            
