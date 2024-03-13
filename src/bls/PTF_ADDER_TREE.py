from OHM import OHM
from BSMEM import BSMEM
from DataReader import DataReader
from ADD import ADD
from OHM_ADDER_TREE import OHM_ADDER_TREE
import math

class PTF_ADDER_TREE(OHM_ADDER_TREE):

    def __init__(self,  numInputs, memD, memK) -> None:        
        super().__init__(numInputs, 1, memD)
        self.memK = memK    
        self.flags = list(self.numInputs * [0])

    def Reset(self) -> None:        
        super().Reset()
        self.flags = list(self.numInputs * [0])
                
    def Calc(self, memInputs, memParam, msb=0) -> None:    

        self.inputs = [memInputs.OutputMSB(aIndex) for aIndex in self.inIndexA]        
        print(f"STACK         input: {self.inputs}")
        self.numBits = int(math.log2(len(self.inputs)))
        # Called for each MSB
        if msb == 1:              
            self.inputs = [1-x for x in self.inputs]
            print(f"     MSB Negating inputs: {self.inputs}")          
            self.flags = list(len(self.inputs) * [0])
            self.latchInput = list(len(self.inputs) * [0])
            self.done = 0
            #print(f">>>>>>>>>>>>>>>>>>>>>>>>>> Negated inputs: {inputs}")
        else:
            for i in range(len(self.inputs)):    
                if self.flags[i] == 1:
                    self.inputs[i] = self.latchInput[i]

        print(f"STACK         flags: {self.flags}")
        print(f"STACK latched input: {self.inputs}")
        # Now run the LSB loop
        self.treeInputs = list(self.numInputs * [0])
        for i in range(len(self.inputs)):            
            if self.inputs[i] == 1:
                self.treeInputs[i] = memParam.Output(self.inIndexB[i])

        ti = 0
        lsb = 1        
        #self.pbfOut = self.CalcPBFStep(self.treeInputs, lsb)            
        self.pbfOut = 1 if (sum(self.treeInputs) == len(self.treeInputs)) else 0
        #self.pbfOut = 1 if (sum(self.treeInputs) > 0) else 0
        print(f"     PBF is {self.pbfOut} from {sum(self.treeInputs)}")
        # lsb = 0
        # for ti in range(1, self.numBits):
        #     memParam.Step()            
            
        #     self.treeInputs = list(self.numInputs * [0])            
        #     for i in range(len(self.inputs)):            
        #         if self.inputs[i] == 1:
        #             self.treeInputs[i] = memParam.Output(self.inIndexB[i])
            
        #     #self.pbfOut = self.CalcPBFStep(self.treeInputs, lsb)            
        #     #self.pbfOut = 1 if (sum(self.treeInputs) == len(self.treeInputs)) else 0
        #     self.pbfOut = 1 if (sum(self.treeInputs) > 0) else 0
        #     #print(f"     == LSB PBF Step {ti}: {self.pbfOut}")
 
        for i in range(len(self.inputs)):
            if self.flags[i] == 0:
                if self.inputs[i] != self.pbfOut:
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
            
