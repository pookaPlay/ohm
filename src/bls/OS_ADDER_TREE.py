from OHM import OHM
from BSMEM import BSMEM
from DataReader import DataReader
from ADD import ADD
from OHM_ADDER_TREE import OHM_ADDER_TREE
import math

class OS_ADDER_TREE(OHM_ADDER_TREE):

    def __init__(self,  numInputs, memD, memK) -> None:        
        super().__init__(numInputs, 1, memD)
        self.memK = memK    
        self.flags = list(self.numInputs * [0])        

    def Reset(self) -> None:        
        super().Reset()
        self.flags = list(self.numInputs * [0])
                
    def Calc(self, memInputs, memParam, msb=0) -> None:    

        self.inputs = [memInputs.OutputMSB(aIndex) for aIndex in self.inIndexA]        
        #print(f"STACK         input: {self.inputs}")
        self.numBits = int(math.log2(len(self.inputs)))
        # Called for each MSB
        if msb == 1:              
            self.inputs = [1-x for x in self.inputs]
            #print(f"     MSB Negating inputs: {self.inputs}")          
            self.flags = list(len(self.inputs) * [0])
            self.latchInput = list(len(self.inputs) * [0])
            self.done = 0
            #print(f">>>>>>>>>>>>>>>>>>>>>>>>>> Negated inputs: {inputs}")
        else:
            for i in range(len(self.inputs)):    
                if self.flags[i] == 1:
                    self.inputs[i] = self.latchInput[i]

        #print(f"STACK         flags: {self.flags}")
        #print(f"STACK latched input: {self.inputs}")

        self.CalcOST(memParam)        

        for i in range(len(self.inputs)):
            if self.flags[i] == 0:
                if self.inputs[i] != self.pbfOut:
                    self.flags[i] = 1
                    self.latchInput[i] = self.inputs[i]
                                        
        if (sum(self.flags) == (len(self.inputs)-1)):
            #print(f"STACK DONE ########")
            self.done = 1            
        
        if msb == 1:
            self.pbfOut = 1 - self.pbfOut
        
        return self.pbfOut

    def Output(self):        
        return self.pbfOut


    def CalcOST(self, memParam):    
        # hack uses this as threshold
        thresh = memParam.Output(0)
        #print(f" STACK-WOS {thresh}")
        self.pbfOut = 1 if (sum(self.inputs) > thresh) else 0
               
