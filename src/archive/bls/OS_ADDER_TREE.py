import math
from bls.ADD import ADD

class OS_ADDER_TREE:

    def __init__(self,  numInputs, memD, memK) -> None:        
            
        self.numInputs = numInputs
        self.numOutputs = 1
        self.memD = memD
        self.memK = memK    
        self.flags = list(self.numInputs * [0])                
        self.adders = [ADD() for _ in range(self.numInputs)]        

        self.inIndexA = list(range(self.numInputs))
        self.inIndexB = list(range(self.numInputs))
        self.outIndex = list(range(self.numOutputs))

        self.tree = list()        
        if (self.numOutputs == 1) and (self.numInputs > 1):
            numStart = int(self.numInputs/2)                            
            if numStart > 1:
                self.tree.append([ADD() for _ in range(numStart)])
                numStart = int(numStart / 2)
                while numStart > 1:
                    self.tree.append([ADD() for _ in range(numStart)])
                    numStart = int(numStart / 2)            
                    
            self.tree.append([ADD()])                                      

    def Reset(self) -> None:        
        for ai in range(len(self.adders)):
            self.adders[ai].Reset()
        for ti in range(len(self.tree)):
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Reset()

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
                       
    def Step(self) -> None:
        for ai in range(len(self.adders)):
            self.adders[ai].Step()
        
        for ti in range(len(self.tree)):
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Step()

                
    def Print(self, prefix="", verbose=1) -> None:        
        print(f"{prefix}OS_ADDER_TREE: {len(self.adders)} adders")
        for ai in range(len(self.adders)):
            self.adders[ai].Print(prefix + "  ", verbose)

        for ti in range(len(self.tree)):
            print(prefix + f"---")
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Print(prefix + "  ", verbose)        