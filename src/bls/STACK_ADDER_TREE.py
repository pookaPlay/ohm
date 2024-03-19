from OHM import OHM
from BSMEM import BSMEM
from DataReader import DataReader
from ADD import ADD
import math

class STACK_ADDER_TREE():

    def __init__(self,  numInputs, memD, memK) -> None:        
        self.numInputs = numInputs
        self.numOutputs = 1
        self.memD = memD                

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
        self.memK = memK    
        self.flags = list(self.numInputs * [0])

    def Reset(self) -> None:        
        for ti in range(len(self.tree)):
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Reset()

        self.flags = list(self.numInputs * [0])
                
    def Calc(self, memInputs, memParam, msb=0) -> None:    

        self.inputs = [memInputs.OutputMSB(aIndex) for aIndex in self.inIndexA]        
        #print(f"STACK         input: {self.inputs}")        
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
        
        self.CalcPTF(memParam)

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

    def CalcPTF(self, memParam):    
        
        self.numBits = int(math.log2(len(self.inputs)))
        # LSB inner loop
        memParam.ResetIndex()
        self.ResetTree()
        
        self.treeInputs = list(self.numInputs * [0])
        for i in range(len(self.inputs)):                        
            self.treeInputs[i] = self.inputs[i] * memParam.Output(self.inIndexB[i])

        temp = 1 if (sum(self.treeInputs) >= len(self.treeInputs)/2) else 0
        print(f" -->SPBF: {temp}              from {sum(self.treeInputs)}")
        #self.pbfOut = temp 

        #return
        ti = 0
        lsb = 1                

        self.pbfOut = self.CalcTree(self.treeInputs, lsb)            
        #print(f" --> First tree PBF got {self.pbfOut}")
        #self.PrintTree()
        self.Step()

        #self.pbfOut = 1 if (sum(self.treeInputs) == len(self.treeInputs)) else 0
        #self.pbfOut = 1 if (sum(self.treeInputs) > 0) else 0
        #print(f"     PBF is {self.pbfOut} from {sum(self.treeInputs)}")

        lsb = 0
        for ti in range(1, self.numBits):
            
            memParam.Step()       

            self.treeInputs = list(self.numInputs * [0])            
            for i in range(len(self.inputs)):            
                self.treeInputs[i] = self.inputs[i] * memParam.Output(self.inIndexB[i])
            
            self.pbfOut = self.CalcTree(self.treeInputs, lsb)
            #self.PrintTree()
            self.Step()

        print(f" -->FPBF: {self.pbfOut}          after {self.numBits} ticks")
            
    
    def CalcTree(self, inputs, lsb=0):    
        #print(f"     CalcTree inputs: {inputs}" )
        if len(self.tree) > 0:
            for ai in range(len(self.tree[0])):
                self.tree[0][ai].Calc(inputs[ai*2], inputs[ai*2+1], lsb)
        if len(self.tree) > 1:
            for ti in range(1, len(self.tree)):
                for ai in range(len(self.tree[ti])):
                    self.tree[ti][ai].Calc(self.tree[ti-1][ai*2].Output(), self.tree[ti-1][ai*2+1].Output(), lsb)
            
        self.pbfOut = self.tree[-1][0].Output()
        
        return self.pbfOut
            
    def PrintTree(self):    
        #print(f"     CalcTree inputs: {inputs}" )
        for ti in range(len(self.tree)):
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Print()                

    def ResetTree(self):    
        #print(f"     CalcTree inputs: {inputs}" )
        for ti in range(len(self.tree)):
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Reset()                

    def Step(self) -> None:
        for ti in range(len(self.tree)):
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Step()

               
    def Print(self, prefix="", verbose=1) -> None:        
        print(f"{prefix}STACK_ADDER_TREE: {len(self.adders)} adders")

        for ti in range(len(self.tree)):
            print(prefix + f"---")
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Print(prefix + "  ", verbose)        
