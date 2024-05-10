from bls.ADD import ADD
import math


class STACK_WEIGHTED_LATTICE:

    def __init__(self,  numInputs, memD, memK, param) -> None:        
        self.numInputs = numInputs
        self.numOutputs = 1
        self.memD = memD                
        self.param = param
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
        self.posCount = 0
        self.negCount = 0
        self.stepCount = 0

    def Reset(self) -> None:        
        for ti in range(len(self.tree)):
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Reset()

        self.flags = list(self.numInputs * [0])

                
    def Calc(self, memInputs, memParam, memThresh, msb=0, sampleIndex=0) -> None:    

        self.inputs = [memInputs.OutputMSB(aIndex) for aIndex in self.inIndexA]        

        if msb == 1:              
            self.inputs = [1-x for x in self.inputs]
            self.origInputs = [1-x for x in self.inputs]
            self.flags = list(len(self.inputs) * [0])
            self.latchInput = list(len(self.inputs) * [0])
            self.done = 0
            self.doneIndex = -1
            self.threshCount = 0
            self.weightCount = list(len(self.inputs) * [0])
            self.posCount = 0
            self.negCount = 0
            self.stepCount = 0
            self.lastOut = 0
        else:
            self.origInputs  = self.inputs.copy()
            for i in range(len(self.inputs)):    
                if self.flags[i] == 1:
                    self.inputs[i] = self.latchInput[i]
                
        intParam = memParam.GetLSBIntsHack()
        threshParam = memThresh.GetLSBIntsHack()        
        
        halfSum = threshParam[0]

        self.treeInputs = list(self.numInputs * [0])
        for i in range(len(self.inputs)):                                    
            self.treeInputs[i] = self.inputs[i] * intParam[i]
        
        self.pbfOut = 1 if (sum(self.treeInputs) >= halfSum) else 0                
        signOut = self.pbfOut*2-1
        
        self.stepCount = self.stepCount + 1
        if signOut > 0:
            self.posCount = self.posCount + 1
        else:
            self.negCount = self.negCount + 1        
        
        for i in range(len(self.inputs)):
            if self.flags[i] == 0:
                if self.inputs[i] != self.pbfOut:
                    self.flags[i] = 1
                    self.latchInput[i] = self.inputs[i]

        for i in range(len(self.flags)):
            if self.done == 0:
                if self.flags[i] == 0:                        
                    self.weightCount[i] = self.weightCount[i] + 1
        
        if (sum(self.flags) == (len(self.inputs)-1)):          
            self.done = 1
            self.doneIndex = [i for i, flag in enumerate(self.flags) if flag == 0]
                
        if self.param['adaptThreshCrazy'] > 0:

            ptfWeights = memParam.GetLSBIntsHack()
            totalWeight = sum(ptfWeights)
            halfWeight = sum(ptfWeights)/2
            ptfThresh = memThresh.GetLSBIntsHack()        
            if msb == 1: 
                self.memK                         
                self.stepWeight = halfWeight/2
            else:
                self.stepWeight = self.stepWeight/2

            #print(f"Step Weight: {self.stepWeight} at step {self.stepCount}")
            if self.stepWeight >= 1:
                if signOut > 0:            
                    ptfThresh[0] = ptfThresh[0] + self.stepWeight
                    if ptfThresh[0] > totalWeight:
                        ptfThresh[0] = totalWeight
                        di = self.doneIndexOut[0]
                        assert(di >= 0)
                        ptfWeights[di] = ptfWeights[di] + 1                    
                        print(f"UPPER Thresh: {di}")       

                else:
                    ptfThresh[0] = ptfThresh[0] - self.stepWeight
                    if ptfThresh[0] < 1:
                        ptfThresh[0] = 1              
                        di = self.doneIndexOut[0]
                        assert(di >= 0)
                        ptfWeights[di] = ptfWeights[di] + 1                    
                        print(f"LOWER Thresh: {di}")       

            #memParam.SetLSBIntsHack(ptfWeights)            
            memThresh.SetLSBIntsHack(ptfThresh)
            #print(f" {self.stepCount}: setting to  {ptfThresh[0]}")


        if msb == 1:
            self.pbfOut = 1 - self.pbfOut

        return self.pbfOut

    def Output(self):        
        return self.pbfOut               

    def Print(self, prefix="", verbose=1) -> None:        
        print(f"{prefix}STACK_WEIGHTED_LATTICE")


def GetNegativeIndex(din, N):
    if din < N/2:
        dout = int(din + N/2)
    else:
        dout = int(din - N/2)
    return dout
