import math
from bls.ADD import ADD

class PTF:
    def __init__(self, D=3) -> None:
        self.D = D
        self.local_weights = list(self.D * [1])
        self.local_threshold = self.D/2
        #self.threshold = self.D
        self.y = 0
        self.lastx = list(self.D * [0])
        
        # binary tree implementation
        self.tree = list()        
        numStart = int(D/2)                            
        if numStart > 1:
            self.tree.append([ADD() for _ in range(numStart)])
            numStart = int(numStart / 2)
            while numStart > 1:
                self.tree.append([ADD() for _ in range(numStart)])
                numStart = int(numStart / 2)            
                    
        self.tree.append([ADD()])                
            
    def Output(self) -> int:
        return self.y

    def Calc(self, x) -> None:
        
        self.lastx = x.copy()
        temp = sum([self.local_weights[i] for i in range(self.D) if x[i] == 1])
        self.y = 1 if temp >= self.local_threshold else 0        

        
        #intParam = memParam.GetLSBIntsHack()
        #threshParam = memThresh.GetLSBIntsHack()                
        #halfSum = threshParam[0]
        #self.treeInputs = list(self.numInputs * [0])
        #for i in range(len(self.inputs)):                                    
        #    self.treeInputs[i] = self.inputs[i] * intParam[i]        
        #self.pbfOut = 1 if (sum(self.treeInputs) >= halfSum) else 0     

    def Print(self, prefix=""):
        print(f"{prefix} PTF: {self.lastx} -> {self.y}")

    def Step(self, x) -> None:
        #self.Calc(x)
        pass

    def Reset(self) -> None:
        self.y = 0
        for ti in range(len(self.tree)):
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Reset()


    def SetMin(self) -> None:
        self.local_weights = list(self.D * [1])
        self.local_threshold = self.D 

    def SetMax(self) -> None:
        self.local_weights = list(self.D * [1])
        self.local_threshold = 1

    def SetMedMax(self, val=1) -> None:
        self.local_weights = list(self.D * [1])
        self.local_threshold = self.D/2 - val

    def SetMedian(self) -> None:
        self.local_weights = list(self.D * [1])
        self.local_threshold = self.D/2
