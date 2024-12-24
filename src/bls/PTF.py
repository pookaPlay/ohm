from bls.ADD import ADD
from bls.lsbSource import lsbSource
from bls.DataIO import SerializeLSBTwos, SerializeMSBTwos, SerializeMSBOffset, SerializeLSBOffset
import math

class PTF:
    def __init__(self, param) -> None:
        self.param = param
        self.D = 2 * param["D"]
        
        self.K = math.ceil(math.log2(param["D"]))
        
        bOne = SerializeLSBTwos(1, self.K)
        self.wp = [lsbSource(self.K, bOne) for _ in range(param["D"])]        
        self.wn = [lsbSource(self.K, bOne) for _ in range(param["D"])]
        self.wt = lsbSource(self.K, bOne)

        self.local_weights = list(self.D * [1])
        self.local_threshold = self.D/2        
        self.y = 0        
        self.x = list(self.D * [0])
        # Some presets for debugging
        if param["ptf"] == "min":
            self.SetMin()
        elif param["ptf"] == "max":          
            self.SetMax()
        elif param["ptf"] == "medmax":          
            self.SetMedMax()                           
        else:       
            self.SetMedian()
        
        # binary tree implementation
        self.tree = list()        
        numStart = int(self.D/2)                            
        if numStart > 1:
            self.tree.append([ADD() for _ in range(numStart)])
            numStart = int(numStart / 2)
            while numStart > 1:
                self.tree.append([ADD() for _ in range(numStart)])
                numStart = int(numStart / 2)            
                    
        self.tree.append([ADD()])    # threshold?            
            
    def Output(self) -> int:
        return self.y

    def Calc(self, x) -> None:
        if self.param["debugTree"] == 1:        
            temp = sum([self.local_weights[i] for i in range(self.D) if x[i] == 1])
            self.y = 1 if temp >= self.local_threshold else 0        
            self.x = x
        else:
            pass
        #intParam = memParam.GetLSBInts Hack()
        #threshParam = memThresh.GetLSBIntsHack()                
        #halfSum = threshParam[0]
        #self.treeInputs = list(self.numInputs * [0])
        #for i in range(len(self.inputs)):                                    
        #    self.treeInputs[i] = self.inputs[i] * intParam[i]        
        #self.pbfOut = 1 if (sum(self.treeInputs) >= halfSum) else 0     

    def Step(self) -> None:
        #self.Calc(x)
        pass


    def Print(self, prefix=""):
        print(f"{prefix} PTF: {self.x} -> {self.y}")


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
