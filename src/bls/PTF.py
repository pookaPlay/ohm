from bls.ADD import ADD
from bls.lsbSource import lsbSource
from bls.DataIO import SerializeLSBTwos, SerializeMSBTwos, SerializeMSBOffset, SerializeLSBOffset
import math

class PTF:
    def __init__(self, param) -> None:
        self.param = param

        # assume +ve and -ve already given
        self.D = param["D"]
        self.D2 = 2 * self.D
        
        self.K = math.ceil(math.log2(self.D2))
        
        bOne = SerializeLSBTwos(1, self.K)
        self.wp = [lsbSource(self.K, bOne) for _ in range(param["D"])]        
        self.wn = [lsbSource(self.K, bOne) for _ in range(param["D"])]
        bOnet = SerializeLSBTwos(-1, self.K*2)
        self.wt = lsbSource(self.K*2, bOnet)

        self.local_weights = list(self.D2 * [1])
        self.local_threshold = self.D2/2        
        self.y = 0        
        self.x = list(self.D2 * [0])
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
        numStart = int(param["D"])                            
        if numStart > 1:
            self.tree.append([ADD() for _ in range(numStart)])
            numStart = int(numStart / 2)
            while numStart > 1:
                self.tree.append([ADD() for _ in range(numStart)])
                numStart = int(numStart / 2)            
                    
        self.tree.append([ADD()])   # top of binary tree 
        self.tree.append([ADD()])   # threshold            
        self.PrintTree("PTF Init: ")

        self.ptfCount = 0
            
    def Output(self) -> int:
        return self.y

    def Calc(self, x) -> None:
        if self.param["debugTree"] == 1:        
            temp = sum([self.local_weights[i] for i in range(self.D2) if x[i] == 1])
            self.y = 1 if temp >= self.local_threshold else 0        
            self.x = x
        else:
            #intParam = memParam.GetLSBInts Hack()
            #threshParam = memThresh.GetLSBIntsHack()                
            #halfSum = threshParam[0]
            self.treeInputs = list(self.D2 * [0])
            for i in range(self.param["D"]):                                    
                self.treeInputs[i] = x[i] * self.wp[i].Output()        
                self.treeInputs[self.param["D"] + i] = x[self.param["D"] + i] * self.wn[i].Output()
            self.x = self.treeInputs

            if len(self.tree) > 0:
                for ai in range(len(self.tree[0])):
                    self.tree[0][ai].Calc(self.treeInputs[ai*2], self.treeInputs[ai*2+1], self.ptfCount==0)
            if len(self.tree) > 1:
                for ti in range(1, len(self.tree)):
                    if (ti == len(self.tree) - 1) and len(self.tree[ti]) == 1:
                        self.tree[ti][0].Calc(self.tree[ti-1][0].Output(), self.wt.Output(), self.ptfCount==0)                    
                    else:
                        for ai in range(len(self.tree[ti])):
                            self.tree[ti][ai].Calc(self.tree[ti-1][ai*2].Output(), self.tree[ti-1][ai*2+1].Output(), self.ptfCount==0)                        

                
            self.y = self.tree[-1][0].Output()    

    def Step(self) -> None:

        self.ptfCount = self.ptfCount + 1
        
        [wni.Step() for wni in self.wn]
        [wpi.Step() for wpi in self.wp]
        self.wt.Step()
        for ti in range(len(self.tree)):
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Step()


    def Print(self, prefix=""):
        print(f"{prefix} PTF: {self.x} -> {self.y}")

    def PrintTree(self, prefix=""):        
        for ti in range(len(self.tree)):
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Print(f"   {prefix} Tree[{ti}][{ai}]: ")
        
    def PrintParams(self, prefix=""):
        [wni.Print(f"   {prefix} Wn{i}: ") for i, wni in enumerate(self.wn)]
        [wpi.Print(f"   {prefix} Wp{i}: ") for i, wpi in enumerate(self.wp)]
        self.wt.Print(f"   {prefix} Wt: ")        


    def Reset(self) -> None:
        self.y = 0
        self.ptfCount = 0
        for ti in range(len(self.tree)):
            for ai in range(len(self.tree[ti])):
                self.tree[ti][ai].Reset()

    def SetMin(self) -> None:
        self.local_weights = list(self.D2 * [1])
        self.local_threshold = self.D2 
        bT = SerializeLSBTwos(-self.local_threshold, self.K*2)
        self.wt = lsbSource(self.K*2, bT)
        print(f"SetMin: {self.local_threshold}")

    def SetMax(self) -> None:
        self.local_weights = list(self.D2 * [1])
        self.local_threshold = 1
        bT = SerializeLSBTwos(-self.local_threshold, self.K*2)
        self.wt = lsbSource(self.K*2, bT)


    def SetMedMax(self, val=1) -> None:
        self.local_weights = list(self.D2 * [1])
        self.local_threshold = self.D2/2 - val
        bT = SerializeLSBTwos(-self.local_threshold, self.K*2)
        self.wt = lsbSource(self.K*2, bT)


    def SetMedian(self) -> None:
        self.local_weights = list(self.D2 * [1])
        self.local_threshold = self.D2/2
        bT = SerializeLSBTwos(-self.local_threshold, self.K*2)
        self.wt = lsbSource(self.K*2, bT)

