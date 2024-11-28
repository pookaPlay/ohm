import math

class PTF:
    def __init__(self, D=3) -> None:
        self.D = D
        self.weights = list(self.D * [1])
        self.threshold = self.D/2
        #self.threshold = self.D
        self.y = 0
        self.lastx = list(self.D * [0])
            

    def SetMin(self) -> None:
        self.weights = list(self.D * [1])
        self.threshold = self.D 

    def SetMax(self) -> None:
        self.weights = list(self.D * [1])
        self.threshold = 1

    def SetMedMax(self, val=1) -> None:
        self.weights = list(self.D * [1])
        self.threshold = self.D/2 - val

    def SetMedian(self) -> None:
        self.weights = list(self.D * [1])
        self.threshold = self.D/2

    def Output(self) -> int:
        return self.y

    def Calc(self, x) -> None:
        
        self.lastx = x.copy()
        temp = sum([self.weights[i] for i in range(self.D) if x[i] == 1])
        self.y = 1 if temp >= self.threshold else 0        

    def Print(self, prefix=""):
        print(f"{prefix} PTF: {self.lastx} -> {self.y}")

    def Step(self, x) -> None:
        self.Calc(x)
        pass

    def Reset(self) -> None:
        self.y = 0