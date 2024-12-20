import math

class PBF:
    def __init__(self, D=3) -> None:
        self.D = D
        self.weights = list(self.D * [1])
        #self.threshold = self.D/2
        self.threshold = self.D
        self.y = 0
        self.lastx = list(self.D * [0])
            

    def SetMin(self) -> None:
        self.weights = list(self.D * [1])
        self.threshold = self.D 

    def SetMax(self) -> None:
        self.weights = list(self.D * [1])
        self.threshold = 1

    def SetMedian(self) -> None:
        self.weights = list(self.D * [1])
        self.threshold = self.D/2

    def Output(self) -> int:
        return self.y

    def Calc(self, x) -> None:
        self.lastx = x
        temp = sum([self.weights[i] for i in range(self.D) if x[i] == 1])
        self.y = 1 if temp >= self.threshold else 0        

    def Print(self, prefix=""):
        print(f"{prefix} PBF: {self.lastx} -> {self.y}")

    def Step(self, x) -> None:
        pass

    def Reset(self, x) -> None:
        self.Calc(x)