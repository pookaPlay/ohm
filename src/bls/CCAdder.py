from SRL import SRL
from ADD import ADD

class CCAdder:
    # Constant Coefficient Adder
    
    def __init__(self, N) -> None:
        self.N = N
        self.w1 = SRL(N, 1)     # 1 makes SRL a constant coefficient 
        self.w2 = SRL(N, 1)
        self.add = ADD()
        
    def Reset(self, a, b) -> None:
        self.w1.Reset(a)
        self.w2.Reset(b)
        self.add.Reset(self.w1.Output(), self.w2.Output())

    def Output(self) -> int:
        return self.add.Output()
        
    def Step(self) -> None:
        self.w1.Step()
        self.w2.Step()
        self.add.Step(self.w1.Output(), self.w2.Output())

    def Print(self) -> None:
        self.w1.Print()
        self.w2.Print()
        self.add.Print()    

