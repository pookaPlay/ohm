
class ADD:
    def __init__(self) -> None:        
        self.sum = 0
        self.cin = 0
        self.a = 0
        self.b = 0

    def Print(self, prefix="") -> None:        
        print(prefix + f"ADD : {self.sum}, {self.cin} from {self.a} and {self.b}")
    
    def Output(self) -> int:
        return self.sum

    def Reset(self, a, b) -> None:
        self.sum = 0
        self.cin = 0
        self.Calc(a, b)

    def Calc(self, a, b) -> None:                
        self.a = a
        self.b = b
        self.sum  = (a+b+self.cin) % 2
        self.cout = 1 if (a+b+self.cin) > 1 else 0

    def Step(self) -> None:                
        self.cin = self.cout
        #print(f"  After step ADD : {self.sum} and {self.cin}")
