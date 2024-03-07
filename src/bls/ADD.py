###################
## Bit Serial Adder

class ADD:
    def __init__(self) -> None:        
        self.sum = 0
        self.cin = 0
        self.a = 0
        self.b = 0
        self.cout = 0

    def Print(self, prefix="", verbose=1) -> None:
        if verbose > 0:
            print(prefix + f"ADD : {self.sum}, {self.cout} from {self.a}, {self.b} and {self.cin}")
    
    def Output(self) -> int:
        return self.sum

    def Reset(self) -> None:            
        self.sum = 0
        self.cin = 0
        self.cout = 0        

    def Calc(self, a, b, lsb=0) -> None:                
        if lsb == 1:
            self.cin = 0

        self.a = a
        self.b = b
        self.sum  = (a+b+self.cin) % 2
        self.cout = 1 if (a+b+self.cin) > 1 else 0

    def Step(self) -> None:                
        self.cin = self.cout
        #print(f"  After step ADD : {self.sum} and {self.cin}")
