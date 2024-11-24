
class lsbSource:    
    
    def __init__(self, N, listDefault = []) -> None:
        self.N = N     
        self.Reset()   
        if len(listDefault) > 0:
            self.state = listDefault.copy()

    def Reset(self) -> None:
        self.state = list(self.N * [0])        
        self.ri = 0

    def Print(self, prefix="") -> None:                
        print(f"{prefix}lsbSource: {self.state}")


    def Output(self) -> int:
        return self.state[self.ri]        
    
    def Step(self) -> None:       
        self.ri += 1
        if self.ri == self.N:
            self.ri = 0
