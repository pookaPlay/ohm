
class lsbSource:    
    
    def __init__(self, N, listDefault = []) -> None:
        self.N = N  
        self.default = list(self.N * [0])    

        if len(listDefault) > 0:
            self.default = listDefault.copy()
            
        self.Reset()   

    def Reset(self) -> None:           
        self.state = self.default.copy()
        self.ri = 0

    def Print(self, prefix="") -> None:                
        print(f"{prefix}lsbSource: {self.state}")


    def Output(self) -> int:
        return self.state[self.ri]        
    
    def Step(self) -> None:       
        self.ri += 1
        if self.ri == self.N:
            self.ri = 0
