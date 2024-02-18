
class CCSRL:    
    
    def __init__(self, Nin, Nout) -> None:
        ## Nin is the stored precision of weights
        ## NOut is the sign extened precision
        self.Nin = Nin    
        self.Nout = Nout    # sign extend    
        self.state = list(Nin * [0])
        self.ri = self.Nin-1        
    
    def Reset(self, data=list()) -> None:
        if len(data) > 0:
            self.state = data
        else:
            self.state = list(self.Nin * [0])                
        self.ri = self.Nin-1        

    def Output(self) -> int:
        if self.ri < 0:
            return self.state[0]
        else:
            return self.state[self.ri]  

    def Print(self, prefix="") -> None:        
        temps = prefix + f" CC: ["
        mem0 = [str(self.state[i]) for i in range(self.Nin)]        
        for i in range(self.Nin):
            if i == self.ri:
                temps += "(" + mem0[i] + ")"
            else:
                temps += " " + mem0[i]
        print(temps + "]")

    def Step(self, input=0) -> None:
        extra = self.Nin - self.Nout
        self.ri = self.ri - 1
        if self.ri < extra:
            self.ri = self.Nin-1
        


