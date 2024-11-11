
class lsb2msb:    
    
    def __init__(self, N) -> None:
        self.N = N        
        self.Reset()
    
    def Reset(self) -> None:
        self.state = [list(self.N * [0]), list(self.N * [0])]        
        self.mode = 0
        self.wi = self.N-1
        self.ri = 0
        self.done = 0

    def Print(self, prefix="") -> None:        
        temps = prefix + "L2M: "

        mem0 = [str(self.state[0][i]) for i in range(self.N)]            
        mem1 = [str(self.state[1][i]) for i in range(self.N)]

        if self.mode == 0:    
            temps += str("W[")
            for i in range(self.N):
                if i == self.wi:
                    temps += "(" + mem0[i] + ")"
                else:
                    temps += " " + mem0[i]
            temps += "] R["
            for i in range(self.N):
                if i == self.ri:
                    temps += "(" + mem1[i] + ")"
                else:
                    temps += " " + mem1[i]            
        else:
            temps += str("R[")
            for i in range(self.N):
                if i == self.ri:
                    temps += "(" + mem0[i] + ")"
                else:
                    temps += " " + mem0[i]
            temps += "] W["
            for i in range(self.N):
                if i == self.wi:
                    temps += "(" + mem1[i] + ")"
                else:
                    temps += " " + mem1[i]
            
        print(temps + "]")


    def Output(self) -> int:
        readMode = 1 - self.mode        
        return self.state[readMode][self.ri]        
    
    def Step(self, input) -> None:
        #print(f"L2M write at {self.wi} : {input}")
        self.state[self.mode][self.wi] = input                      
        
        #print(f"L2M step {self.wi} : {input} with hold {holdFlag}")
        self.ri += 1
        
        self.wi -= 1
        if self.wi < 0:
            self.wi = self.N-1
            self.ri = 0
            self.mode = 1 - self.mode                    
