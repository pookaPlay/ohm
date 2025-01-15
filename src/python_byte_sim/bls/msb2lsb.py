
class msb2lsb:    
    
    def __init__(self, N) -> None:
        self.N = N     
        self.Reset()   

    def Reset(self) -> None:
        self.state = [list(self.N * [0]), list(self.N * [0])]        
        self.mode = 0
        self.wi = 0
        self.ri = self.N-1

    def Print(self, prefix="") -> None:        
        temps = prefix + "M2L: "

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
    
    def Step(self, input, holdFlag=0) -> None:
        self.state[self.mode][self.wi] = input
        
        if holdFlag == 0:
            self.ri -= 1
        
        self.wi += 1
        if self.wi == self.N:
            self.wi = 0
            self.ri = self.N-1
            self.mode = 1 - self.mode                    
