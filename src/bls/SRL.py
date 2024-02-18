
class SRL:    
    # inputMode: 0 - fifo, 1 - constant coefficient
    def __init__(self, N) -> None:
        self.N = N        
        self.state = list(N * [0])
        self.ri = N-1
        self.wi = 0

    # input: MSB list 
    def Reset(self) -> None:
        self.state = list(self.N * [0])                
        self.ri = self.N-1
        self.wi = 0

    # output: LSB bit
    def Output(self) -> int:
        return self.state[self.ri]  

    def Print(self, prefix="") -> None:        
        ##print(f"--------- HERE {self.ri}")

        temps = prefix + f"SRL: "
        mem0 = [str(self.state[i]) for i in range(self.N)]
        for i in range(self.N):
            if i == self.ri:
                temps += "(" + mem0[i] + ")"
            else:
                temps += " " + mem0[i]
        print(temps)

    def Step(self, input=0) -> None:
        self.state[1:] = self.state[0:]
        self.state[0] = input
        


