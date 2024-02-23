# Bit Serial Memory

class BSMEM():

    def __init__(self, D, K):
        self.D = D
        self.K = K
        
        self.Reset()
        

    def Reset(self):
        self.ri = 0
        self.wi = 0
        self.nextInput = list(self.D * [0])
        self.mem = list(self.D * list(self.K * [0]))

    def SetInput(self, di, input):
        self.nextInput[di] = input

    def GetOutput(self, di):
        return self.mem[di]

    def GetOutput(self):
        return self.mem
    
    def Step(self):
        self.ri += 1
        if self.ri == self.K:
            self.ri = 0

    def Print(self):
        print("BSMEM")
        print(f"width: {self.D} depth: {self.K}")
        print(f"mem: {self.mem}")   

        