from DataIO import SerializeMSBTwos

# Bit Serial Memory
class BSMEM():

    def __init__(self, D, K, writeMode=1):
        self.D = D
        self.K = K
        self.mode = writeMode

        self.Reset()
        

    def Reset(self):
        self.ri = 0
        self.wi = 0
        self.nextInput = list(self.D * [0])
        self.mem = [list(self.K * [0]) for _ in range(self.D)]        
    
    def ClearInput(self):
        self.nextInput = list(self.D * [0])

    def SetInput(self, di, input):
        self.nextInput[di] = input

    def SetInputs(self, input):
        self.nextInput = input

    def GetOutput(self, di):
        return self.mem[di]

    def GetOutput(self):
        return self.mem
    
    def Step(self):
        # write
        if self.mode == 1:
            print("####################################")
            print(self.nextInput)
            print(self.mem)
            for di in range(self.D):
                print(self.mem[di])
                self.mem[di][self.wi] = self.nextInput[di]

            self.wi += 1
            if self.wi == self.K:
                self.wi = 0
        # read 
        self.ri += 1        
        if self.ri == self.K:
            self.ri = 0

    def Print(self):
        print("BSMEM")
        print(f"width: {self.D} depth: {self.K}")
        print(f"mem: {self.mem}")   

    
    
"""     def LoadData(self, data, Kin=8, Kout=8):
        self.N = len(data)
        for n in range(self.N):
            self.D = len(data[n])
            data = [SerializeMSBTwos(data[n][d], Kin) for d in range(self.D)]
            for d in range(self.D):
                data[d].reverse()                            

            # maybe want to sign extend twos complement here
            #self.slice = [self.data[d][self.NIn-1] for d in range(self.D)]
 """