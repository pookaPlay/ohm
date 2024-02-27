from DataIO import SerializeMSBTwos,  DeserializeMSBTwos, DeserializeLSBTwos

# Bit Serial Memory
class BSMEM():

    def __init__(self, D, K, writeMode=1):
        self.D = D
        self.K = K
        self.mode = writeMode
        self.mem = [list(self.K * [0]) for _ in range(self.D)]        
        self.Reset()
        

    def Reset(self):
        self.ri = 0
        self.wi = 0
        self.nextInput = list(self.D * [0])
        self.mem = [list(self.K * [0]) for _ in range(self.D)]        
    
    def ClearInput(self):
        self.nextInput = list(self.D * [0])

    def SetInput(self, input, di=-1):
        if di == -1:
            self.nextInput = input  
        else:   
            self.nextInput[di] = input

    def Output(self, di=-1):
        if di != -1:
            return self.mem[di][self.ri]
        else:
            return [self.mem[ai][self.ri] for ai in len(self.mem)]
    
    def Step(self, input):
        #print(f"###################################")
        #self.Print("DEBUG IN")

        # write
        if self.mode == 1:
            #print("####################################")
            #print(self.nextInput)
            #print(self.mem)
            for di in range(len(input)):                
                self.mem[di][self.wi] = input[di]
                #print(f"Setting {di}, {self.wi} value {self.nextInput[di]}")

            self.wi += 1
            if self.wi == self.K:
                self.wi = 0
        else:   # read 
            self.ri += 1        
            if self.ri == self.K:
                self.ri = 0

        #self.Print("DEBUG OUT")

    def GetInts(self):
        result = list()
        for di in range(len(self.mem)):
            result.append(DeserializeLSBTwos(self.mem[di]))
        
        return result
    
    def Print(self, prefix="", verbose=2):
        print(f"{prefix}|BSMEM--------------------------|")
        print(f"{prefix}|Size: {self.D} Depth: {self.K} ")
        
        for di in range(len(self.mem)):
                print(f"{prefix}{self.mem[di]}")


                

    
    
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