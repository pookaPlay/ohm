from DataIO import SerializeMSBTwos,  DeserializeMSBTwos, DeserializeLSBTwos
from SingleDataReader import SingleDataReader

# Read Only Serial Memory
class RDMEM():

    def __init__(self, input, DK, K):
        self.D = DK
        self.K = K        
        self.rd = [SingleDataReader(input[ni], self.K, self.K) for ni in range(len(input))]
    
        self.Reset()
        

    def Reset(self):
        self.ri = 0
        self.wi = 0
        [a.Reset() for a in self.rd]        
    

    def Output(self, di=-1):
        if di != -1:
            return self.rd[di].Output()
        else:
            return [a.Output() for a in self.rd]         
    
    def Step(self):
        [a.Step() for a in self.rd]         


    def Print(self, prefix="", verbose=2):
        print(f"{prefix}|RDMEM--------------------------|")
        print(f"{prefix}| {str(self.Output())}")
        
    
    
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