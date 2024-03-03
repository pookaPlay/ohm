from DataIO import DeserializeLSBTwos, DeserializeMSBTwos
from DataIO import SerializeMSBTwos

# Bit Serial Memory
class BSMEM():

    def __init__(self, D, K):
        self.D = D
        self.K = K
        
        self.Reset()
        

    def Load(self, data):                
        for n in range(len(data)):
            self.mem[n] = SerializeMSBTwos(data[n], self.K)
            self.mem[n].reverse()

    def Reset(self):
        self.ri = 0
        self.rib = self.K-1
        self.wi = 0        

        self.mem = [list(self.K * [0]) for _ in range(self.D)]        
    

    def Output(self, di=-1):
        if di != -1:
            return self.mem[di][self.ri]
        else:
            return [self.mem[ai][self.ri] for ai in len(self.mem)]
    
    def OutputMSB(self, di=-1):
        if di != -1:
            return self.mem[di][self.rib]
        else:
            return [self.mem[ai][self.rib] for ai in len(self.mem)]
            
    def Step(self, input = None):

        if input is not None:
            for di in range(len(input)):                
                self.mem[di][self.wi] = input[di]
            
        self.ri += 1                
        if self.ri == self.K:
            self.ri = 0

        self.rib -= 1            
        if self.rib == -1:
            self.rib = self.K-1        

        self.wi += 1
        if self.wi == self.K:
            self.wi = 0

    def GetLSBInts(self):
        result = list()
        for di in range(len(self.mem)):
            print(f"DESERIALIZING: {self.mem[di]}")
            result.append(DeserializeLSBTwos(self.mem[di]))
        
        return result

    def GetMSBInts(self):
        result = list()
        for di in range(len(self.mem)):            
            result.append(DeserializeMSBTwos(self.mem[di]))
        
        return result

    def Print(self, prefix="", verbose=2):
        print(f"{prefix}|BSMEM--------------------------|")
        print(f"{prefix}|Size: {self.D} Depth: {self.K} ")
        #for i in range(self.K)

        for di in range(len(self.mem)):
            print(f"{prefix}{self.mem[di]}")
        """ 

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
        """
                

    
    
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