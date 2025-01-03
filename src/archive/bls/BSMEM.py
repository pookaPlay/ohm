from bls.DataIO import DeserializeLSBTwos, DeserializeMSBTwos
from bls.DataIO import SerializeLSBTwos
import pickle

# Bit Serial Memory
class BSMEM():

    # ...
    def __init__(self, D, K):
        self.D = D
        self.K = K        
        self.Reset()

    def ResetIndex(self):
        self.ri = 0
        self.rib = self.K-1
        self.wi = 0

    def ReverseContent(self):
        self.ResetIndex()
        #print("##########################")
        #print(self.mem)
        for d in range(len(self.mem)):
            temp = self.mem[d].copy()
            #print(">>>>>")
            #print(temp)
            #print(">>>>>")
            for i in range(len(temp)):
                self.mem[d][i] = temp[len(temp)-i-1]
            #print(self.mem[d])
            #print(">>>>>")
            #self.mem[d] = self.mem[d].reverse()

        #print("##########################")
        #print(self.mem)

    def Reset(self):
        self.ResetIndex()        
        self.mem = [list(self.K * [0]) for _ in range(self.D)]        
        self.memhack = [0] * self.D

    def GetLSBIntsHack(self):        
        return self.memhack

    def SetLSBIntsHack(self, vals):        
        self.memhack = vals.copy()

    def LoadList(self, data):                
        self.Reset()       
        self.memhack = data.copy()     
        
        for n in range(len(data)):
            self.mem[n] = SerializeLSBTwos(data[n], self.K)            

    def LoadScalar(self, scalar):            
        self.Reset()            
        for n in range(len(self.mem)):
            for i in range(len(self.mem[n])):
                self.mem[n][i] = scalar            
    

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
            #print(f"Saving to loc {self.wi}")
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



    def Save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def Load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def GetLSBInts(self):
        result = list()
        for di in range(len(self.mem)):
            #print(f"DESERIALIZING: {self.mem[di]}")
            result.append(DeserializeLSBTwos(self.mem[di]))
        
        return result

    def GetMSBInts(self):
        result = list()
        for di in range(len(self.mem)):            
            result.append(DeserializeMSBTwos(self.mem[di]))
        
        return result



    def Print(self, prefix="", verbose=2):        
        print(f"{prefix} MEM Size: {self.D} Depth: {self.K} ")
        #for i in range(self.K)
        
        for di in range(len(self.mem)):            
            msbInt = DeserializeMSBTwos(self.mem[di])
            lsbInt = DeserializeLSBTwos(self.mem[di])
            spacedInt = "{:>4}".format(lsbInt)

            temps = str(f"{prefix}: {spacedInt} : [")
            for ki in range(len(self.mem[di])):
                if ki == self.wi:                    
                    if ki == self.ri:
                        temps += ">(" + str(self.mem[di][ki]) + ")<"
                    else:
                        temps += ">" + str(self.mem[di][ki]) + "<"
                elif ki == self.ri:
                    temps += "(" + str(self.mem[di][ki]) + ")"
                elif ki == self.rib:
                    temps += "<" + str(self.mem[di][ki]) + ">"                        
                else:
                    temps += " " + str(self.mem[di][ki])
            spacedInt = "{:>4}".format(msbInt)
            print(temps + "] : " + spacedInt) 
