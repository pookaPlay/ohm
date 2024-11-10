from bls.DataIO import DeserializeLSBTwos, DeserializeMSBTwos
from bls.DataIO import SerializeLSBTwos
import pickle

# Bit Serial Memory
class BSM():

    # ...
    def __init__(self, K):        
        self.K = K        
        self.Reset()

    def ResetIndex(self):
        self.ri = 0
        self.rib = self.K-1
        self.wi = 0

    def Reset(self):
        self.ResetIndex()        
        self.x = list(self.K * [0])
        self.c = list(self.K * [0])                

    def LoadList(self, data):                        
        self.Reset()               
        self.x = SerializeLSBTwos(data, self.K)            
        self.c[0] = 1

    def LoadScalar(self, scalar):            
        self.Reset()            
        for n in range(len(self.x)):            
            self.x[n] = scalar            
    
    def OutputLSB(self):
        return self.x[self.ri]        
    
    def OutputMSB(self):        
        return self.x[self.rib]
            
    def Step(self, input = None):

        if input is not None:            
            self.x[self.wi] = input
            
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
        result = DeserializeLSBTwos(self.x)        
        return result

    def GetMSBInts(self):        
        result = DeserializeMSBTwos(self.x)        
        return result

    def Print(self, prefix="", verbose=2):        
        print(f"{prefix} MEM Depth: {self.K} ")
        #for i in range(self.K)
                
        msbInt = DeserializeMSBTwos(self.x)
        lsbInt = DeserializeLSBTwos(self.x)
        spacedInt = "{:>4}".format(lsbInt)

        temps = str(f"{prefix}: {spacedInt} : [")
        for ki in range(len(self.x)):
            if ki == self.wi:                    
                if ki == self.ri:
                    temps += ">(" + str(self.x[ki]) + ")<"
                else:
                    temps += ">" + str(self.x[ki]) + "<"
            elif ki == self.ri:
                temps += "(" + str(self.x[ki]) + ")"
            elif ki == self.rib:
                temps += "<" + str(self.x[ki]) + ">"                        
            else:
                temps += " " + str(self.x[ki])

            spacedInt = "{:>4}".format(msbInt)
            print(temps + "] : " + spacedInt) 
