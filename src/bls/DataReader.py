from DataIO import SerializeMSBTwos

class DataReader():
    def __init__(self, input=[5, 7, 6], NBitsIn=7, NBitsOut=8):
        
        self.NIn = NBitsIn
        self.NOut = NBitsOut

        self.D = len(input)
        self.bi = 0

        self.data = [SerializeMSBTwos(input[d], self.NIn) for d in range(self.D)]
        for d in range(self.D):
            self.data[d].reverse()

        #print(self.data)        
        self.slice = [self.data[d][self.bi] for d in range(self.D)]        

    def Reset(self):
        self.bi = 0
        self.slice = [self.data[d][self.bi] for d in range(self.D)]        
        self.lsb = 1
        self.msb = 0
 
    def Print(self):        
        print(f"Data Reader: {self.slice}     lsb({self.lsb}) msb({self.msb})")

    def Output(self):
        return self.slice

    def isLsb(self):
        return self.lsb

    def isMsb(self):
        return self.msb

    def Step(self):
        self.bi = self.bi + 1
        
        if self.bi < self.NIn:
            self.slice = [self.data[d][self.bi] for d in range(self.D)]
        elif self.bi < self.NOut:
            # sign extend
            self.slice = [self.data[d][self.NIn-1] for d in range(self.D)]
        else:            
            self.slice = self.D * [0]

        self.msb = 0
        self.lsb = 0

        if self.bi == self.NOut-1:
            self.msb = 1
        if self.bi == 0:
            self.lsb = 1
