from bls.DataIO import DeserializeLSBTwos

class DataWriter():
    def __init__(self):
        self.result = list()
        self.lastLSB = -1
        self.lastMSB = -1
        self.lastResult = -1
        self.finalResult = list()

    def Print(self):        
        print(f"Data Writer: {self.lastResult}")

    def PrintAll(self):        
        print(f"Data Writer: {self.finalResult}")

    def Output(self):
        return(self.finalResult)

    def Reset(self):
        self.result = list()        
        self.finalResult = list()
        self.lastLSB = -1
        self.lastMSB = -1
        self.lastResult = -1


    def Step(self, x, lsb, msb):
                
        if (lsb == 1):
            self.result = list()

        self.result.append(x)

        if (msb == 1):        
            self.lastResult = DeserializeLSBTwos(self.result)                
            print(f"   Writer: {self.lastResult} from {self.result}")
            self.finalResult.append(self.lastResult)

        
        

