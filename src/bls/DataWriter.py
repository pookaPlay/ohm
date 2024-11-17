from bls.DataIO import DeserializeLSBTwos

class DataWriter():
    def __init__(self):
        self.result = list()
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
        self.lastResult = -1
        self.lastStep = 0


    def Step(self, x, lsb):

        if self.lastStep == 1:
            if (lsb == 1):
                print(f"Double Last Step so waiting")
                self.result = list()
        else:
            if (lsb == 1):
                if len(self.result) > 0:
                    self.lastResult = DeserializeLSBTwos(self.result)                
                    print(f"   Writer: {self.lastResult} from {self.result} of length {len(self.result)}")
                    self.finalResult.append(self.lastResult)
                self.result = list()
        
        self.lastStep = lsb
        
        self.result.append(x)

        

        
        

