from bls.DataIO import DeserializeLSBTwos

class DataWriter_v2():
    def __init__(self):
        self.result = list()
        self.lsb = list()
        self.lastResult = -1
        self.finalResult = list()

    def Print(self):        
        print(f"LSB   : {self.lsb}")
        print(f"Result: {self.result}")
        
        #print(f"Data Writer: {self.lastResult}")

    def PrintAll(self):        
        pass
        #print(f"Data Writer: {self.finalResult}")

    def Output(self):
        return(self.finalResult)

    def Reset(self):
        self.result = list()        
        self.lsb = list()
        self.finalResult = list()


    def Step(self, x, lsb):

        self.result.append(x)
        self.lsb.append(lsb)

        # if (lsb == 1):
        #     if len(self.result) > 0:
        #         self.lastResult = DeserializeLSBTwos(self.result)                
        #         print(f"   Writer: {self.lastResult} from {self.result} of length {len(self.result)}")
        #         self.finalResult.append(self.lastResult)
        #     self.result = list()
                
        # self.result.append(x)

        

        
        

