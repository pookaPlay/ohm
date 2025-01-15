from bls.MultiDataWriter import MultiDataWriter

class LayeredMultiDataWriter():
    def __init__(self, L, W):
        self.writers = [MultiDataWriter(W) for _ in range(L)]

    def Print(self):        
        [writer.Print() for writer in self.writers]        

    def PrintFinal(self):        
        [writer.PrintFinal() for writer in self.writers]        

    def Output(self):
        result = [writer.Output() for writer in self.writers]
        return(result)

    def Reset(self):
        [writer.Reset() for writer in self.writers]

    def Step(self, x, lsb, debugTicksTaken):
        assert len(x) == len(self.writers)
        for l in range(len(x)):
            self.writers[l].Step(x[l], lsb[l], debugTicksTaken[l])            
    
    def BatchProcess(self):      
        [writer.BatchProcess() for writer in self.writers]
                
                    
        

        
        

