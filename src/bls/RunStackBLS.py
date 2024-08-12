from BSMEM import BSMEM
from OHM_STACK_BLS import OHM_STACK_BLS

class RunStackBLS:

    def __init__(self, param):
    
        self.NN = numNodes      # number of parallel nodes        
        self.numNodes = numNodes
        self.nodeD = nodeD

        self.memD = memD        
        self.K = 8
        self.input = input
        self.weights = weights

        self.lsbMem = BSMEM(self.memD, self.K)
        self.msbMem = BSMEM(self.memD, self.K)                
        self.dataMem = BSMEM(self.K, self.K)
        self.paramMem = BSMEM(self.K, self.K)
        self.ptfMem = BSMEM(self.K, self.K)        
                
        self.ohm = OHM_STACK_BLS(self.NN, self.memD, self.K, param)        

        self.Reset()

    def Reset(self) -> None:

        self.lsbMem.Reset()
        self.msbMem.Reset()
        self.dataMem.Reset()
        self.paramMem.Reset()
        self.ptfMem.Reset()        

        self.biases.Reset()                
        self.ptf.Reset()

        self.dataMem.LoadList(self.input)        
        self.paramMem.LoadList(self.weights)
        
        self.ptfMem.LoadList(list(self.numNodes*[1]))        
        self.ptfMem.Print("PTF")


        
    def Run(self) -> None:      
            

        print(f">>>>>>>>>>> LSB PASS ")
        self.RunLSB(0)
        self.lsbMem.ResetIndex()
        self.lsbMem.Print("LSB")
        print(f">>>>>>>>>>> MSB PASS ")
        self.RunMSB(0)
        #self.lsbMem.Print("LSB")
        self.msbMem.Print("MSB")
                
    
    def RunMSB(self, stepi=0) -> None:            
            ti = 0                        
            msb = 1
            print(f"     == {stepi}:{ti} ")
            
            self.ptf.Calc(self.lsbMem, self.ptfMem, msb)            
            self.denseOut = list(self.NN * [0])
            self.denseOut[0] = self.ptf.Output()
            
            print(f"     == {stepi}:{ti} {self.denseOut}")
            self.msbMem.Step(self.denseOut)            
            self.ptf.Step()                        

            #self.lsbMem.Print("LSB")
            #self.msbMem.Print("MSB")
            msb = 0                
            for ti in range(1, self.K):
                print(f"     == {stepi}:{ti} ")                
                
                self.lsbMem.Step()
                #self.ptfMem.Step()
                
                self.ptf.Calc(self.lsbMem, self.ptfMem, msb)
                self.denseOut = list(self.NN * [0])
                self.denseOut[0] = self.ptf.Output()
                print(f"     == {stepi}:{ti} {self.denseOut}")
                self.msbMem.Step(self.denseOut)                
                self.ptf.Step()                                                                        
                #self.lsbMem.Print("LSB")
                #self.msbMem.Print("MSB")



    def RunLSB(self, stepi=0) -> None:            
            ti = 0                        
            print(f"     == {stepi}:{ti} ")
            lsb = 1            

            self.biases.Calc(self.dataMem, self.paramMem, lsb)            
            self.denseOut = self.biases.Output()
            #self.Print()
            self.lsbMem.Step(self.denseOut)
            #self.lsbMem.Print()            
            self.biases.Step()                        
            
            lsb = 0
            for ti in range(1, self.K):
                print(f"     == {stepi}:{ti} ")                                                
                self.dataMem.Step()                
                self.paramMem.Step()
        
                self.biases.Calc(self.dataMem, self.paramMem, lsb)
                self.denseOut = self.biases.Output()
                #self.Print()
                self.lsbMem.Step(self.denseOut)
                #self.lsbMem.Print()
                self.biases.Step()                                                                        
                

    def SaveState(self, filename) -> None:
        # Call the static method Load
        self.dataMem.Save(filename + "_dataMem")
        self.paramMem.Save(filename + "_paramMem")
        self.lsbMem.Save(filename + "_lsbMem")
        #self.biases.Save(filename + "_biases")
            
    def LoadState(self, filename) -> None:
        self.dataMem = BSMEM.Load(filename + "_dataMem")
        self.paramMem = BSMEM.Load(filename + "_paramMem")
        self.lsbMem = BSMEM.Load(filename + "_lsbMem")        

    def PrintMem(self):        
        self.dataMem.Print("Data")
        self.paramMem.Print("Param")
        self.lsbMem.Print("Out ")

    def Print(self, prefix="", showInput=1) -> None:        
        print(prefix + f"RunWord:")
        print(prefix + f"  lsbOut: {self.denseOut}")        
        self.biases.Print(prefix + "  ", showInput)        
 