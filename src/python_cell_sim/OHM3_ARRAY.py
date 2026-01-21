import numpy as np
from OHM3 import OHM3

class OHM3_ARRAY:

    def __init__(self, N, K, ptf="max"):
        self.D = 3
        self.K = K
        self.N = N        
        self.ohmArray = [OHM3(ptf=ptf, debugDone=K, debugIndex=i) for i in range(self.N)]        
        self.weights = [0] * self.D

    def InitState(self, input) -> None:        
        assert(len(input) == self.N)
        # input = [5, 7, 6, 3]
        for ni in range(self.N):
            leftInput = input[ni-1] if ni > 0 else input[self.N-1]
            rightInput = input[ni+1] if ni < self.N-1 else input[0]
            
            nodeInputs = [leftInput, input[ni], rightInput]
            self.ohmArray[ni].InitState(nodeInputs, self.K)                        

    def Calc(self, bi) -> None:

        inputs = [ohmArray.Output() for ohmArray in self.ohmArray]
        lsbIns = [ohmArray.lsbOut() for ohmArray in self.ohmArray]

        for ni in range(self.N):
            leftInput = inputs[ni-1] if ni > 0 else inputs[self.N-1]
            leftInputMSB = lsbIns[ni-1] if ni > 0 else lsbIns[self.N-1]            

            rightInput = inputs[ni+1] if ni < self.N-1 else inputs[0]
            rightInputMSB = lsbIns[ni+1] if ni < self.N-1 else lsbIns[0]            
            
            nodeInputs = [leftInput, inputs[ni], rightInput]
            lsbInputs = [leftInputMSB, lsbIns[ni], rightInputMSB]
                        
            self.ohmArray[ni].Calc(nodeInputs, self.weights, lsbInputs)        

    def Step(self) -> None:
        for ohmArray in self.ohmArray:            
            ohmArray.Step()

    def Reset(self) -> None:                
        [ohmArray.Reset() for ohmArray in self.ohmArray] 
        
    def Run(self, NSteps) -> None:      
        # self.ohmArray[1].Print()

        states = [ohmArray.GetState() for ohmArray in self.ohmArray]

        for bi in range(NSteps):
            print(f"= STEP {bi} ======================================")            
            print(f"= {states}")                        
            print(f"==================================================")

            self.Calc(bi) 

            lsbIns = [ohmArray.lsbOut() for ohmArray in self.ohmArray]
            for i, ohm in enumerate(self.ohmArray):
                #ohm.Print(f"Cell {i} lsb({ohm.lsbOut()})")
                if ohm.lsbOut() ==1:                      
                    newstate = ohm.GetState()
                    states[i] = newstate                    
            
            self.Step()            



if __name__ == "__main__":
    N = 4
    K = 4
    state0 = [5, 2, -1, 7]
    print(f"State0: {state0}")
    ohm = OHM3_ARRAY(N, K, ptf="min")
    ohm.InitState(state0)
    ohm.Run(16)

