from OHM3 import OHM3

class OHM3_ARRAY:

    def __init__(self, N, K):
        self.D = 3
        self.K = K
        self.N = N        
        self.ohmArray = [OHM3(ptf="max", debugDone=K, debugIndex=i) for i in range(self.N)]        
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

        self.ohmArray[1].Print()

        for bi in range(NSteps):
            print(f"========================================")
            print(f"== STEP {bi} ============================")            
            self.Calc(bi) 
            self.Step()            
            self.ohmArray[1].Print()


if __name__ == "__main__":
    N = 4
    K = 4
    state0 = [5, 2, 6, 3]
    print(f"State0: {state0}")
    ohm = OHM3_ARRAY(N, K)
    ohm.InitState(state0)
    ohm.Run(4)

