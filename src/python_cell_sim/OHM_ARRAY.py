import argparse
import numpy as np
from OHM import OHM
from lsbSource  import lsbSource
import matplotlib.pyplot as plt

class OHM_ARRAY:

    def __init__(self, N, K, ptf="max", debugDone=0, verbosity=0) -> None:
        
        self.verbosity = verbosity
        self.K = K
        self.N = N                
        self.D = N
        self.D2 = self.D*2
        self.DD = self.D2+2                

        self.inputs = [lsbSource() for i in range(self.D)]
        self.inputBiases = [lsbSource() for i in range(self.D2)]

        self.ohmArray = [OHM(ptf=ptf, D=self.D, debugDone=debugDone, debugIndex=i) for i in range(self.D)]                
        self.stateBiases = [[lsbSource() for i in range(self.D2)] for j in range(self.D)]        
        
        self.fig = None
        self.ax = None

    def Reset(self) -> None:                
        [ohmArray.Reset() for ohmArray in self.ohmArray] 
        [[stateBias.Reset() for stateBias in stateBiases] for stateBiases in self.stateBiases]        
        [input.Reset() for input in self.inputs]
        [bias.Reset() for bias in self.inputBiases]
        

    def InitState(self, inputs) -> None:        
        assert len(inputs) == self.D
        
        for i in range(len(inputs)): 
            self.inputs[i].InitState(inputs[i], self.K)

        for ohmi in self.ohmArray:
            ohmi.ZeroState(self.K)

        for bi in range(len(self.inputBiases)):        
            if bi % 2 == 0:
                self.inputBiases[bi].ZeroState(self.K)
            else:
                self.inputBiases[bi].OneState(self.K)

        for stateBiases in self.stateBiases:
            for si in range(len(stateBiases)):
                if si < self.D:
                    stateBiases[si].ZeroState(self.K)
                else:
                    stateBiases[si].OneState(self.K)
                        

    def Calc(self, bi) -> None:
        # def Calc(self, u, ulsb, x, xlsb, ww) -> None:        
        states = [ohmArray.Output() for ohmArray in self.ohmArray]
        stateLsbs = [ohmArray.lsbOut() for ohmArray in self.ohmArray]
        
        if self.verbosity > 2:
            #print(f"Step {bi}: States: {states} -- State LSBs: {stateLsbs}")
            pass

        for ni in range(self.D):
            input = self.inputs[ni].Output()
            inputLsb = self.inputs[ni].LsbOut()
            myStateBiases = [stateBias.Output() for stateBias in self.stateBiases[ni]]
            myInputBiases = [self.inputBiases[ni*2].Output(), self.inputBiases[ni*2+1].Output()]
            ww = myStateBiases + myInputBiases
            
            self.ohmArray[ni].Calc(input, inputLsb, states, stateLsbs, ww)
                    

    def Step(self) -> None:
        for ohm in self.ohmArray:            
            ohm.Step()

        for stateBiases in self.stateBiases:
            for stateBias in stateBiases:
                stateBias.Step()
        
        for input in self.inputs:
            input.Step()

        for bias in self.inputBiases:
            bias.Step()        


    def Print(self, prefix="", showInput=1, showOutput=1) -> None:

        for ohmArray, input in zip(self.ohmArray, self.inputs):
            input.Print(prefix)
            ohmArray.Print(prefix, showInput=showInput, showOutput=showOutput)

    def Run(self, NSteps) -> None:      
        # self.ohmArray[1].Print()

        history = list() 
        states, lenn, switchStep = zip(*[ohmArray.GetState() for ohmArray in self.ohmArray])
        latestStates = list(states)
        latestLength = list(lenn)        

        for bi in range(NSteps):
            history.append(states)
            #print(f"= STEP {bi} ======================================")            
            self.Calc(bi)             

            if self.verbosity > 2:            
                plt.pause(0.1)
                plt.waitforbuttonpress()
            
            if self.verbosity > 1:
                self.Print(f"{bi}:", showInput=1, showOutput=1)


            states, lenn, switchStep = zip(*[ohmArray.GetState() for ohmArray in self.ohmArray])            
            # print(f"{bi}: {states} from {lenn} on switch {switchStep}")                                            
            fs = f""
            for ni in range(len(states)):                 
                if switchStep[ni] == 1:
                    latestStates[ni] = states[ni]
                    latestLength[ni] = lenn[ni]
                    fs += f"<{states[ni]}>"
                else:
                    fs += f"{latestStates[ni]}"  
                if ni < len(states)-1:
                    fs += f", "
            

            print(f"{bi}: [{fs}] from {latestLength}")
            self.Step()            

        print(f"History Length: {len(history)}")

        if self.fig is not None:
            print("Press any key to close the visualization window...")
            try:
                plt.waitforbuttonpress()
                plt.close(self.fig)
            except Exception:
                pass
        

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="OHM3 Array Simulation")
    parser.add_argument("--K", type=int, default=4, help="Number of bits (K)")
    parser.add_argument("--state0", type=int, nargs='+', default=[7, 1, 5, 2], help="Initial state list")
    parser.add_argument("--ptf", type=str, default="max", help="PTF function (min, max, median)")
    parser.add_argument("--debugDone", type=int, default=0, help="Debug done flag")
    parser.add_argument("--initMode", type=int, default=1, help="Init mode: 0 state or 1 for data in state")
    parser.add_argument("--steps", type=int, default=3, help="Number of steps to run")
    parser.add_argument("--v", type=int, default=0, help="Print the node")
    args = parser.parse_args()
    
    K = args.K
    state0 = args.state0    
    N = len(state0)    

    print(f"State0({N}): {state0}")
    ohm = OHM_ARRAY(N, K, ptf=args.ptf, debugDone=args.debugDone, verbosity=args.v)

    ohm.InitState(state0)
    ohm.Run(args.steps)
