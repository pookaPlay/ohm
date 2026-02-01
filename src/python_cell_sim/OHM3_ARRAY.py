import argparse
import numpy as np
from OHM3 import OHM3
from lsbSource  import lsbSource
import matplotlib.pyplot as plt

class OHM3_ARRAY:

    def __init__(self, N, K, ptf="max", debugDone=0):
        self.D = 3
        self.K = K
        self.N = N                
        
        self.ohmArray = [OHM3(ptf=ptf, debugDone=debugDone, debugIndex=i) for i in range(self.N)]                
        self.biases = [lsbSource() for i in range(self.N)]
        self.fig = None
        self.ax = None

    def Reset(self) -> None:                
        [ohmArray.Reset() for ohmArray in self.ohmArray] 


    def InitState(self, input, initMode=1) -> None:        
        assert(len(input) == self.N)
        # initMode 0: State init is zero with data in biases
        # initMode 1: State init is data with biases zero
        # input = [5, 7, 6, 3]
        for ni in range(self.N):
            
            leftInput = input[ni-1] if ni > 0 else input[self.N-1]
            rightInput = input[ni+1] if ni < self.N-1 else input[0]
            
            nodeInputs = [leftInput, input[ni], rightInput]
            self.ohmArray[ni].InitState(nodeInputs, self.K, initMode=initMode)                        
            #self.biases[ni].InitState(input[ni], self.K)
            if initMode == 0:
                self.biases[ni].InitState(input[ni], self.K)
            else:
                self.biases[ni].InitState(0, self.K)

        self.lastinputs = [ohmArray.Output() for ohmArray in self.ohmArray]            

    def Calc(self, bi) -> None:

        #hasOutput = [ohmArray.msb2lsb.GotOutput() for ohmArray in self.ohmArray]
        inputs = [ohmArray.Output() for ohmArray in self.ohmArray]
        lsbIns = [ohmArray.lsbOut() for ohmArray in self.ohmArray]
        biases = [bias.Output() for bias in self.biases]
        #for i in range(len(hasOutput)):
        #    if hasOutput[i] == 1:
        #        inputs[i] = self.ohmArray[i].Output()

        for ni in range(self.N):
            leftInput = inputs[ni-1] if ni > 0 else inputs[self.N-1]
            leftInputMSB = lsbIns[ni-1] if ni > 0 else lsbIns[self.N-1]            
            leftBias = biases[ni-1] if ni > 0 else biases[self.N-1]            
            
            rightInput = inputs[ni+1] if ni < self.N-1 else inputs[0]
            rightInputMSB = lsbIns[ni+1] if ni < self.N-1 else lsbIns[0]            
            rightBias = biases[ni+1] if ni < self.N-1 else biases[0]            
            
            nodeInputs = [leftInput, inputs[ni], rightInput]
            lsbInputs = [leftInputMSB, lsbIns[ni], rightInputMSB]
            biasInputs = [leftBias, biases[ni], rightBias]
            
            self.ohmArray[ni].Calc(nodeInputs, biasInputs, lsbInputs)        

    def Step(self) -> None:
        for ohmArray in self.ohmArray:            
            ohmArray.Step()
        for bias in self.biases:
            bias.Step()

    def Draw(self) -> None:
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()

        self.ax.clear()
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ohmArray[0].Draw(ax=self.ax)
        plt.pause(0.1)


    def Print(self, prefix="", showInput=1, showOutput=1) -> None:
        for ohmArray in self.ohmArray:
            ohmArray.Print(prefix, showInput=showInput, showOutput=showOutput)
        print(f"{prefix} BIASES- ")
        for bias in self.biases:
            bias.Print(prefix)

    def Run(self, NSteps, verbose=0) -> None:      
        # self.ohmArray[1].Print()

        history = list() 
        states, lenn, switchStep = zip(*[ohmArray.GetState() for ohmArray in self.ohmArray])
        latestStates = list(states)
        latestLength = list(lenn)        

        for bi in range(NSteps):
            history.append(states)
            #print(f"= STEP {bi} ======================================")            
            self.Calc(bi)             

            if verbose > 2:
                self.Draw()
                self.Print(f"{bi}:", showInput=1, showOutput=1)
            elif verbose > 1:
                self.Print(f"{bi}:", showInput=1, showOutput=1)
            elif verbose > 0:
                self.Print(f"{bi}:", showInput=0, showOutput=1)


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
    ohm = OHM3_ARRAY(N, K, ptf=args.ptf, debugDone=args.debugDone)

    ohm.InitState(state0, initMode=args.initMode)
    ohm.Run(args.steps, args.v)
