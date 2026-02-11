import argparse
import numpy as np
from OHM import OHM
from lsbSource  import lsbSource
import matplotlib.pyplot as plt

class OHM_ARRAY:

    def __init__(self, N, K, ptf="max", debugDone=0):
        self.D = N
        self.K = K
        self.N = N                
        self.WINDOW_WIDTH = 200        

        self.ohmArray = [OHM(D=self.D, ptf=ptf, debugDone=debugDone, debugIndex=i) for i in range(self.N)]                
        
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
            self.ohmArray[ni].InitState(ni, input, self.K, initMode=initMode)                        

        self.lastinputs = [ohmArray.Output() for ohmArray in self.ohmArray]            

    def Calc(self, bi) -> None:

        inputs = [ohmArray.Output() for ohmArray in self.ohmArray]
        lsbIns = [ohmArray.lsbOut() for ohmArray in self.ohmArray]        

        for ni in range(self.N):            
            self.ohmArray[ni].Calc(inputs, lsbIns)        

    def Step(self) -> None:
        for ohmArray in self.ohmArray:            
            ohmArray.Step()

    def Draw(self) -> None:
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()
        
        self.ax.clear()
        self.ax.set_xlim(0, self.WINDOW_WIDTH)
        self.ax.set_ylim(0, self.WINDOW_WIDTH)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ohmArray[0].Draw(self.ax, 0, 0, self.WINDOW_WIDTH, self.WINDOW_WIDTH)
        plt.pause(0.1)


    def Print(self, prefix="", showInput=1, showOutput=1) -> None:
        for ohmArray in self.ohmArray:
            ohmArray.Print(prefix, showInput=showInput, showOutput=showOutput)

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
                
                plt.pause(0.1)
                plt.waitforbuttonpress()

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
    ohm = OHM_ARRAY(N, K, ptf=args.ptf, debugDone=args.debugDone)

    ohm.InitState(state0, initMode=args.initMode)
    ohm.Run(args.steps, args.v)
