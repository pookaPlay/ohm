from bls.DataReader import DataReader
from bls.DataWriter import DataWriter
from bls.OHM import OHM
from bls.lsbSource import lsbSource
from bls.DataIO import SerializeLSBTwos, SerializeMSBTwos, SerializeMSBOffset, SerializeLSBOffset
import random

random.seed(0)

def test_NODE():
        
    K = 16
    D = 16
    max_value = 2 ** (K - 1)

    param = {
        "debugDone" : 0,
        "flagThresh" : -1,
        "ptf" : "max",
        "nsteps" : K*4,        
        "K" : K,
        "D" : D,
    }

        
    input = [random.randint(1, max_value) for _ in range(D)]
    #input = [6, 7, 5]     
    #input = [1, 2, 4]  
    print(f"Input: {input}")
    
    input = [input.copy() for _ in range(10)]

    param["D"] = len(input[0])   
    D = param["D"]
    K = param["K"]

    data = DataReader(input, K, K)    

    #wZero = SerializeLSBOffset(0, K)
    #wOne = SerializeLSBOffset(1, K)
    wZero = SerializeLSBTwos(0, K)
    wOne = SerializeLSBTwos(1, K)

    print(f"Defaults in offset code: {wZero} and {wOne}")
    
    wp = [lsbSource(K, wZero) for _ in range(D)]        
    wn = [lsbSource(K, wOne) for _ in range(D)]        
    
    ohm = OHM(param)         

    output = DataWriter()    
    
    data.Reset()        
    ohm.Reset()        
    output.Reset()
    [wpi.Reset() for wpi in wp]
    [wni.Reset() for wni in wn]

    print(f"== {0} ============================")
    data.Print()

    wpin = [wpi.Output() for wpi in wp]
    wnin = [wni.Output() for wni in wn]

    #print(f"Going into adders")
    #[wpi.Print() for wpi in wp]
    #[wni.Print() for wni in wn]
    
    ohm.Calc(data.Output(), wpin, wnin, data.lsbIn())    
    #ohm.Print("", 1)

    output.Step(ohm.Output(), ohm.lsbOut(), ohm.debugTicksTaken)        
    
    ohm.Step()                        

    for bi in range(param["nsteps"]):
        print(f"======================================")
        print(f"== {bi+1} ============================")
        data.Step()
        #data.Print()
                
        [wpi.Step() for wpi in wp]
        [wni.Step() for wni in wn]

        wpin = [wpi.Output() for wpi in wp]
        wnin = [wni.Output() for wni in wn]

        #print(f"Going into adders")
        #[wpi.Print() for wpi in wp]
        #[wni.Print() for wni in wn]

        ohm.Calc(data.Output(), wpin, wnin, data.lsbIn())        
        #ohm.Print("   ", 1)                
        output.Step(ohm.Output(), ohm.lsbOut(), ohm.debugTicksTaken)            
                
        ohm.Step()
        
    output.Print()
    output.BatchProcess()
    output.PrintFinal()
    
    return output.Output()

test_NODE()
