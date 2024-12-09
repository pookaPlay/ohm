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
    #input = [25248, 27563, 2654, 16969, 31846, 26538, 19878, 31235, 23466, 14316, 9128, 18471, 9159, 6215, 16418, 9632, 20326, 6473, 4833, 21640, 30943, 6600, 23187, 28454, 20723, 13401, 31262, 29013, 17072, 4082, 921, 6113]   
    # K=16 D=32
    # MAX
    #Data Writer: [-1, 15, -1, 15, -1]
    #Data Length: [5, 11, 5, 11, 5]
    # MIN
    #Data Writer: [0, -16, 0, -16, 0]
    #Data Length: [5, 11, 5, 11, 5]    
    # MED
    #Data Writer: [16, -6, 16, -6, 16]
    #Data Length: [8, 8, 8, 8, 8]
    # MEDMAX
    #Data Writer: [34, -2, 34, -2, 34]
    #Data Length: [8, 8, 8, 8, 8]

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
