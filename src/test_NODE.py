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
        "debugDone" : 1,
        "debugTree" : 1,
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
    bZero = SerializeLSBTwos(0, K)
    bOne = SerializeLSBTwos(1, K)

    print(f"Defaults in offset code: {bZero} and {bOne}")
    
    bp = [lsbSource(K, bZero) for _ in range(D)]        
    bn = [lsbSource(K, bOne) for _ in range(D)]        

    ohm = OHM(param)         

    output = DataWriter()    
    
    data.Reset()        
    ohm.Reset()        
    output.Reset()
    [bpi.Reset() for bpi in bp]
    [bni.Reset() for bni in bn]

    print(f"== {0} ============================")
    data.Print()

    wpin = [bpi.Output() for bpi in bp]
    wnin = [bni.Output() for bni in bn]

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
                
        [bpi.Step() for bpi in bp]
        [bni.Step() for bni in bn]

        bpin = [bpi.Output() for bpi in bp]
        bnin = [bni.Output() for bni in bn]

        #print(f"Going into adders")
        #[wpi.Print() for wpi in wp]
        #[wni.Print() for wni in wn]

        ohm.Calc(data.Output(), bpin, bnin, data.lsbIn())        
        #ohm.Print("   ", 1)                
        output.Step(ohm.Output(), ohm.lsbOut(), ohm.debugTicksTaken)            
                
        ohm.Step()
        
    output.Print()
    output.BatchProcess()
    output.PrintFinal()
    
    #Data Writer: [-1, 3, -1, -1, -13, -1, -1, 51]
    #Data Length: [3, 7, 6, 3, 7, 6, 3, 7]    
    
    return output.Output()

test_NODE()
