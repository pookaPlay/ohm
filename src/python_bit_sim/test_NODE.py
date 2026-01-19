from bls.DataReader import DataReader
from bls.DataWriter import DataWriter
#from bls.OHM_POS import OHM_POS
from bls.OHM_BLOB import OHM
from bls.lsbSource import lsbSource
from bls.DataIO import SerializeLSBTwos, SerializeMSBTwos, SerializeMSBOffset, SerializeLSBOffset
import random

random.seed(0)

def test_NODE(ptf):
        
    K = 16
    D = 16
    max_value = 2 ** (K - 1)

    param = {
        "debugDone" : 1,
        "debugTree" : 1,
        "flagThresh" : -1,
        "ptf" : ptf,
        "nsteps" : K*4,        
        "K" : K,
        "D" : D,
    }

        
    input = [random.randint(1, max_value) for _ in range(D)]
    
    print(f"Input: {input}")
    
    input = [input.copy() for _ in range(10)]

    param["D"] = len(input[0])   
    D = param["D"]
    K = param["K"]

    data = DataReader(input, K, K)    

    bZero = SerializeLSBTwos(0, K)
    bOne = SerializeLSBTwos(1, K)

    print(f"Defaults in offset code: {bZero} and {bOne}")
    
    bp = [lsbSource(K, bZero) for _ in range(D)]            

    ohm = OHM(param)         

    output = DataWriter()    
    
    data.Reset()        
    ohm.Reset()        
    output.Reset()
    [bpi.Reset() for bpi in bp]
    

    print(f"== {0} ============================")
    data.Print()

    wpin = [bpi.Output() for bpi in bp]    
    
    ohm.Calc(data.Output(), wpin, data.lsbIn())    

    output.Step(ohm.Output(), ohm.lsbOut(), ohm.debugTicksTaken)        
    
    ohm.Step()                        

    for bi in range(param["nsteps"]):
        #print(f"======================================")
        #print(f"== {bi+1} ============================")
        data.Step()
                
        [bpi.Step() for bpi in bp]        

        bpin = [bpi.Output() for bpi in bp]          

        ohm.Calc(data.Output(), bpin, data.lsbIn())        
        #ohm.Print("   ", 1)                
        output.Step(ohm.Output(), ohm.lsbOut(), ohm.debugTicksTaken)            
                
        ohm.Step()
        
    output.Print()
    output.BatchProcess()
    output.PrintFinal()
        
    return output.Output()

#out = test_NODE("min")
#expected = [0, -31846, -31846]
#assert out == expected, f"Expected {expected}, got {out}"

out = test_NODE("max")
expected = [0, 31262, 31262]
# or 
expected = [0, 31846, 31846]
assert out == expected, f"Expected {expected}, got {out}"

