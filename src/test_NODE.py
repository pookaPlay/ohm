from bls.DataReader import DataReader
from bls.DataWriter import DataWriter
from bls.OHM import OHM
from bls.lsbSource import lsbSource
from bls.DataIO import SerializeLSBTwos, SerializeMSBTwos, SerializeMSBOffset, SerializeLSBOffset

def test_NODE():
    
    param = {
        "ptf" : "med",
        "nsteps" : 12,        
        "K" : 4,
        "D" : 3,
    }

    #input = [6, 1, 2]  # produces alternating 2 bit outputs
    input = [6, 7, 5]  # works with max    
    input = [6, 4, 5]  # works with min and med!
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
    #print(f"--- OUT: {ohm.Output()}      LSB: {ohm.lsbOut()}")
    #ohm.Print("", 1)

    output.Step(ohm.Output(), ohm.lsbOut())    
    
    
    ohm.Step()                        

    for bi in range(param["nsteps"]):
        print(f"======================================")
        print(f"== {bi+1} ============================")
        data.Step()
        data.Print()
                
        [wpi.Step() for wpi in wp]
        [wni.Step() for wni in wn]

        wpin = [wpi.Output() for wpi in wp]
        wnin = [wni.Output() for wni in wn]

        #print(f"Going into adders")
        #[wpi.Print() for wpi in wp]
        #[wni.Print() for wni in wn]

        ohm.Calc(data.Output(), wpin, wnin, data.lsbIn())        
        ohm.Print("   ", 1)        
        #print(f"--- OUT: {ohm.Output()} LSB: {ohm.lsbOut()}")
        output.Step(ohm.Output(), ohm.lsbOut())            
                
        ohm.Step()
        
    output.Print()
    output.BatchProcess()
    output.PrintFinal()
    
    return output.Output()

test_NODE()
