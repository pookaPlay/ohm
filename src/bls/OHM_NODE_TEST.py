from bls.DataReader import DataReader
from bls.DataWriter import DataWriter
from bls.OHM_v2 import OHM_v2


def RunNode(input, ptf, NBitsIn = 4, NBitsOut = 4, NSteps = 120):

    D = len(input[0])   

    data = DataReader(input, NBitsIn, NBitsOut)    
    ohm = OHM_v2(D, NBitsIn, NBitsOut, ptf=ptf)        
    output = DataWriter()    
    
    data.Reset()        
    ohm.Reset()        
    output.Reset()

    print(f"== {0} ============================")
    data.Print()

    ohm.Calc(data.Output(), data.lsbIn())
    print(f"--- PBF: {ohm.pbfOut()}  OUT: {ohm.Output()}      LSB: {ohm.lsbOut()}")
    ohm.Print("", 1)

    output.Step(ohm.Output(), ohm.lsbOut())    
    
    print(f"--- Stepping OHM with data lsb {data.lsbIn()}")
    ohm.Step(data.lsbIn())                    

    for bi in range(NSteps):
        print(f"======================================")
        print(f"== {bi+1} ============================")
        data.Step()
        data.Print()
        
        ohm.Calc(data.Output(), data.lsbIn())
        print(f"--- PBF: {ohm.pbfOut()}  OUT: {ohm.Output()}      LSB: {ohm.lsbOut()}")
        ohm.Print("", 1)
        output.Step(ohm.Output(), ohm.lsbOut())            
        
        print(f"--- Stepping OHM with data lsb {data.lsbIn()}")
        ohm.Step(data.lsbIn())
        
    output.PrintAll()
    
    return output.Output()



def test_RunNode():
    ptf = "median"
    #ptf = "max"
    input = [[1, 2, 4], [1, 2, 4], [1, 2, 4]]
    expected = [0, 2, 2, 2, 2] 
    ret = RunNode(input, ptf)    
    print(f"Ret: {ret}")


if __name__ == "__main__":
    print(f"#######################################")
    print(f"NODE TEST BEGIN")
    print(f"#######################################")
    test_RunNode()
    print(f"#######################################")
    print(f"NODE TEST END")
    print(f"#######################################")