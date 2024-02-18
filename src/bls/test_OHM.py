from OHM import OHM
from DataIO import DeserializeLSBTwos, DeserializeLSBOffset
from DataReader import DataReader
 
def test_OHM():

    NBitsIn = 4
    NBitsOut = 5
    input = [5, 2]
    D = len(input)
    NSteps = 15
    resultStart = 10
    resultEnd = 14

    data = DataReader(input, NBitsIn, NBitsOut)
    #result = DataWriter(input, NBitsIn, NBitsOut)
    data.Reset()    
    
    #print(inputs)
    result = list()   
    
    ohm = OHM(D, NBitsIn, NBitsOut)    
    ohm.Reset(data.Output())    
    
    ohm.Calc(data.Output())
    output = ohm.Output()    
    result.append(output)

    print(f"== {0} ============================")
    data.Print()
    print(f"----------------------")
    ohm.Print("", 1)

    ohm.Step(data.isMsb())
        

    for bi in range(NSteps):
        print(f"== {bi+1} ============================")
        data.Step()
        data.Print()
        ohm.Calc(data.Output())                
        result.append(ohm.Output())
        ohm.Print("", 1)
        ohm.Step(data.isMsb())
    
    #print(result) 
    result = result[resultStart:resultEnd]
    print(result) 
    output = DeserializeLSBOffset(result)    
    print(output)
    
    return
        

test_OHM()