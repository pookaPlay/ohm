from RunOHMS import RunOHMS

def test_RUN_OHM():

    
    memK = 8
        
    input = [8, -8, 4, -4, 2, -2, 1, -1]
    #input += [-x for x in input]  # Add the negative values

    memD = len(input)
    numNodes = memD
    weights = numNodes * [0]    # bias weights        
        
    ohm = RunOHMS(memD, memK, numNodes, input, weights)    
    ohm.Run()
    
    #ohm.PrintMem()
    #result = ohm.lsbMem.GetLSBInts()
    #print(result)
    
    
    return

test_RUN_OHM()