from RunSortNetwork import RunSortNetwork

def test_RUN_SORT_NETWORK():

    
    memK = 8
    
    input = [64, 32, 16, 8, 4, 2, 1, -1, -2, -4, -8, -16, -32, -64]
    memD = len(input)
    numNodes = memD
    weights = numNodes * [0]            
        
    ohm = RunSortNetwork(memD, memK, numNodes, input, weights)    
    ohm.Run()
    
    #ohm.PrintMem()
    #result = ohm.lsbMem.GetLSBInts()
    #print(result)
    
    
    return

test_RUN_SORT_NETWORK()