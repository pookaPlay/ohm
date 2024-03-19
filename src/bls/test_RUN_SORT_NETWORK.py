from RunSortNetwork import RunSortNetwork

def test_RUN_SORT_NETWORK():

    
    memK = 8
    
    input = [64, 32, 16, 8, 4, 2, 1, -1, -2, -4, -8, -16, -32, -64]
    memD = len(input)
    numNodes = memD
    weights = numNodes * [0]            
        
    ohm = RunSortNetwork(memD, memK, numNodes, input, weights)    
    ohm.Run()  
    
    expected = [64, 32, 16, 8, 4, 2, 1, -1, -2, -4, -8, -16, -32, -64].reverse()
    expectedTicks = [2, 3, 4, 5, 6, 7, 7, 6, 6, 5, 4, 3, 2, 1]
    
    return

test_RUN_SORT_NETWORK()
