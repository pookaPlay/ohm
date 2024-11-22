from bls.OHM_NODE_TEST import RunNode

def test_RunNode():
    #ptf = "median"
    ptf = "max"    
    NBitsIn = 4
    NBitsOut = 4
    NSteps = 16
    rep = 10
    
    input = [3, 7, 1]  # needs 2 bits 
    input = [4, 7, 1]  # needs 3 bits
    #input = [6, 7, 1]  # needs 4 bits
    input = [input.copy() for _ in range(rep)]

    expected = [0, 2, 2, 2, 2] 
    ret = RunNode(input, ptf, NBitsIn, NBitsOut, NSteps)    
    print(f"Ret: {ret}")


def test_RunNode8():
    #ptf = "median"
    ptf = "max"    
    NBitsIn = 8
    NBitsOut = 8
    NSteps = 64
    rep = 10
        
    input = [33, 4, 1]  # Done at step 5 (2 bits)
    input = [input.copy() for _ in range(rep)]

    expected = [0, 2, 2, 2, 2] 
    ret = RunNode(input, ptf, NBitsIn, NBitsOut, NSteps)    
    print(f"Ret: {ret}")



if __name__ == "__main__":
    print(f"#######################################")
    print(f"NODE TEST BEGIN")
    
    test_RunNode()
    #test_RunNode8()
    