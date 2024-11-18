from bls.OHM_NODE_TEST import RunNode



def test_RunNode():
    #ptf = "median"
    ptf = "max"    
    NBitsIn = 4
    NBitsOut = 4
    NSteps = 30
    rep = 10
    
    input = [4, 2, 1]
    
    input = [input.copy() for _ in range(rep)]

    expected = [0, 2, 2, 2, 2] 
    ret = RunNode(input, ptf, NBitsIn, NBitsOut, NSteps)    
    print(f"Ret: {ret}")


if __name__ == "__main__":
    print(f"#######################################")
    print(f"NODE TEST BEGIN")
    
    test_RunNode()
    