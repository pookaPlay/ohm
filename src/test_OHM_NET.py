from bls.OHM_NET_TEST import RunNet

def test_RunNet():
    ptf = "max"
    #ptf = "max"    
    NBitsIn = 4
    NBitsOut = 4
    NSteps = 16
    rep = 10
    
    
    input = [6, 1, 2]  # produces alternating 2 bit outputs
    input = [6, 7, 5]  # works with max => 7    
    input = [6, -7, 5]  # 
    input = [input.copy() for _ in range(rep)]

    
    ret = RunNet(input, ptf, NBitsIn, NBitsOut, NSteps)    
    print(f"Ret: {ret}")


if __name__ == "__main__":    
    test_RunNet()    
    