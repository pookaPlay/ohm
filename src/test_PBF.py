#from bls.DataReader import DataReader
#from bls.DataWriter import DataWriter
from bls.PTF import PTF
#from bls.lsbSource import lsbSource
#from bls.DataIO import SerializeLSBTwos, SerializeMSBTwos, SerializeMSBOffset, SerializeLSBOffset
import random
import math

random.seed(0)
K = 16
D = 4

def test_PBF(x, ptf):
        
    max_value = 2 ** (K - 1)
    ptfBits = int(math.log2(D)+2)
    param = {
        "debugDone" : 1,
        "debugTree" : 0,
        "flagThresh" : -1,
        "ptf" : ptf,
        "nsteps" : 2*D,        
        "K" : K,
        "D" : D,
    }

    ptf = PTF(param)

    #print(x)
    #print(f"PTF BITS: {ptfBits}")
    
    ptf.Calc(x, 1)
    #ptf.Print()
    out = ptf.Output() 

    if param["debugTree"] == 0:   
        #print(f"Step {bi}: {out}")
        #for bi in range(param["nsteps"]):
        for bi in range(ptfBits-1):           
            ptfBits
            ptf.Step()
            ptf.Calc(x, 0)
            #ptf.Print() 
            out = ptf.Output()
            #print(f"Step {bi+1}: {out}")

        #print(f"FinalOut: {out}")
    
    return out


x = D * [0] + D * [0]
expected = 0
r = test_PBF(x, "min")
assert r == expected, f"Expected {expected}, got {r}"

x = D * [0] + D * [0]
expected = 0
r = test_PBF(x, "max")
assert r == expected, f"Expected {expected}, got {r}"

x = D * [0] + D * [1]
expected = 0
r = test_PBF(x, "min")
assert r == expected, f"Expected {expected}, got {r}"

x = D * [0] + D * [1]
expected = 1
r = test_PBF(x, "max")
assert r == expected, f"Expected {expected}, got {r}"

x = D * [0] + [0] + (D-1) * [1]
expected = 0
r = test_PBF(x, "median")
assert r == expected, f"Expected {expected}, got {r}"

x = D * [0] + [1] + (D-1) * [1]
expected = 1
r = test_PBF(x, "median")
assert r == expected, f"Expected {expected}, got {r}"

x = [1] + D * [0] + (D-1) * [1] 
expected = 1
r = test_PBF(x, "median")
assert r == expected, f"Expected {expected}, got {r}"
