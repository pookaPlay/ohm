#from bls.DataReader import DataReader
#from bls.DataWriter import DataWriter
from bls.PTF import PTF
#from bls.lsbSource import lsbSource
#from bls.DataIO import SerializeLSBTwos, SerializeMSBTwos, SerializeMSBOffset, SerializeLSBOffset
import random
import math

random.seed(0)

def test_PBF():
        
    K = 16
    D = 4
    max_value = 2 ** (K - 1)
    ptfBits = math.log2(D)
    param = {
        "debugDone" : 1,
        "debugTree" : 0,
        "flagThresh" : -1,
        "ptf" : "max",
        "nsteps" : 2*D,        
        "K" : K,
        "D" : D,
    }

    ptf = PTF(param)

    x = D * [1] + D * [0]
    print(x)
    print(f"PTF BITS: {ptfBits}")
    
    ptf.Calc(x)
    ptf.Print()


    out = ptf.Output()    
    bi = 0
    print(f"Step {bi}: {out}")
    for bi in range(param["nsteps"]):
        ptf.Step()
        ptf.Calc(x)
        ptf.Print() 
        out = ptf.Output()
        print(f"Step {bi+1}: {out}")


test_PBF()
