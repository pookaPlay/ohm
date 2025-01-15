import progressbar
import sys
import glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import pdb
import math
import random
from operator import itemgetter
from fractions import gcd
#from cvxopt import matrix, solvers
#from pulp import *

ZERO_TOL = 1.0e-6

def FindCard(w, t):
    indices, wSorted = zip(*sorted(enumerate(w), reverse=False, key=itemgetter(1)))
    idx = list(indices) 
    wsort = list(wSorted)     
    N = len(wsort)
    #print(wsort)
    #print(idx)
        
    M = dict()
    def Card(psum, branch, level, sgn):                
        i = branch
        #if level == 5:
        #    print("FindTerm at level " + str(level) + " and branch " + str(i) + " with sgn " + str(sgn) + " from " + str(N)) 
        while i < N:            
            
            if psum + ws[i-1] < t:                
                #aset.append(i-1)                 
                Card(psum + ws[i-1], i+1, level+1, sgn)
                #aset.pop()
                #print("return")                
                #print(aset)                
                i = i + 1
            else:  
                for k in range(1, N-i+1):
                    cr = math.comb(N-i,k)
                    lkey = level + k - 1
                    if lkey in M[j]:
                        M[j][lkey] = M[j][lkey] + sgn * cr
                    else: 
                        M[j][lkey] = sgn * cr
                if sgn == 1:
                    Card(psum - wj, i, level, -sgn)
                i = N

    for ji in range(N):        
        #print("In " + str(ji))
        j = idx[ji] 
        ws = wsort.copy()        
        wj = ws[ji]
        del ws[ji]
        M[j] = dict()
        if wj >= t:
            M[j][0] = 1
        else:
            M[j][0] = 0
        #Card(wj, 0, 1, 1)
        Card(wj, 1, 1, 1)
    
    P = np.zeros((N, N))
    for jk in M:
        for ik in M[jk]:
            P[ik, jk] = M[jk][ik] * math.factorial(ik) * math.factorial(N-ik-1) / math.factorial(N)
    #print("\n")
    #print("P:")
    #print(P)
    sj = np.sum(P, 0)
    ri = np.sum(P, 1)    
    #print("sj:")
    #print(sj)
    #print("ri:")
    #print(ri)
    acc = 0
    for i in range(len(ri)):
        acc = acc + (i + 1) * ri[N-i-1]
    TT = 1.0 - (acc - 1.0 ) / (N - 1.0)
    #print("T:")
    #print(TT)

    return ri, sj, TT

def BinarySearchT(WW, numT, rt):

    totalW = np.sum(WW)
    maxW = np.max(WW)
    currentT = numT
    if currentT <= maxW:
        currentT = maxW+1
    print("Initial T is " + str(currentT))

    currentStep = 1
    notFound = True
    count = 0
    T0 = 0
    T1 = 0
    sj1 = 0
    rt1 = 0
    gotAbove = -1
    gotBelow = -1
 
    while notFound:
        print(f'T: {currentT} with step {currentStep} has {rt1} verse {rt} from {gotBelow} and {gotAbove}')
        
        ri1, sj1, rt1 = FindCard(WW, currentT)
        
        count = count + 1

        if (currentStep < 1.0) and (gotAbove > 0) and (gotBelow > 0): 
            if gotBelow > gotAbove:
                if rt1 > rt:
                    T1 = currentT
                    T1S = sj1
                    T1T = rt1
                else:
                    T0 = currentT
                    T0S = sj1
                    T0T = rt1

                if abs(T0-T1) > 1:
                    currentT = currentT + 1
                else:
                    #print(f'Transition is {T0} -> {T1} from {gotBelow} and {gotAbove}')            
                    notFound = False
            else:
                if rt1 > rt:
                    T1 = currentT
                    T1S = sj1
                    T1T = rt1
                else:
                    T0 = currentT
                    T0S = sj1
                    T0T = rt1

                if abs(T0-T1) > 1:
                    currentT = currentT - 1
                else:
                    #print(f'Transition is {T0} -> {T1} from {gotBelow} and {gotAbove}')            
                    notFound = False
        else:
            if abs(rt1 - rt) < ZERO_TOL:
                currentStep = 1
                if (gotAbove < 1) and (gotBelow < 1):
                    if currentT > 1:
                        gotAbove = count
                        T1 = currentT
                        T1S = sj1
                        T1T = rt1
                        currentT = currentT - 1
                    else:
                        gotBelow = count                        
                        T0 = currentT
                        T0S = sj1
                        T0T = rt1
                        currentT = currentT + 1
                elif (gotAbove > 0) and (gotBelow < 1):
                    gotAbove = count
                    T1 = currentT
                    T1S = sj1
                    T1T = rt1
                    if currentT > 1:                                                
                        currentT = currentT - 1
                    else:
                        gotBelow = count                        
                        T0 = currentT
                        T0S = sj1
                        T0T = rt1                        
                        notFound = False
                elif (gotAbove < 0) and (gotBelow > 0):
                    gotBelow = count
                    T0 = currentT
                    T0S = sj1
                    T0T = rt1
                    if currentT < totalW:                                                
                        currentT = currentT + 1
                    else:
                        gotAbove = count                        
                        T1 = currentT
                        T1S = sj1
                        T1T = rt1                        
                        notFound = False
                else:
                    if gotBelow > gotAbove:
                        T0 = currentT
                        T0S = sj1
                        T0T = rt1
                    else:
                        T1 = currentT
                        T1S = sj1
                        T1T = rt1
                    notFound = False
            elif rt1 > rt:
                gotAbove = count                  
                T1 = currentT
                T1S = sj1
                T1T = rt1                
                if gotBelow > 0:
                    mydiff = currentT - T0
                else:
                    mydiff = currentT - 1
                if mydiff >= 1:
                    if mydiff > 1:
                        currentStep = math.floor(mydiff / 2.0)
                        currentT = currentT - currentStep
                    else:
                        currentT = currentT - 1
                        currentStep = 0                    
                    if currentT < 1:
                        currentT = 1                            
                else:
                    currentStep = 0                
            else: 
                gotBelow = count
                T0 = currentT
                T0S = sj1
                T0T = rt1                
                if gotAbove > 0:
                    mydiff = T1 - currentT
                else:
                    mydiff = 1
                if mydiff >= 1:
                    if mydiff > 1:
                        currentStep = math.floor(mydiff / 2.0)
                        currentT = currentT + currentStep                        
                    else:
                        currentT = currentT + 1
                        currentStep = 0                                        
                    if currentT > totalW:
                        currentT = totalW                                                
                else:
                    currentStep = 0                

    return T0, T0S, T0T, T1, T1S, T1T

def FindIntWOS(win, t):
    es = 0.01
    er = 0.01
    
    NN = win.shape[0]
    FW =  np.zeros((NN))  
    
    # remove zero weights
    nzw = (win > ZERO_TOL).squeeze()
    cnzw = np.sum(nzw)
    nzwIndex = np.zeros((cnzw), dtype=np.int64) 
    upto = 0
    for wi in range(nzw.shape[0]):
        if nzw[wi]:
            nzwIndex[upto] = wi
            upto = upto + 1

    w = win[nzw] 
    numt = np.sum(w < t)

    print("Calculating sample probabilities")
    rio, sjo, rto = FindCard(w, t)    
    print(sjo)    
    print(rto)
    print("Estimating integer weights and threshold")
    #swin = np.sum(sjo)
    #fract = swin / rto
    #print(f'Sum is {swin} with t {rto} and frac {fract}')

    #print(rio)        
    # remove zero prob
    nzp = (sjo > ZERO_TOL).squeeze()

    cnzp = np.sum(nzp)

    nzpIndex = np.zeros((cnzp), dtype=np.int64) 
    upto = 0
    for wi in range(nzp.shape[0]):
        if nzp[wi]:
            nzpIndex[upto] = wi
            upto = upto + 1

    if cnzp < 1:
        print("Got an all zero case") 
        exit()
    elif cnzp == 1:        
        ni = nzpIndex[0]
        wi = nzwIndex[ni]        
        print("One input at " + str(wi))
        FW[wi] = 1.0
        FT = 0.0
        return FW, FT

    sj = sjo[nzp] 
    rt = rto    
    
    indices, sSorted = zip(*sorted(enumerate(sj), reverse=False, key=itemgetter(1)))
    idx = list(indices) 
    ssort = list(sSorted)     
    N = len(ssort)
    
    ss = ssort.copy()
    #print("Sorted")
    #print(ss)
    #print(idx)

    WW = np.zeros((N))        
    WI = np.zeros((N))        

    if ss[0] > ZERO_TOL:
        cw = 1
        WW[0] = 1
    else:
        cw = 0
    
    for i in range(1, N):
        if ss[i] > ss[i-1]:
            cw = cw + 1
        WW[i] = cw
    
    #print("Init WW")
    #print(WW)
    TT = 1
    if rt <= ss[0]:
        TT = 1
    else:
            
        notdone = True
        while notdone:
            notdone = False
            print('.', end='', flush=True)
            #print(WW)
            #print("Binary Search for T with " + str(numt))            
            T0, T0S, T0T, T1, T1S, T1T = BinarySearchT(WW, numt, rt)
            #print("T0: " + str(T0) + " T1: " + str(T1) )
            de0 = np.max(np.abs(T0S - ss))
            te0 = np.abs(T0T - rt)
            de1 = np.max(np.abs(T1S - ss))    
            te1 = np.abs(T1T - rt)
            
            if de0 < de1 and te0 < te1:
                #print("Accuracy T0: " + str(de0) + ", " + str(te0))
                if de0 < es and te0 < er:
                    TT = T0
                    break
            elif de0 > de1 and te0 > te1:
                #print("Accuracy T1: " + str(de1) + ", " + str(te1))
                if de1 < es and te1 < er:                
                    TT = T1
                    break
            else:
                TT = T1
                #print("Ambig Accuracy T0: " + str(de0) + ", " + str(te0) + " T1: " + str(de1) + ", " + str(te1))
            
            
            ds0 = ss - T0S 
            ds1 = ss - T1S 
            highestBelow = -math.inf 
            highestBelowIndex = -1
            for i in range(N-1, -1, -1):
                if (ds0[i] > 0) and (ds1[i] > 0):
                    val = min(ds0[i], ds1[i])
                    if val > highestBelow:
                        highestBelow = val
                        highestBelowIndex = i
            if highestBelowIndex >= 0:
                WW[highestBelowIndex] = WW[highestBelowIndex] + 1
                #print("============ Incrementing " + str(highestBelowIndex))            
                notdone = True
            if notdone == False:
                pass
                #print("Ran out of oommm")
        print("\n")

    #print("Targets:")    
    #print(ss)    
    #print(rt)    

    #print("Compared to")
    #print(sj0)
    #print(te0)
    #print(sj1)
    #print(te1)
    

    #print("WW is")
    #print(WW)
    WI[idx] = WW
    #print("WI is")
    #print(WI)
    
    FT = TT
    #print(nzwIndex)
    for i in range(WI.shape[0]):
        ni = nzpIndex[i]
        wi = nzwIndex[ni]
        #print(ni)
        FW[wi] = WI[i]
    #print("And final")
    #print(FW)
    return FW, FT    



def FindMinTerms(w, t):
    indices, wSorted = zip(*sorted(enumerate(w), reverse=True, key=itemgetter(1)))
    idx = list(indices) 
    ws = list(wSorted) 
    N = len(ws)
    print(ws)
    aset = []
    
    branch = 0
    level = 0
    psum = 0
    minterms = []
    def FindTerm(psum, branch, aset):                
        i = branch
        while i < N:
            #print("FindTerm at level " + str(level) + " and branch " + str(i) + "") 
            if psum + ws[i] < t:                
                aset.append(i) 
                FindTerm(psum + ws[i], i+1, aset)
                aset.pop()
                #print("return")                
                #print(aset)                
                i = i + 1
            else:  
                #print("Last term for level " + str(level) + " and branch " + str(i))
                aset.append(i)
                #print("MinTerm: " + str(aset)) 
                iset = [idx[d] for d in aset]
                #print("IndexTerm: " + str(iset)) 
                minterms.append(iset)
                aset.pop()
                i = i + 1

    FindTerm(0, 0, aset)

    return minterms


# # only use non-zero elements
# nzw = (weight > ZERO_TOL).squeeze()
# cnzw = torch.sum(nzw)
# ctx.nzw = torch.zeros((cnzw)) 
# upto = 0
# for wi in range(cnzw):
#     if nzw[wi]:
#         ctx.nzw[upto] = wi
#         upto = upto + 1

# lweight = weight[0, nzw]

##############################################################
## Find min wos
if __name__ == '__main__':

    verbose = 0
    theSeed = 0
    random.seed(theSeed)
    torch.manual_seed(theSeed)
    
    #w = [1, 1, 1]
    #t = 2
    #w = [1.1, 1.2, 1.3]
    #t = 2.3
    #w = [1, 1, 3, 3, 5]
    #t = 8    
    #w = [2.8286e-02, 5.7857e-01, 1.2069e-01, 1.1145e+00, 3.4073e-01, 2.7815e+00, 2.0151e+00, 1.3109e+01, 7.0684e-01, 4.8815e+00]
    #t = 0.5831
    #w = [1.0197e+00, 5.8769e-02, 2.2438e+00, 7.7785e-01, 8.9192e+00, 1.5759e-01, 5.0447e+00, 1.4421e-01, 3.3851e-02, 2.5646e-01]
    #t = 1.5895
    
    w1 = np.array([10, 0, 10, 20, 0, 1e-4, 40])
    t1 = 2
    #w2 = np.array([1, 1, 1, 2, 3])
    #t2 = 5

    #w1 = np.array([0.58803266, 0.11673988, 0.6980023, 0.03421865, 1.049372,1.1613598, 0.943338, 1.27698])
    #t1 = 2.1625123
    w1 = np.array([0, 1.3896897, 1.1953336, 1.8096126, 2.1296542, 0.9661723, 0.0409216, 0.33562475])
    t1 = 2.3519576

    print(w1)        
    print(t1)
    WW1, TT1 = FindIntWOS(w1, t1)
    print(WW1)
    print(TT1)



# else:
#     TT = 1
#     notdone = True
#     while notdone:
#         notdone = False
#         T0 = 1
#         T1 = 1
#         ri1, sj1, rt1 = FindCard(WW, T1)
#         #print("Targets:")    
#         #print(ss)    
#         #print(rt)    

#         #print("Compared to")
#         #print(sj1)
#         #print(ri1)
#         #print(rt1)
#         ds1 = sj1 - ss
#         dt1 = rt1 - rt
#         TW = np.sum(WW)
#         #print("   T:  " + str(T1) + " out of " + str(TW) + " has " + str(rt1) + " verse " + str(rt))
#         while (rt1 < rt) and (T1 <= TW):
#             T0 = T1
#             T1 = T1 + 1
#             ds0 = ds1
#             sj0 = sj1
#             rt0 = rt1            
#             ri1, sj1, rt1 = FindCard(WW, T1)            
#             #print("   T:  " + str(T1) + " out of " + str(TW) + " has " + str(rt1) + " verse " + str(rt))
#             #print(rt1)
#             #print("verse")
#             #print(rt)

#         if T1 > TW:
#             T0 = TW
#             T1 = TW
#         #print(WW)
#         #print("T0 and T1: " + str(T0) + " and " + str(T1))
#         TT = T1
#         if T0 == T1:
#             #print("T0 and T1: " + str(T1)) 
#             TT = T1
#             break
            
#         #ri0, sj0, rt0 = FindCard(WW, T0)
#         #ri1, sj1, rt1 = FindCard(WW, T1)
#         de0 = np.max(np.abs(sj0 - ss))
#         te0 = np.abs(rt0 - rt)
#         de1 = np.max(np.abs(sj1 - ss))    
#         te1 = np.abs(rt1 - rt)
        
#         if de0 < de1 and te0 < te1:
#             #print("Accuracy T0: " + str(de0) + ", " + str(te0))
#             if de0 < es and te0 < er:
#                 TT = T0
#                 break
#         elif de0 > de1 and te0 > te1:
#             #print("Accuracy T1: " + str(de1) + ", " + str(te1))
#             if de1 < es and te1 < er:                
#                 TT = T1
#                 break
#         else:
#             pass
#             #print("Ambig Accuracy T0: " + str(de0) + ", " + str(te0) + " T1: " + str(de1) + ", " + str(te1))
        
        
#         ds0 = ss - sj0 
#         ds1 = ss - sj1 
#         highestBelow = -math.inf 
#         highestBelowIndex = -1
#         for i in range(N-1, -1, -1):
#             if (ds0[i] > 0) and (ds1[i] > 0):
#                 val = min(ds0[i], ds1[i])
#                 if val > highestBelow:
#                     highestBelow = val
#                     highestBelowIndex = i
#         if highestBelowIndex >= 0:
#             WW[highestBelowIndex] = WW[highestBelowIndex] + 1
#             #print("============ Incrementing " + str(highestBelowIndex))            
#             notdone = True
#         if notdone == False:
#             pass
#             #print("Ran out of oommm")