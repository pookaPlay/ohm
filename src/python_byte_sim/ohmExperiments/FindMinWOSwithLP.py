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
#from cvxopt import matrix, solvers
from pulp import *

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


def FindCard(w, t):
    indices, wSorted = zip(*sorted(enumerate(w), reverse=False, key=itemgetter(1)))
    idx = list(indices) 
    wsort = list(wSorted)     
    N = len(wsort)
    #print(wsort)
    #print(idx)
    aset = []
    minterms = []
    M = dict()
    def Card(psum, branch, level, sgn, aset):                
        i = branch
        while i < N:
            #print("FindTerm at level " + str(level) + " and branch " + str(i) + "") 
            #print(i)
            if psum + ws[i-1] < t:                
                aset.append(i-1) 
                Card(psum + ws[i-1], i+1, level+1, sgn, aset)
                aset.pop()
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
                    Card(psum - wj, i, level, -sgn, aset)
                i = N

    for ji in range(N):
        j = idx[ji] 
        ws = wsort.copy()        
        wj = ws[ji]
        del ws[ji]
        M[j] = dict()
        if wj >= t:
            M[j][0] = 1
        else:
            M[j][0] = 0
        Card(wj, 1, 1, 1, aset)

    P = np.zeros((N, N))
    for jk in M:
        for ik in M[jk]:
            P[ik, jk] = M[jk][ik] * math.factorial(ik) * math.factorial(N-ik-1) / math.factorial(N)

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


def FindIntWOS(w, t):
    es = 0.01
    er = 0.01

    ri, sj, rt = FindCard(w, t)
    print("Probs for input")
    print(sj)
    print(rt)
    indices, sSorted = zip(*sorted(enumerate(sj), reverse=False, key=itemgetter(1)))
    idx = list(indices) 
    ssort = list(sSorted)     
    N = len(ssort)
    
    ss = ssort.copy()
    print(ss)
    print(idx)

    WW = np.ones((N))        
    WI = np.ones((N))        
    cw = 1
    for i in range(1, N):
        if ss[i] > ss[i-1]:
            cw = cw + 1
        WW[i] = cw

    TT = 1
    notdone = True
    while notdone:
        notdone = False
        T0 = 1
        T1 = 1
        ri1, sj1, rt1 = FindCard(WW, T1)
        ds1 = sj1 - ss
        dt1 = rt1 - rt

        while rt1 < rt:
            T0 = T1
            T1 = T1 + 1
            ds0 = ds1
            sj0 = sj1
            rt0 = rt1
            ri1, sj1, rt1 = FindCard(WW, T1)
        
        WI[idx] = WW
        print(WW)
        print("T0 and T1: " + str(T0) + " and " + str(T1))

        #ri0, sj0, rt0 = FindCard(WW, T0)
        #ri1, sj1, rt1 = FindCard(WW, T1)
        de0 = np.max(np.abs(sj0 - ss))
        te0 = np.abs(rt0 - rt)
        de1 = np.max(np.abs(sj1 - ss))    
        te1 = np.abs(rt1 - rt)

        if de0 < es/2 and te0 < er/2:
            print("exit 0")
            WI[idx] = WW
            return WI, T0
        if de1 < es/2 and te1 < er/2:
            print("exit 1")
            WI[idx] = WW
            return WI, T1
        
        # jset = list()
        # for j in range(N):
        #     if ds0[j] <= 0 and ds1[j] <= 0:
        #         if not((ds0[j] == 0) and (ds1[j] == 0)):
        #             jset.append(j)
        # if len(jset) > 0:
        #     minj = 0
        #     jval = math.inf
        #     for j in jset:
        #         maxv = max(ds0[j], ds1[j])
        #         if maxv < jval:
        #             jval = maxv
        #             minj = j
        #     WW[minj] = WW[minj] + 1
        #     notdone = True

        ds0 = ss - sj0 
        ds1 = ss - sj1 
        highestBelow = -math.inf 
        highestBelowIndex = -1
        for i in range(N-1, -1, -1):
            if (ds0[i] > 0) and (ds1[i] > 0):
                val = min(ds0[i], ds1[i])
                if val > highestBelow:
                    highestBelow = val
                    highestBelowIndex = i
        if highestBelowIndex >= 0:
            #print("Updating weight " + str(highestBelowIndex))
            #print(ss)
            #print(sj0)
            #print(sj1)
            #print(ds0)
            #print(ds1)
            WW[highestBelowIndex] = WW[highestBelowIndex] + 1
            notdone = True

    print("No stopiing condition!")
    WI[idx] = WW
    print(WI)
    return WW, TT    

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
    w = [1.0197e+00, 5.8769e-02, 2.2438e+00, 7.7785e-01, 8.9192e+00, 1.5759e-01, 5.0447e+00, 1.4421e-01, 3.3851e-02, 2.5646e-01]
    t = 1.5895
    N = len(w)
    print(w)        
    print(t)

    #minterms = FindMinTerms(w, t)
    #print(minterms)
    #M = len(minterms)
    
    WW, TT = FindIntWOS(w, t)
    print(WW)
    print(TT)
    
    exit()
    #exit()
    #cat='Integer'
    #lowBound=0,cat='Continuous')
    wvars =  [i for i in range(N)]
    w_vars = LpVariable.dicts("weights",wvars,lowBound=1,cat='Integer')
    #t_var = LpVariable("thresh", lowBound=0,cat='Integer')
    prob = LpProblem("myProblem", LpMinimize)
    #prob += lpSum([w_vars, t_var])
    prob += lpSum(w_vars)

    for term in minterms:
        A = [0 for i in range(N)]
        lt = len(term)                 
        for el in term:
            A[el] = 1        
        # Add constraint
        prob += lpSum([A[i] * w_vars[i] for i in wvars]) >= lt
    
    print(prob)
    #exit()
    status = prob.solve()
    print("Status: ", LpStatus[status])
    for v in prob.variables():    
        print(v.name, "=", v.varValue)

    exit()
    nA = np.ones((N, M))
    nB = np.zeros((M))
    nC = np.ones((N))

    termi = 0
    for term in minterms:        
        for el in term:
            nA[el, termi] = -1.0 #w[el]
        nB[termi] = -1.0
        termi = termi + 1

    A = matrix(nA)
    B = matrix(nB)
    C = matrix(nC)
    print(A)
    print(B)
    print(C)
    #A = matrix([ [-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, -1.0, -2.0] ])
    #b = matrix([ 1.0, -2.0, 0.0, 4.0 ])
    #c = matrix([ 2.0, 1.0 ])
    sol=solvers.lp(C,A,B)
    print(sol['x'])

# Weight: tensor([[ 9.4316,  0.6470,  0.1231,  1.8518],
#         [ 0.0000,  0.0000,  9.8163,  4.0869],
#         [13.9305,  0.2862,  0.0689,  6.3638],
#         [ 4.9135,  2.5520,  2.9236,  0.3823],
#         [ 0.0935, 19.6987,  6.3404,  0.0583]])
# Bias  : tensor([[0.3684],
#         [0.0032],
#         [0.1317],
#         [2.9119],
#         [0.0596]])
# Mask  : tensor([[-3.6757, -6.0769, -5.8796, -3.4744],
#         [ 0.5314,  0.4371,  3.8052,  3.9098],
#         [-3.8469, -6.2460, -6.0484, -3.6517],
#         [ 0.5883,  0.9188,  0.6766, -2.1685],
#         [-6.2052, -3.8057, -3.7380, -6.1163]])
# Weight: tensor([[2.8286e-02, 5.7857e-01, 1.2069e-01, 1.1145e+00, 3.4073e-01, 2.7815e+00,
#          2.0151e+00, 1.3109e+01, 7.0684e-01, 4.8815e+00],
#         [1.0197e+00, 5.8769e-02, 2.2438e+00, 7.7785e-01, 8.9192e+00, 1.5759e-01,
#          5.0447e+00, 1.4421e-01, 3.3851e-02, 2.5646e-01],
#         [3.6350e-04, 3.1870e-03, 3.6350e-04, 3.6350e-04, 1.8365e-03, 4.3198e+00,
#          2.3482e+00, 4.0201e+00, 7.3374e-01, 3.7054e+00],
#         [2.0212e+00, 3.3702e-03, 1.3853e+00, 1.0860e-02, 9.9063e+00, 2.0119e+00,
#          3.9317e+00, 3.5207e-04, 1.0826e+00, 0.0000e+00],
#         [2.3490e+00, 2.6130e-03, 9.3607e-01, 4.8176e-04, 1.7564e+00, 0.0000e+00,
#          2.8566e+00, 3.9897e+00, 6.9701e+00, 3.0399e+00]])
# Bias  : tensor([[0.5831],
#         [1.5895],
#         [0.0000],
#         [0.0093],
#         [0.0018]])
# Mask  : tensor([[-1.9165,  0.1033, -1.7308, -0.7883, -1.7691,  0.3723, -1.6663,  0.2290,
#          -1.0050,  0.2450],
#         [-3.3138, -1.5985, -3.1430, -2.9311, -0.7881, -2.0368, -1.0597, -2.2652,
#          -6.3137, -2.2047],
#         [-0.4913,  2.6960, -0.3616,  1.6961, -0.3077,  3.0728,  0.8804,  2.8118,
#           1.6316,  2.8247],
#         [-3.2366, -0.8136, -3.0271, -3.8280, -3.1270, -0.8219, -4.0245, -4.0705,
#          -3.5872, -0.8246],
#         [-1.1253, -0.4351, -1.0089, -0.6554, -1.0744,  0.0099, -1.3372, -0.6156,
#          -0.4596, -0.3988]])
# Weight: tensor([[1.9887, 6.6248, 0.0000, 0.7286, 0.6388, 0.0651, 0.0000, 8.6355, 3.9710,
#          1.9386]])
# Bias  : tensor([[0.0111]])
# Mask  : tensor([[-3.9593, -4.0868, -2.6936, -8.5946, -7.8497, -8.3790, -2.6062, -3.8905,
#          -5.8433, -5.2746]])

# def Card(asum, branch, aset, ws, t):
#     #print("Card in with " + str(asum) + " at branch " + str(branch) + " level " + str(level) + " sign " + str(sgn) ) 
#     nsum = asum
#     i = branch
#     while i < N:
#         if asum + ws[i] <= t:
#             #print("Recurse in top")
#             aset.append(ws[i])
#             nsum  = asum + ws[i]
#             bsum, bset = Card(nsum, i+1, aset, ws, t)
#             #print("Recurse out top")
#             i = i + 1
#         else:            
#             print(aset)
#             i = i + 1
#     return nsum, aset



def FindWasWOS(w, t):
    
    ri, sj, rt = FindCard(w, t)
    print("Probs for input")
    print(sj)
    print(rt)
    indices, sSorted = zip(*sorted(enumerate(sj), reverse=False, key=itemgetter(1)))
    idx = list(indices) 
    ssort = list(sSorted)     
    N = len(ssort)
    
    ss = ssort.copy()
    print(ss)

    WW = np.ones((N))
    TT = 0
    cw = 1
    for i in range(1, N):
        if ss[i] > ss[i-1]:
            cw = cw + 1
        WW[i] = cw

    notdone = True
    while notdone:
        notdone = False
        tapprox = sum(WW) * rt 
        t0 = math.floor(tapprox)
        t1 = math.ceil(tapprox)
        print("T0 and T1: " + str(t0) + " and " + str(t1))
        print(WW)
        ri0, sj0, rt0 = FindCard(WW, t0)
        #print("below:")
        #print(sj0) 
        #print(rt0)
        ri1, sj1, rt1 = FindCard(WW, t1)
        #print(sj1) 
        #print(rt1)
        for i in range(1, N):
            if (sj0[i] < ss[i]) and (sj1[i] < ss[i]):
                print("Both below: " + str(i))
                print(sj0)
                print(sj1)
                WW[i] = WW[i] + 1
                notdone = True
                break

    return sj, TT    
