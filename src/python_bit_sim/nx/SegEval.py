import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
#import itertools#some function definitions


DELTA_TOLERANCE = 1.0e-12

def DotProductLabels(a, b):
    ssum = 0.0    
    for key in a: 
        if key in b: 
            ssum = ssum + a[key]*b[key]
            
    return ssum

def GetNumberLabels(a):
    ssum = 0.0    
    for key in a:         
        ssum = ssum + a[key]
            
    return ssum

def CombineLabels(a, b):
    c = a.copy()
    
    for key in b:         
        if key in c:
            c[key] = c[key] + b[key]
        else:
            c[key] = b[key]
            
    return c

def FindRandCounts(WG, nlabels_dict, reverseSort):
    
    mySets = nx.utils.UnionFind()       
    labelCount = dict()
    for n in WG:        
        labelCount[n] = dict()
        labelCount[n][ nlabels_dict[n] ] = 1.0


    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    
    # True = +ve -> -ve
    sortedEdges = sorted(edgeWeights, reverse=reverseSort, key=lambda edge: edge[2]) 
    #print("MST edge affinities: " + str(sortedEdges[0][2]) + " -> " + str(sortedEdges[-1][2]) )
    mstEdges = list()    
    posCounts = list()
    negCounts = list()
    totalPos = 0.0
    totalNeg = 0.0    
    
    upto = 0

    for u, v, w in sortedEdges:
        su = mySets[u]
        sv = mySets[v]
        if su != sv:
            
            labelAgreement = DotProductLabels( labelCount[su], labelCount[sv] )
            numLabelsU = GetNumberLabels( labelCount[su] )
            numLabelsV = GetNumberLabels( labelCount[sv] )
            labelDisagreement = numLabelsU * numLabelsV - labelAgreement
            mstEdges.append( (u,v) )            
            posCounts.append(labelAgreement)
            negCounts.append( labelDisagreement)

            #print(str(u) + " has " str(numLabelsU) + ", " + str(v)  and " + str(numLabelsV) + " got to " + str(labelAgreement) + " and " + str(labelDisagreement))    
            totalPos = totalPos + posCounts[-1] 
            totalNeg = totalNeg + negCounts[-1] 
                        
            allLabels = CombineLabels(labelCount[su], labelCount[sv])
            #numAll = GetNumberLabels(allLabels)
            mySets.union(u, v)
            labelCount[ mySets[u] ] = allLabels.copy()

        upto = upto + 1
    #print(mstEdges)    
    #print("Total Pos: " + str(totalPos) + " Neg: " + str(totalNeg)) 
    return(posCounts, negCounts, mstEdges, totalPos, totalNeg)

def FindMinEnergyAndRandCounts(WG, nlabels_dict, evalw=None):
    
    mySets = nx.utils.UnionFind()   
    nextNode = dict()
    labelCount = dict()

    for n in WG:
        nextNode[n] = mySets[n]
        labelCount[n] = dict()
        labelCount[n][ nlabels_dict[n] ] = 1.0

    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    

    if evalw is None:
        evalWeights = edgeWeights
        evalw = WG
    else:
        evalWeights = [(u,v,w) for (u,v,w) in evalw.edges(data = 'weight')]    

    mstEdges = list()
    mstInd = list()
    mstEdgeWeights = list()
    posCountsRand = list()
    negCountsRand = list()
    totalPosRand = 0.0
    totalNegRand = 0.0    
    

    totalPos = sum([d[2] for d in evalWeights if d[2] > 0.0])
    totalNeg = sum([d[2] for d in evalWeights if d[2] < 0.0])
    accTotal = [0]*len(evalWeights)

    accTotal[0] = totalPos + totalNeg
    #print("Energy 0: " + str(accTotal[0]) + " from Pos: " + str(totalPos) + ", Neg: " + str(totalNeg))

    sortedEdges = sorted(edgeWeights, reverse=True, key=lambda edge: edge[2]) 
    
    ei = 1      # edge index
    lowE = accTotal[0]
    lowT = sortedEdges[0][2] + DELTA_TOLERANCE
    myMstEdge = -np.ones( (WG.number_of_edges()), np.float32)
    upto = 0

    for u, v, w in sortedEdges:
        #if w <= 0.0:
         #   break
        
        su = mySets[u]
        sv = mySets[v]
        if su != sv:
            accWeight = 0.0
            # traverse nodes in u and look at edges
            # if fully connected we should probably traverse nodes u and v instead
            done = False
            cu = u
            while not done:
                for uev in WG[cu]:                
                    if mySets[uev] == mySets[v]:
                        accWeight = accWeight + evalw[cu][uev]['weight']
                        
                cu = nextNode[cu]
                if cu == u:
                    done = True

            labelAgreement = DotProductLabels( labelCount[su], labelCount[sv] )
            numLabelsU = GetNumberLabels( labelCount[su] )
            numLabelsV = GetNumberLabels( labelCount[sv] )
            labelDisagreement = numLabelsU * numLabelsV - labelAgreement
            posCountsRand.append(labelAgreement)
            negCountsRand.append( labelDisagreement)

            #print(str(u) + " has " str(numLabelsU) + ", " + str(v)  and " + str(numLabelsV) + " got to " + str(labelAgreement) + " and " + str(labelDisagreement))    
            totalPosRand = totalPosRand + posCountsRand[-1] 
            totalNegRand = totalNegRand + negCountsRand[-1] 

            mstEdges.append( (u,v) )
            mstEdgeWeights.append( w )
                        
            allLabels = CombineLabels(labelCount[su], labelCount[sv])
            #numAll = GetNumberLabels(allLabels)            

            # Merge sets
            mySets.union(u, v)
            # Merge counts
            labelCount[ mySets[u] ] = allLabels.copy()            
            # Swap next pointers... this incrementally builds a pointer cycle around all the nodes in the component
            tempNext = nextNode[u]
            nextNode[u] = nextNode[v]
            nextNode[v] = tempNext

            accTotal[ei] = accTotal[ei-1] - accWeight            
            #print("Energy at threshold " + str(w) + ": " + str(accTotal[ei]))
            if accTotal[ei] < lowE:
                lowE = accTotal[ei]
                lowT = w - DELTA_TOLERANCE


            ei = ei + 1
        
    #print("Lowest Energy: " + str(lowE) + " at threshold " + str(lowThreshold)) 
    return(lowT, lowE, posCountsRand, negCountsRand, mstEdges, mstEdgeWeights, totalPosRand, totalNegRand)

def GetLabelsBelowThreshold(G,theta=0):
    lg = G.copy()    
    lg.remove_edges_from([(u,v) for (u,v,d) in  G.edges(data=True) if d['weight']<theta])
    L = {node:color for color,comp in enumerate(nx.connected_components(lg)) for node in comp}    
    return L

def GetNodeEdgeLabels(seg):
    
    G = nx.grid_2d_graph(seg.shape[0], seg.shape[1])
    nlabels = dict()
    elabels = dict()
    for (u,v,d) in G.edges(data = True):
        labelu = seg[u[0], u[1]]
        labelv = seg[v[0], v[1]]
        nlabels[u] = labelu
        nlabels[v] = labelv
        
        if abs(labelu - labelv) < 1.0:
            elabels[(u,v)] = 1.0
        else:
            elabels[(u,v)] = -1.0

    return (G, nlabels, elabels) 


def FindMinEnergyThreshold(WG, eval=None):
    
    mySets = nx.utils.UnionFind()   
    nextNode = dict()

    for n in WG:
        nextNode[n] = mySets[n]

    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    

    if eval is None:
        evalWeights = edgeWeights
        eval = WG
    else:
        evalWeights = [(u,v,w) for (u,v,w) in eval.edges(data = 'weight')]    

    totalPos = sum([d[2] for d in evalWeights if d[2] > 0])
    totalNeg = sum([d[2] for d in evalWeights if d[2] < 0])
    accTotal = [0]*len(evalWeights)

    accTotal[0] = totalPos + totalNeg
    #print("Energy 0: " + str(accTotal[0]) + " from Pos: " + str(totalPos) + ", Neg: " + str(totalNeg))

    sortedEdges = sorted(edgeWeights, reverse=True, key=lambda edge: edge[2]) 
    
    ei = 1      # edge index
    lowE = accTotal[0]
    lowT = sortedEdges[0][2] + DELTA_TOLERANCE

    for u, v, w in sortedEdges:
        if w <= 0.0:
            break

        if mySets[u] != mySets[v]:
            accWeight = 0.0
            # traverse nodes in u and look at edges
            # if fully connected we should probably traverse nodes u and v instead
            done = False
            cu = u
            while not done:
                for uev in WG[cu]:                
                    if mySets[uev] == mySets[v]:
                        accWeight = accWeight + eval[cu][uev]['weight']
                cu = nextNode[cu]
                if cu == u:
                    done = True

            # Merge sets
            mySets.union(u, v)
            # Swap next pointers... this incrementally builds a pointer cycle around all the nodes in the component
            tempNext = nextNode[u]
            nextNode[u] = nextNode[v]
            nextNode[v] = tempNext

            accTotal[ei] = accTotal[ei-1] - accWeight            
            #print("Energy at threshold " + str(w) + ": " + str(accTotal[ei]))
            if accTotal[ei] < lowE:
                lowE = accTotal[ei]
                lowT = w - DELTA_TOLERANCE


            ei = ei + 1
        
    #print("Lowest Energy: " + str(lowE) + " at threshold " + str(lowThreshold)) 
    return(lowT, lowE)



def FindBestRandThreshold(posCounts, negCounts, mstEdges, mstEdgeWeights):
    
    sumPosCount = 0.0
    sumNegCount = 0.0
	
    totalPos = sum(posCounts) 
    totalNeg = sum(negCounts)
    localPos = totalPos
    localNeg = 0.0

    # start off with every point in own cluster
    lowE = (localPos + localNeg) / (totalPos + totalNeg)
    lowT = mstEdgeWeights[0] + DELTA_TOLERANCE
    
    for i in range(len(posCounts)):
        localPos = localPos - posCounts[i]
        localNeg = localNeg + negCounts[i]

        newError = (localPos + localNeg) / (totalPos + totalNeg)
		
        if newError < lowE:
            lowE = newError
            lowT = mstEdgeWeights[i] - DELTA_TOLERANCE        

    return(lowT, lowE)

def FindRandErrorAtThreshold(WG, nlabels_dict, T):
    
    mySets = nx.utils.UnionFind()       
    labelCount = dict()

    for n in WG:        
        labelCount[n] = dict()
        labelCount[n][ nlabels_dict[n] ] = 1.0


    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    
    
    sortedEdges = sorted(edgeWeights, reverse=True, key=lambda edge: edge[2]) 

    mstEdges = list()    
    posCounts = list()
    negCounts = list()
    totalPos = 0.0
    totalNeg = 0.0    
    
    upto = 0

    for u, v, w in sortedEdges:
        su = mySets[u]
        sv = mySets[v]
        if su != sv:
            
            labelAgreement = DotProductLabels( labelCount[su], labelCount[sv] )
            numLabelsU = GetNumberLabels( labelCount[su] )
            numLabelsV = GetNumberLabels( labelCount[sv] )
            labelDisagreement = numLabelsU * numLabelsV - labelAgreement
            mstEdges.append( w )            
            posCounts.append(labelAgreement)
            negCounts.append( labelDisagreement)

            #print(str(u) + " has " str(numLabelsU) + ", " + str(v)  and " + str(numLabelsV) + " got to " + str(labelAgreement) + " and " + str(labelDisagreement))    
            totalPos = totalPos + posCounts[-1] 
            totalNeg = totalNeg + negCounts[-1] 
                        
            allLabels = CombineLabels(labelCount[su], labelCount[sv])
            #numAll = GetNumberLabels(allLabels)
            mySets.union(u, v)
            labelCount[ mySets[u] ] = allLabels.copy()

        upto = upto + 1

    localPos = totalPos
    localNeg = 0.0

    # start off with every point in own cluster
    randError = (localPos + localNeg) / (totalPos + totalNeg)
    
    for i in range(len(posCounts)):
        if T > mstEdges[i]:
            return(randError)
        localPos = localPos - posCounts[i]
        localNeg = localNeg + negCounts[i]

        randError = (localPos + localNeg) / (totalPos + totalNeg)
		        
    return(randError)

def mstEdges(WG):
    
    mySets = nx.utils.UnionFind()       
    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    
    sortedEdges = sorted(edgeWeights, reverse=True, key=lambda edge: edge[2]) 
    mstEdges = list()    
    for u, v, w in sortedEdges:
        su = mySets[u]
        sv = mySets[v]
        if su != sv:
            mstEdges.append((u,v))            
            mySets.union(u, v)
    return(mstEdges)

def GetWatershedGraph(G):
    WG = G.copy()    
    #this function returns the graph WG with new weights of max(min(neighbors))    
    for (u,v,d) in WG.edges(data = True):
        #print(u)        
        uev = [ues for ues in WG[u] if ues != v]
        veu = [ves for ves in WG[v] if ves != u]

        uew = [WG[u][ues]['weight'] for ues in uev]
        maxUW = max(uew)
        maxUI = uew.index(maxUW)
        maxUV = uev[maxUI]  # should be vertix v of edge uv that had max

        vew = [WG[v][ves]['weight'] for ves in veu]
        maxVW = max(vew)
        maxVI = vew.index(maxVW)
        maxVU = veu[maxVI]  # should be vertex u of edge vu that had max

        # now do min
        if maxUW < maxVW:
            minMaxW = maxUW
            minMaxU = u
            minMaxV = maxUV
        else:
            minMaxW = maxVW
            minMaxU = v
            minMaxV = maxVU

        d['weight'] = d['weight'] - minMaxW
        d['minmax'] = (minMaxU, minMaxV)
    return WG

def GetWSGraph(G):
    WG = G.copy()    

    mySets = nx.utils.UnionFind()       
    labelCount = dict()
    for n in WG:        
        labelCount[n] = dict()
        labelCount[n][n] = 1.0


    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    
    
    sortedEdges = sorted(edgeWeights, reverse=True, key=lambda edge: edge[2]) 
    #print("MST edge affinities: " + str(sortedEdges[0][2]) + " -> " + str(sortedEdges[-1][2]) )
   
    upto = 0
    for u, v, w in sortedEdges:
        su = mySets[u]
        sv = mySets[v]
        if su != sv:            
            numLabelsU = GetNumberLabels( labelCount[su] )
            numLabelsV = GetNumberLabels( labelCount[sv] )
            if (numLabelsU < 2) or (numLabelsV < 2):
                labelAgreement = DotProductLabels( labelCount[su], labelCount[sv] )
                labelDisagreement = numLabelsU * numLabelsV - labelAgreement
                            
                allLabels = CombineLabels(labelCount[su], labelCount[sv])
                #numAll = GetNumberLabels(allLabels)
                mySets.union(u, v)
                labelCount[ mySets[u] ] = allLabels.copy()
                WG[u][v]['weight'] = 1.0
            else:
                WG[u][v]['weight'] = -1.0


    return(WG)


def FindWSRandCounts(WG, nlabels_dict):
    
    mySets = nx.utils.UnionFind()       
    labelCount = dict()
    for n in WG:        
        labelCount[n] = dict()
        labelCount[n][ nlabels_dict[n] ] = 1.0


    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    
    
    sortedEdges = sorted(edgeWeights, reverse=True, key=lambda edge: edge[2]) 
    #print("MST edge affinities: " + str(sortedEdges[0][2]) + " -> " + str(sortedEdges[-1][2]) )
    mstEdges = list()    
    posCounts = list()
    negCounts = list()
    totalPos = 0.0
    totalNeg = 0.0    
    
    upto = 0

    for u, v, w in sortedEdges:
        su = mySets[u]
        sv = mySets[v]
        if su != sv:            
            numLabelsU = GetNumberLabels( labelCount[su] )
            numLabelsV = GetNumberLabels( labelCount[sv] )
            if (numLabelsU < 2) or (numLabelsV < 2):
                labelAgreement = DotProductLabels( labelCount[su], labelCount[sv] )
                labelDisagreement = numLabelsU * numLabelsV - labelAgreement
                mstEdges.append( (u,v) )            
                posCounts.append(labelAgreement)
                negCounts.append( labelDisagreement)

                #print(str(u) + " has " str(numLabelsU) + ", " + str(v)  and " + str(numLabelsV) + " got to " + str(labelAgreement) + " and " + str(labelDisagreement))    
                totalPos = totalPos + posCounts[-1] 
                totalNeg = totalNeg + negCounts[-1] 
                            
                allLabels = CombineLabels(labelCount[su], labelCount[sv])
                #numAll = GetNumberLabels(allLabels)
                mySets.union(u, v)
                labelCount[ mySets[u] ] = allLabels.copy()

        upto = upto + 1
    #print(mstEdges)    
    #print("Total Pos: " + str(totalPos) + " Neg: " + str(totalNeg)) 
    return(posCounts, negCounts, mstEdges, totalPos, totalNeg)
    