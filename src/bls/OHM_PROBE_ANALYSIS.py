import numpy as np
import networkx as nx


def count_monotonic_pairs(lst):
    increasing_pairs = 0
    decreasing_pairs = 0
    equal_pairs = 0

    for x, y in zip(lst, lst[1:]):
        if x < y:
            increasing_pairs += 1
        elif x > y:
            decreasing_pairs += 1
        elif x == y:
            equal_pairs += 1
    
    total_pairs = len(lst) - 1
    if total_pairs == 0:
        return 0, 0  # Avoid division by zero for lists with fewer than 2 elements
    
    normalized_increasing = increasing_pairs / total_pairs
    normalized_decreasing = decreasing_pairs / total_pairs
    normalized_equal = equal_pairs / total_pairs

    return normalized_increasing, normalized_decreasing, normalized_equal

def WeightAnalysis(ohm):
        
    effectiveInputs = np.zeros((ohm.numLayers, ohm.numStack))
    
    for li in range(len(ohm.paramStackMem)):
        for i in range(len(ohm.paramStackMem[li])):
            weights = ohm.paramStackMem[li][i].GetLSBIntsHack()
            thresh = ohm.paramThreshMem[li][i].GetLSBIntsHack()
            thresh = thresh[0]
            weights = np.array(weights)

            #ConnectedAnalysis(weights)
            funw = np.sum(weights >= thresh)
            
            sumw = np.sum(weights)
            numw = weights.shape[0]
            if sumw == numw:
                numInputs = numw
            else:
                numInputs = 0
            
            effectiveInputs[li][i] = funw

            

    
    effectiveInputs = effectiveInputs.T
    
    return effectiveInputs


def FullyConnectedGraph(W):

    g = nx.Graph()
    for i in range(W.shape[0]):
        g.add_node(i, weight = W[i])
    
    for u in range(W.shape[0]):
        for v in range(W.shape[0]):
            if u != v:
                w = abs(W[u] - W[v])
                g.add_edge(u, v, weight = w)

    return(g)    

def ConnectedAnalysis(W):

    verbose = True

    G = FullyConnectedGraph(W)

    edgeWeights = [(u,v,w) for (u,v,w) in G.edges(data = 'weight')]    
    # reverse = False : small -> big
    sortedEdges = sorted(edgeWeights, reverse=True, key=lambda edge: edge[2]) 
    
    if verbose:
        print("W differences: " + str(sortedEdges[0][2]) + " -> " + str(sortedEdges[-1][2]) )

    WG = G.copy()    
    CG = nx.Graph()
    
    for n in WG:                        
        WG.nodes[n]['label'] = 0

    labelUpto = 1
    for u, v, w in sortedEdges:

        # new basin
        if (WG.nodes[u]['label'] == 0) and (WG.nodes[v]['label'] == 0): 
            WG.nodes[u]['label'] = labelUpto
            WG.nodes[v]['label'] = labelUpto
            CG.add_node(labelUpto, weight = w)
            labelUpto = labelUpto + 1
        elif (WG.nodes[u]['label'] == 0):
            WG.nodes[u]['label'] = WG.nodes[v]['label']
        elif (WG.nodes[v]['label'] == 0):
            WG.nodes[v]['label'] = WG.nodes[u]['label']
        else:   
            nu = WG.nodes[u]['label']
            nv = WG.nodes[v]['label']

            if (nu != nv):
                if (CG.has_edge(nu, nv) == False):            
                    # Standard smallest depth is w - min(b1, b2)
                    # We want to merge smallest depth so we take the negative to make it big as good
                    depth = w - max(CG.nodes[nu]['weight'], CG.nodes[nv]['weight'])
                    CG.add_edge(nu, nv, weight = depth)
    numBasins = labelUpto-1
    
    if verbose:
        print("Watershed has " + str(numBasins) + " basins")
    
    if (numBasins > 1):
        ccWeights = [(u,v,w) for (u,v,w) in CG.edges(data = 'weight')]    
        # reverse = False : small -> big
        ccSorted = sorted(ccWeights, reverse=True, key=lambda edge: edge[2]) 
        if verbose:
            print("CC has " + str(len(ccWeights)) + " affinities: " + str(ccSorted[0][2]) + " -> " + str(ccSorted[-1][2]) )

        # apply predefined threshold
        thresholdi = int(len(ccSorted) * THRESH_OFFSET)
        threshold = ccSorted[thresholdi][2]    
        ccThresh = [ [d[0], d[1], d[2] - threshold] for d in ccSorted]
        #print("CCThresh is " + str(ccThresh[0]) + " -> " + str(ccThresh[-1]) )

        # Now run correlation clustering to find threshold
        if verbose:
            print("Correlation Clustering at threshold " + str(threshold))
        threshSets = nx.utils.UnionFind()   
        nextNode = dict()
        for n in CG:
            nextNode[n] = threshSets[n]
        
        totalPos = sum([d[2] for d in ccThresh if d[2] > 0])
        totalNeg = sum([d[2] for d in ccThresh if d[2] < 0])
        accTotal = [0]*len(ccThresh)
        if verbose:
            print("Correlation Clustering totals +ve: " + str(totalPos) + ", -ve: " + str(totalNeg))
    
        accTotal[0] = totalPos + totalNeg
        #print("Energy 0: " + str(accTotal[0]) + " from Pos: " + str(totalPos) + ", Neg: " + str(totalNeg))
        DELTA_TOLERANCE = 1.0e-6
        ei = 1      # edge index
        lowE = accTotal[0]
        lowT = ccThresh[0][2] + 1.0e3
        prevT = lowT
        for u, v, w in ccThresh:
            # Only need to go to zero weight
            #if w >= 0.0:
            #    break
            if threshSets[u] != threshSets[v]:
                accWeight = 0.0
                # traverse nodes in u and look at edges
                # if fully connected we should probably traverse nodes u and v instead
                done = False
                cu = u
                while not done:
                    for uev in CG[cu]:                
                        if threshSets[uev] == threshSets[v]:
                            threshWeight = CG[cu][uev]['weight'] - threshold
                            accWeight = accWeight + threshWeight                            
                    cu = nextNode[cu]
                    if cu == u:
                        done = True

                # Merge sets
                threshSets.union(u, v)
                # Swap next pointers... this incrementally builds a pointer cycle around all the nodes in the component
                tempNext = nextNode[u]
                nextNode[u] = nextNode[v]
                nextNode[v] = tempNext

                accTotal[ei] = accTotal[ei-1] - accWeight            
                #print("Energy at threshold " + str(w) + ": " + str(accTotal[ei]))
                if accTotal[ei] < lowE:
                    lowE = accTotal[ei]
                    lowT = (w + prevT) / 2.0

                prevT = w
                ei = ei + 1
        
        if verbose:        
            print("Smallest Energy: " + str(lowE) + " at threshold " + str(lowT))     
        
        # threshold graph and run connected components 
        finalThreshold = threshold + lowT
        if verbose:
            print("Final Threshold is: " + str(finalThreshold))

        LG = CG.copy()    
        LG.remove_edges_from([(u,v) for (u,v,d) in  LG.edges(data=True) if (d['weight'] - finalThreshold) < 0.0])
        #LG.remove_edges_from([(u,v) for (u,v,d) in  ccThresh if d < lowT])
        L = {node:color for color,comp in enumerate(nx.connected_components(LG)) for node in comp}
        
        seenLabel = dict()
        count = 0
        for n in L:        
            CG.nodes[n]['label'] = L[n]
            if L[n] not in seenLabel:
                count = count + 1
                seenLabel[L[n]] = 1
        if verbose:
            print("Final Segmentation has " + str(count) + " labels")

    else:
        if verbose:        
            print("One basin at Energy: " + str(lowE) + " at threshold " + str(lowT) + " and label 1")     

        threshold = 0.0
        lowE = 0.0
        lowT = 0.0
                        
        for n in CG:        
            CG.nodes[n]['label'] = 1

    return(WG, CG)

