import sys
import glob
import torch
import torch.nn.functional as F
from unet import UNet
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import pdb
import skimage.io as skio
from scipy.signal import medfilt as med_filt
import math
import random
import skimage.transform
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from scipy.ndimage.measurements import label


verbose = 0
THRESH_OFFSET = 0.75
############################################################################################################################
############################################################################################################################
############################################################################################################################
## DSN Inference Methods
############################################################################################################################
############################################################################################################################
def GetSegImages(WG, CG, W, H):
    imgWS = np.zeros((W, H), np.single)
    imgCC = np.zeros((W, H), np.single)
    
    for u, v, d in WG.edges(data = True):
        ul = WG.nodes[u]['label']
        imgWS[u[0], u[1]] = ul
        imgCC[u[0], u[1]] = CG.nodes[ul]['label']

        vl = WG.nodes[v]['label']
        imgWS[v[0], v[1]] = vl
        imgCC[v[0], v[1]] = CG.nodes[vl]['label']

    return(imgWS, imgCC)

def ApplyDSN(uOut):    
    netOut = torch.squeeze(uOut).cpu()
    netOut = netOut.unsqueeze(0)    
    npGXY = netOut.detach().numpy()

    W = npGXY.shape[1]
    H = npGXY.shape[2]
    if verbose:
        print("ApplyDSN to image  " + str(W) + ", " + str(H))

    # Setup input graph 
    G = nx.grid_2d_graph(W, H)
    
    for u, v, d in G.edges(data = True):
        d['weight'] =  (npGXY[0, u[0], u[1]] + npGXY[0, v[0], v[1]])/2.0

    [WG, CG] = ApplyDSNGraph(G)
    wsImg, ccImg = GetSegImages(WG, CG, W, H)
    return (wsImg, ccImg)

def ApplyDSNGraph(G):
    WG = G.copy()    
    CG = nx.Graph()
    
    for n in WG:                        
        WG.nodes[n]['label'] = 0

    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    
    # reverse = False : small -> big
    sortedEdges = sorted(edgeWeights, reverse=True, key=lambda edge: edge[2]) 
    if verbose:
        print("WS affinities: " + str(sortedEdges[0][2]) + " -> " + str(sortedEdges[-1][2]) )

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


############################################################################################################################
############################################################################################################################
############################################################################################################################
## Kruskal DSN Eval with disjoint sets datastructure
############################################################################################################################
############################################################################################################################

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

def EvalDSN(G, nlabels_dict, W, H, lossType):
    WG = G.copy()    
    CG = nx.Graph()
    wsSets = nx.utils.UnionFind()       

    labelCount = dict()            
    wsfirstNode = dict()
    wsEdge = dict()
    wsPos = dict()
    wsNeg = dict()
    
    if verbose:
        print("-----------------------------"); 
    
    ################################################################################################
    ## Watershed-Cuts in first layer        
    for n in WG:        
        WG.nodes[n]['label'] = 0
        labelCount[n] = dict()
        labelCount[n][ nlabels_dict[n] ] = 1.0        
        #print("Label for " + str(n) + " is " + str(nlabels_dict[n]))

    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    
    # reverse = True : +ve -> -ve
    sortedEdges = sorted(edgeWeights, reverse=True, key=lambda edge: edge[2]) 
    if verbose:
        print("WS affinities: " + str(sortedEdges[0][2]) + " -> " + str(sortedEdges[-1][2]) )
    
    labelUpto = 1
    for u, v, w in sortedEdges:
        lu = -1
        if (WG.nodes[u]['label'] == 0) and (WG.nodes[v]['label'] == 0):     # new basin
            WG.nodes[u]['label'] = labelUpto
            WG.nodes[v]['label'] = labelUpto
            wsEdge[labelUpto] = list()
            wsPos[labelUpto] = list()
            wsNeg[labelUpto] = list()
            wsfirstNode[labelUpto] = u           # Save one WS node to access the labelCounts from CG            
            CG.add_node(labelUpto, weight = w)  # One node in second graph for each WS basin
            lu = labelUpto
            labelUpto = labelUpto + 1
        elif (WG.nodes[u]['label'] == 0):                       # extend basin            
            WG.nodes[u]['label'] = WG.nodes[v]['label']
            lu = WG.nodes[v]['label']
        elif (WG.nodes[v]['label'] == 0):                       # extend basin
            WG.nodes[v]['label'] = WG.nodes[u]['label']                    
            lu = WG.nodes[u]['label']
        else:   
            nu = WG.nodes[u]['label']
            nv = WG.nodes[v]['label']
            if (nu != nv):
                if (CG.has_edge(nu, nv) == False):       
                    # Standard smallest depth is w - min(b1, b2)
                    # We want to merge smallest depth so we take the negative to make it big as
                    depth = w - max(CG.nodes[nu]['weight'], CG.nodes[nv]['weight'])
                    CG.add_edge(nu, nv, weight = depth)
                    CG.edges[nu, nv]['edge'] = [u, v]

        su = wsSets[u]
        sv = wsSets[v]
        if su != sv:
            if lu > 0:        
                labelAgreement = DotProductLabels( labelCount[su], labelCount[sv] )
                numLabelsU = GetNumberLabels( labelCount[su] )
                numLabelsV = GetNumberLabels( labelCount[sv] )
                labelDisagreement = numLabelsU * numLabelsV - labelAgreement
                #print("+ve: " + str(labelAgreement) + " -ve: " + str(labelDisagreement) + " for " + str(u) + ", " + str(v))
                allLabels = CombineLabels(labelCount[su], labelCount[sv])
                wsSets.union(u, v)                
                labelCount[ wsSets[u] ] = allLabels.copy()
                
                wsEdge[lu].append([u,v])
                wsPos[lu].append(labelAgreement)
                wsNeg[lu].append(labelDisagreement)                

    numBasins = labelUpto-1
    if verbose:
        print("Watershed has " + str(numBasins) + " basins")

    ##########################################
    ## Initialize basin stats and second layer
    ccSets = nx.utils.UnionFind()       
    cclabelCount = dict()
    basinPos = dict()
    basinNeg = dict()
    totalPosWS = 0.0
    totalNegWS = 0.0
    for n in CG:
        # Setup the sets for CC
        wsIndex = wsSets[ wsfirstNode[n] ]                 
        cclabelCount[n] = labelCount[ wsIndex  ].copy()
        # Accumulate counts for each basin
        #print(wsPos[n])
        basinPos[n] = sum([d for d in wsPos[n]])
        basinNeg[n] = sum([d for d in wsNeg[n]])
        #print("Basin  " + str(n) + " Pos: " + str(basinPos[n]) + "   and Neg: " + str(basinNeg[n]))
        totalPosWS = totalPosWS + basinPos[n]
        totalNegWS = totalNegWS + basinNeg[n]

    ccEdge = list()    
    ccBasin = list()    
    ccPos = list()
    ccNeg = list()
    totalPos = totalPosWS
    totalNeg = totalNegWS
    threshPos = totalPosWS
    threshNeg = totalNegWS
    if verbose:
        print("Watershed Total Pos " + str(totalPos) + " and Total Neg " + str(totalNeg)) 
    
    if numBasins > 1:
        ################################################################################################
        ## Correlation clustering on Connected Components to find threshold
        ccWeights = [(u,v,w) for (u,v,w) in CG.edges(data = 'weight')]    
        # reverse = True : +ve -> -ve
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
        
        totalPosCC = sum([d[2] for d in ccThresh if d[2] > 0])
        totalNegCC = sum([d[2] for d in ccThresh if d[2] < 0])
        accTotal = [0]*(len(ccThresh)+1)
        if verbose:
            print("Correlation Clustering totals +ve: " + str(totalPosCC) + ", -ve: " + str(totalNegCC))

        accTotal[0] = totalPosCC + totalNegCC
        
        DELTA_TOLERANCE = 1.0e-6
        ei = 1      # edge index
        lowE = accTotal[0]
        lowT = ccThresh[0][2] + 1.0e3
        prevT = lowT
        for u, v, w in ccThresh:
            # Only need to go to zero weight
            #if w <= 0.0:
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
                            threshWeight = CG.edges[cu, uev]['weight'] - threshold
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
                    lowT = (prevT + w)/2.0
                prevT = w
                ei = ei + 1
        if verbose:        
            print("Lowest Energy: " + str(lowE) + " at threshold " + str(lowT))     

        # threshold graph and run connected components 
        finalThreshold = threshold + lowT
        if verbose:
            print("Final Threshold is: " + str(finalThreshold))

        ################################################################################################
        ## Final Connected Components at Correlation Clustering Threshold                    
        for u, v, w in ccThresh:
            finalW = w - lowT
            su = ccSets[u]
            sv = ccSets[v]            
            if su != sv:                                
                labelAgreement = DotProductLabels( cclabelCount[su], cclabelCount[sv] )
                numLabelsU = GetNumberLabels( cclabelCount[su] )
                numLabelsV = GetNumberLabels( cclabelCount[sv] )
                labelDisagreement = numLabelsU * numLabelsV - labelAgreement                
                allLabels = CombineLabels(cclabelCount[su], cclabelCount[sv])
                ccSets.union(u, v)                
                cclabelCount[ ccSets[u] ] = allLabels.copy()
                # Basin specific counts
                ccBasin.append([u,v])
                ccEdge.append(CG[u][v]['edge'])            
                ccPos.append(labelAgreement)
                ccNeg.append(labelDisagreement)
                totalPos = totalPos + labelAgreement
                totalNeg = totalNeg + labelDisagreement
                if finalW >= 0.0:
                    threshPos = threshPos + labelAgreement
                    threshNeg = threshNeg + labelDisagreement        
    else:
        threshold = 0.0
        lowT = 0.0     
        finalThreshold = 0.0  
    
    posError = totalPos - threshPos
    negError = threshNeg
    randError = (posError + negError) / (totalPos + totalNeg)
    if verbose:        
        print("Rand Error: " + str(randError))
        print("From #pos: " + str(totalPos) + " #neg: " + str(totalNeg))
        print("   and FN: " + str(posError) + "   FP: " + str(negError))


    ######################################################
    ## Now Assign Errors back to image (neural net output)    
    if verbose:
        print("Assigning Errors to ws edges ")

    labels = np.zeros((1, W, H), np.single)
    weights = np.zeros((1, W, H), np.single)        
    
    allPos = 0.0
    allNeg = 0.0

    if lossType == 'purity':
        randWeight = randError        
        for n in wsEdge:
            for i in range(len(wsEdge[n])):
                [u, v] = wsEdge[n][i]                
                if wsPos[n][i] >= wsNeg[n][i]:
                    label = 1
                    weight = (wsPos[n][i] - wsNeg[n][i])/ (wsPos[n][i] + wsNeg[n][i]) 
                else:
                    label = -1
                    weight = (wsNeg[n][i] - wsPos[n][i])/ (wsPos[n][i] + wsNeg[n][i])                        
                labels[0, u[0], u[1]] = label
                weights[0, u[0], u[1]] = weight * randWeight
                labels[0, v[0], v[1]] = label
                weights[0, v[0], v[1]] = weight * randWeight
        
        for n in range(len(ccEdge)):
            [u, v] = ccEdge[n]            
            if ccPos[n] >= ccNeg[n]:
                label = 1
                weight = (ccPos[n] - ccNeg[n])/ (ccPos[n] + ccNeg[n])            
            else:
                label = -1
                weight = (ccNeg[n] - ccPos[n])/ (ccPos[n] + ccNeg[n])
            labels[0, u[0], u[1]] = label
            weights[0, u[0], u[1]] = weight * randWeight
            labels[0, v[0], v[1]] = label
            weights[0, v[0], v[1]] = weight * randWeight
    elif lossType == 'equal':
        randWeight = randError        
        for n in wsEdge:
            for i in range(len(wsEdge[n])):
                [u, v] = wsEdge[n][i]                
                if wsPos[n][i] >= wsNeg[n][i]:
                    label = 1
                    weight = 1
                else:
                    label = -1
                    weight = 1
                labels[0, u[0], u[1]] = label
                weights[0, u[0], u[1]] = weight * randWeight
                labels[0, v[0], v[1]] = label
                weights[0, v[0], v[1]] = weight * randWeight
        
        for n in range(len(ccEdge)):
            [u, v] = ccEdge[n]            
            if ccPos[n] >= ccNeg[n]:
                label = 1
                weight = 1
            else:
                label = -1
                weight = 1
            labels[0, u[0], u[1]] = label
            weights[0, u[0], u[1]] = weight * randWeight
            labels[0, v[0], v[1]] = label
            weights[0, v[0], v[1]] = weight * randWeight
    else:
        print("Unknown loss type " + lossType)
        exit()
        randWeight = randError
        for n in wsEdge:
            for i in range(len(wsEdge[n])):
                [u, v] = wsEdge[n][i]
                ypred = G.edges[u, v]['weight']

                if wsNeg[n][i] < 0.1:
                    label = 1.0
                    weight = wsPos[n][i]
                elif wsPos[n][i] < 0.1:
                    label = -1.0
                    weight = wsNeg[n][i]
                else:
                    if ypred >= 0.0: 
                        label = -1.0
                        weight = wsNeg[n][i]
                    else:
                        label = 1.0
                        weight = wsPos[n][i]

                labels[0, u[0], u[1]] = label
                weights[0, u[0], u[1]] = weight * randWeight
                labels[0, v[0], v[1]] = label
                weights[0, v[0], v[1]] = weight * randWeight

        for n in range(len(ccEdge)):
            [u, v] = ccEdge[n]
            ypred = G.edges[u, v]['weight']

            if ccNeg[n] < 0.1:
                label = 1.0
                weight = ccPos[n]
            elif ccPos[n] < 0.1:
                label = -1.0
                weight = ccNeg[n]
            else:
                if ypred >= 0.0: 
                    label = -1.0
                    weight = ccNeg[n]
                else:
                    label = 1.0
                    weight = ccPos[n]

            labels[0, u[0], u[1]] = label
            weights[0, u[0], u[1]] = weight * randWeight
            labels[0, v[0], v[1]] = label
            weights[0, v[0], v[1]] = weight * randWeight            

    #print('Totals Pos: ' + str(allPos) + ' Neg: ' + str(allNeg) + '\n')
    return [finalThreshold, labels, weights, randError]

def RandLossDSN(uOut, nodeLabels, lossType, trn_idx):
    W = nodeLabels.shape[0]
    H = nodeLabels.shape[1]

    netOut = torch.squeeze(uOut).cpu()
    netOut = netOut.unsqueeze(0)
    
    npGXY = netOut.detach().numpy()    
    numpLabels = nodeLabels.detach().numpy()
    if verbose:
        print("EvalDSN with image  " + str(W) + ", " + str(H))

    # Setup input graph
    G = nx.grid_2d_graph(W, H)
    nlabels_dict = dict()
    

    for u, v, d in G.edges(data = True):
        d['weight'] =  (npGXY[0, u[0], u[1]] + npGXY[0, v[0], v[1]])/2.0
        nlabels_dict[u] = numpLabels[u[0], u[1]]
        nlabels_dict[v] = numpLabels[v[0], v[1]]
            
    # Run the DSN
    [threshold, labels, weights, randError] = EvalDSN(G, nlabels_dict, W, H, lossType)

    # Apply final threshold 
    #finalThreshold = torch.tensor(threshold)
    #finalOut = netOut - finalThreshold

    tlabels = torch.tensor(labels)
    tweights = torch.tensor(weights)    

    #temp = torch.sum(tweights)
    #print("Total weight is: ") 
    #print(temp)
    # Squared loss    
    errors = (netOut - tlabels) ** 2
    werrors = torch.mul(errors, tweights)
    randLoss = torch.sum(werrors)

    return(randLoss, randError)    

