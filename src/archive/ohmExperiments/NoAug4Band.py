import progressbar
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

# MATPLOTLIB defaults
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15



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

def ScaleAndShow(img, fignum):
    minv = np.min(img)
    maxv = np.max(img)
    print("Fig %i: Range %f -> %f" % (fignum, minv, maxv))
    plt.figure(fignum)
    simg = img - minv 
    if abs(maxv - minv) > 1e-4:
        simg = simg / (maxv - minv)
    
    plt.imshow(simg, cmap='gray')

############################################################################################################################
############################################################################################################################
############################################################################################################################
## DSN Inference Methods
############################################################################################################################
############################################################################################################################

def ApplyDSN(uOut):    
    netOut = torch.squeeze(uOut).cpu()
    if len(netOut.shape) == 2:
        netOut = netOut.unsqueeze(0)
        numInputs = 1
    else:
        numInputs = 2

    npGXY = netOut.detach().numpy()
    W = npGXY.shape[1]
    H = npGXY.shape[2]
    if verbose:
        print("ApplyDSN to image  " + str(W) + ", " + str(H) + " with " + str(numInputs) + " inputs")

    # Setup input graph 
    G = nx.grid_2d_graph(W, H)
    
    if numInputs == 1:
        for u, v, d in G.edges(data = True):
            d['weight'] =  (npGXY[0, u[0], u[1]] + npGXY[0, v[0], v[1]])/2.0
            nlabels_dict[u] = nodeLabels[u[0], u[1]]
            nlabels_dict[v] = nodeLabels[v[0], v[1]]
    else:
        for u, v, d in G.edges(data = True):
            if u[0] == v[0]:    #  vertical edges
                d['weight'] =  npGXY[1, u[0], u[1]]
            else:               # horizontal edges
                d['weight'] =  npGXY[0, u[0], u[1]]
            nlabels_dict[u] = nodeLabels[u[0], u[1]]
            nlabels_dict[v] = nodeLabels[v[0], v[1]]

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

def EvalDSN(G, nlabels_dict, W, H, numInputs):
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
        print("Assigning Errors")

    labels = np.zeros((numInputs, W, H), np.single)
    weights = np.zeros((numInputs, W, H), np.single)        

    #randWeight = randError
    randWeight = 1.0

    allPos = 0.0
    allNeg = 0.0

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


            #if wsPos[n][i] >= wsNeg[n][i]:
            #    label = 1
            #    weight = (wsPos[n][i] - wsNeg[n][i])/ (wsPos[n][i] + wsNeg[n][i]) 
            #else:
            #    label = -1
            #    weight = (wsNeg[n][i] - wsPos[n][i])/ (wsPos[n][i] + wsNeg[n][i])        
            
            if numInputs == 1:
                labels[0, u[0], u[1]] = label
                weights[0, u[0], u[1]] = weight * randWeight
                labels[0, v[0], v[1]] = label
                weights[0, v[0], v[1]] = weight * randWeight
            else:
                if u[0] == v[0]:    #  vertical edges
                    labels[1, u[0], u[1]] = label
                    weights[1, u[0], u[1]] = weight * randWeight
                else:               # horizontal edges
                    labels[0, u[0], u[1]] = label
                    weights[0, u[0], u[1]] = weight * randWeight

    #randWeight = randError
    randWeight = 1.0
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

        #if ccPos[n] >= ccNeg[n]:
        #    label = 1
        #    weight = (ccPos[n] - ccNeg[n])/ (ccPos[n] + ccNeg[n])            
        #else:
        #    label = -1
        #    weight = (ccNeg[n] - ccPos[n])/ (ccPos[n] + ccNeg[n])

        if numInputs == 1:                
            labels[0, u[0], u[1]] = label
            weights[0, u[0], u[1]] = weight * randWeight
            labels[0, v[0], v[1]] = label
            weights[0, v[0], v[1]] = weight * randWeight
        else:        
            if u[0] == v[0]:    #  vertical edges
                labels[1, u[0], u[1]] = label
                weights[1, u[0], u[1]] = weight * randWeight
            else:               # horizontal edges
                labels[0, u[0], u[1]] = label
                weights[0, u[0], u[1]] = weight * randWeight

    #print('Totals Pos: ' + str(allPos) + ' Neg: ' + str(allNeg) + '\n')
    return [finalThreshold, labels, weights, randError]

def RandLossDSN(uOut, nodeLabels, trn_idx):
    W = nodeLabels.shape[0]
    H = nodeLabels.shape[1]

    netOut = torch.squeeze(uOut).cpu()
    if len(netOut.shape) == 2:
        netOut = netOut.unsqueeze(0)
        numInputs = 1
    else:
        numInputs = 2
    npGXY = netOut.detach().numpy()    
    
    if verbose:
        print("EvalDSN with image  " + str(W) + ", " + str(H) + " with " + str(numInputs) + " inputs")

    # Setup input graph
    G = nx.grid_2d_graph(W, H)
    nlabels_dict = dict()
    
    if numInputs == 1:
        for u, v, d in G.edges(data = True):
            d['weight'] =  (npGXY[0, u[0], u[1]] + npGXY[0, v[0], v[1]])/2.0
            nlabels_dict[u] = nodeLabels[u[0], u[1]]
            nlabels_dict[v] = nodeLabels[v[0], v[1]]
    else:
        for u, v, d in G.edges(data = True):
            if u[0] == v[0]:    #  vertical edges
                d['weight'] =  npGXY[1, u[0], u[1]]
            else:               # horizontal edges
                d['weight'] =  npGXY[0, u[0], u[1]]
            nlabels_dict[u] = nodeLabels[u[0], u[1]]
            nlabels_dict[v] = nodeLabels[v[0], v[1]]
            
    # Run the DSN
    [threshold, labels, weights, randError] = EvalDSN(G, nlabels_dict, W, H, numInputs)

    # Apply final threshold 
    finalThreshold = torch.tensor(threshold)
    #finalOut = netOut - finalThreshold

    tlabels = torch.tensor(labels)
    tweights = torch.tensor(weights)    

    #ones = torch.ones((1, 128, 128))
    #print(ones.shape)
    #print(netOut.shape)
    #finalOut = torch.div(ones, netOut)
    #hinge_loss = torch.mul(finalOut, tlabels)
    #werror = torch.mul(hinge_loss, tweights)
    #minv = torch.min(netOut)
    #maxv = torch.max(netOut)
    #print("NetOut Range %f -> %f" % (minv, maxv))    
    #randLoss = torch.sum(werror)          
    # if trn_idx > 3:
    #     hinge_loss = 1.0 - torch.mul(netOut, tlabels)
    #     hinge_loss[hinge_loss < 0.0] = 0.0
    #     werror = torch.mul(hinge_loss, tweights)
    #     randLoss = torch.sum(werror)      
    # else:
    # Squared loss    
    errors = (netOut - tlabels) ** 2
    werrors = torch.mul(errors, tweights)
    randLoss = torch.sum(werrors)

    # Hinge loss
    #hinge_loss = 1.0 - torch.mul(netOut, tlabels)
    #hinge_loss[hinge_loss < 0.0] = 0.0
    #werror = torch.mul(hinge_loss, tweights)
    #randLoss = torch.sum(werror)      

    #print("\t\t" + str(trn_idx) + " Loss " + str(randLoss.item()) + "  and Rand " + str(randError))
    return(randLoss, randError)    


##############################################################
## Basic Training Program
if __name__ == '__main__':
    verbose = 0
    theSeed = 0

    numEpochs = 10000
    THRESH_OFFSET = 0.75
    unetFeatures = 7
    unetDepth = 5

    learningRate = 1
    rhoMemory = 0.99
    rateStep = 100
    learningRateGamma = 0.7
    
    numTrain = -1
    numValid = -1  

    random.seed(theSeed)
    torch.manual_seed(theSeed)

    extractDir = "d:\\image_data\\snemi3d-extracts\\"

    trainName0 = extractDir + "train0." + str(theSeed) + ".pkl"
    trainName1 = extractDir + "train1." + str(theSeed) + ".pkl"
    trainName2 = extractDir + "train2." + str(theSeed) + ".pkl"
    trainName4 = extractDir + "train4." + str(theSeed) + ".pkl"
    trainNameLabel = extractDir + "trainLabel." + str(theSeed) + ".pkl"

    validName0 = extractDir + "valid0." + str(theSeed) + ".pkl"
    validName1 = extractDir + "valid1." + str(theSeed) + ".pkl"
    validName2 = extractDir + "valid2." + str(theSeed) + ".pkl"
    validName4 = extractDir + "valid4." + str(theSeed) + ".pkl"
    validNameLabel = extractDir + "validLabel." + str(theSeed) + ".pkl"

    

    with open(trainName4, 'rb') as f:
        X = pickle.load(f)
    with open(trainNameLabel, 'rb') as f:
        Y = pickle.load(f)
    with open(validName4, 'rb') as f:
        XV = pickle.load(f)
    with open(validNameLabel, 'rb') as f:
        YV = pickle.load(f)

    X = X.astype(np.single)
    XV = XV.astype(np.single)
    Y = Y.astype(np.single)
    YV = YV.astype(np.single)

    #Ysyn = Ysyn.astype(np.single)

    XT = torch.tensor(X, requires_grad=False)                
    YT = torch.tensor(Y, requires_grad=False)                                                
    XVT = torch.tensor(XV, requires_grad=False)                
    YVT = torch.tensor(YV, requires_grad=False)                                                

    if numTrain < 0:
        numTrain = XT.shape[0]
    if numValid < 0:
        numValid = XVT.shape[0]                

    if XT.shape[0] < numTrain:
        numTrain = XT.shape[0]
    if XVT.shape[0] < numValid:
        numValid = XVT.shape[0]

    # Setting up U-net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet(in_channels=4, n_classes=1, depth=unetDepth, wf=unetFeatures, padding=True, batch_norm=True, up_mode='upconv').to(device)

    optimizer = optim.Adadelta(model.parameters(), rho=rhoMemory, lr=learningRate)
    
    # Initializing training and validation lists to be empty
    trn_loss_lst   = []
    trn_rerror_lst = []
    val_loss_lst   = []
    val_rerror_lst  = []

    # Initializing best loss and random error
    val_best_loss   = math.inf
    val_best_rerror = math.inf
    trn_best_loss   = math.inf
    trn_best_rerror = math.inf

    for cepoch in range(0,numEpochs):
        print("Epoch :    ",str(cepoch))
        
        ti = np.random.permutation(numTrain)

        model.train()
        with torch.enable_grad():                            

            # Bar
            bar = progressbar.ProgressBar(maxval=numTrain, widgets=[progressbar.Bar('=', '    trn[', ']'), ' ', progressbar.Percentage()])
            bar.start()

            # Training
            loss_lst_epoch   = []
            rerror_lst_epoch = []
            for trn_idx in range(0, numTrain):
                bar.update(trn_idx+1)

                optimizer.zero_grad()
                X1 = XT[ ti[trn_idx] ]
                Y1 = YT[ ti[trn_idx] ]

                #X11 = X1.detach().numpy()    
                #Y11 = Y1.detach().numpy()    
                #ScaleAndShow(X11[0].squeeze(), 1)
                #ScaleAndShow(X11[1].squeeze(), 2)
                #ScaleAndShow(X11[2].squeeze(), 3)
                #ScaleAndShow(X11[3].squeeze(), 4)
                #ScaleAndShow(Y11.squeeze(), 5)
                #plt.show()

                X1 = X1.unsqueeze(0)                
                X1 = X1.to(device)                         
                
                uOut = model(X1)
                loss, randError = RandLossDSN(uOut, Y1, trn_idx)
                rerror_lst_epoch  = rerror_lst_epoch + [randError]
                loss_lst_epoch    = loss_lst_epoch       + [loss.detach().numpy().tolist()]
                #print("\t\tLoss   : " + str(loss.detach().numpy()) + "   Error: " + str(randError))
                # Don't change model on last pass so that train and validation are aligned
                if trn_idx < numTrain-1:
                    loss.backward()
                    optimizer.step()

            # Finish bar
            bar.finish()
            trn_cepoch_loss    = sum(loss_lst_epoch)/len(loss_lst_epoch)
            trn_cepoch_rerror  = sum(rerror_lst_epoch)/len(rerror_lst_epoch)
            trn_rerror_lst     = trn_rerror_lst + [trn_cepoch_rerror]
            trn_loss_lst       = trn_loss_lst   + [trn_cepoch_loss]
    
            if trn_cepoch_loss < trn_best_loss:
                trn_best_loss       = trn_cepoch_loss
                trn_best_loss_epoch = cepoch
                trn_best_loss_model = model

            if trn_cepoch_rerror < trn_best_rerror:
                trn_best_rerror       = trn_cepoch_rerror
                trn_best_rerror_epoch = cepoch
                trn_best_rerror_model = model
                print("\t\tSaving training model with error : ", str(trn_best_rerror))                
                torch.save(trn_best_rerror_model.state_dict(), "trn_error_model_0.pth")

            print("AVG Loss   : " + str(trn_cepoch_loss) + "   Err: " + str(trn_cepoch_rerror))
            print("BEST Loss  : " + str(trn_best_loss) + "   Err: " + str(trn_best_rerror) + " at " + str(trn_best_loss_epoch) + ", " + str(trn_best_rerror_epoch))
            
            # Saving all the losses and rand_errors
            if ((cepoch % 500) == 0): 
                print("Saving train loss and errors")
                vname = 'trn_loss_0_' + str(cepoch) + '.pkl'
                with open(vname, 'wb') as f:
                    pickle.dump(trn_loss_lst, f)
                tname = 'trn_error_0_' + str(cepoch) + '.pkl'                    
                with open(tname, 'wb') as f:
                    pickle.dump(trn_rerror_lst, f)            

        # Validation every epoch        
        loss_lst_epoch   = []
        rerror_lst_epoch = []
        # Bar
        model.eval()
        with torch.no_grad():                

            bar = progressbar.ProgressBar(maxval=numValid, widgets=[progressbar.Bar('-', '    Val[', ']'), ' ', progressbar.Percentage()])
            bar.start()

            for val_idx in range(0, numValid):
                bar.update(val_idx+1)
                # print("\t Validating on ",str(val_idx)," image")

                XV1 = XVT[val_idx].unsqueeze(0)
                XV1 = XV1.to(device)                         
                
                uOut = model(XV1)
                
                loss, randError = RandLossDSN(uOut, YVT[val_idx], cepoch)
                rerror_lst_epoch  = rerror_lst_epoch + [randError]
                loss_lst_epoch    = loss_lst_epoch       + [loss.detach().numpy().tolist()]

            # Finish bar
            bar.finish()            

            val_cepoch_loss    = sum(loss_lst_epoch)/len(loss_lst_epoch)
            val_cepoch_rerror  = sum(rerror_lst_epoch)/len(rerror_lst_epoch)
            val_rerror_lst     = val_rerror_lst + [val_cepoch_rerror]
            val_loss_lst       = val_loss_lst   + [val_cepoch_loss]

            if val_cepoch_loss < val_best_loss:
                # Saving best loss model
                val_best_loss       = val_cepoch_loss
                val_best_loss_epoch = cepoch
                val_best_loss_model = model
                #print("\t\tBest val loss ", str(val_best_loss))
                #print("\t\tCurrent val error ", str(val_best_rerror))
                #torch.save(val_best_loss_model.state_dict(), "val_loss_model_1.pth")

            if val_cepoch_rerror < val_best_rerror:
                # Saving best rerror model
                val_best_rerror       = val_cepoch_rerror
                val_best_rerror_epoch = cepoch
                val_best_rerror_model = model
                print("Saving model with error ", str(val_best_rerror))
                torch.save(val_best_rerror_model.state_dict(), "val_error_model_0.pth")

            if ((cepoch % 500) == 0): 
                print("Saving valid loss and error")
                vname = 'valid_loss_0_' + str(cepoch) + '.pkl'
                with open(vname, 'wb') as f:
                    pickle.dump(val_loss_lst, f)
                tname = 'valid_error_0_' + str(cepoch) + '.pkl'                    
                with open(tname, 'wb') as f:
                    pickle.dump(val_rerror_lst, f)            

    
    # Loading model
    # loaded_model = UNet(in_channels=1, n_classes=1, depth=5, padding=True, up_mode='upsample').to(device)
    # loaded_model.load_state_dict(torch.load("best_val_model.pth"))
    # uOut1 = model(X3)
    # uOut2 = loaded_model(X3)
