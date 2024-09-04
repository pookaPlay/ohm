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
    allweights = np.zeros((ohm.numLayers, ohm.numStack*ohm.numInputs*2))
    allthresh = np.zeros((ohm.numLayers, ohm.numStack))

    for li in range(len(ohm.paramStackMem)):
        for i in range(len(ohm.paramStackMem[li])):
            weights = ohm.paramStackMem[li][i].GetLSBIntsHack()
            allweights[li][i*len(weights):(i+1)*len(weights)] = weights
                                        
    for li in range(len(ohm.paramStackMem)):
        for i in range(len(ohm.paramStackMem[li])):
            thresh = ohm.paramThreshMem[li][i].GetLSBIntsHack()
            allthresh[li][i] = thresh[0]

    return allweights, allthresh

