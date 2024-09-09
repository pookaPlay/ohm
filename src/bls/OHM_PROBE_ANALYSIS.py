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
            sumw = np.sum(weights)
            numw = weights.shape[0]
            if sumw == numw:
                numInputs = numw
            else:
                numInputs = 0
            
            effectiveInputs[li][i] = numInputs

    return effectiveInputs

