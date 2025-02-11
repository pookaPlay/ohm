from bs_add import bs_add
import pyrtl
import operator
from functools import reduce

def bs_masked_adder_tree(inputs, masks, lsbs, threshold):
    D = len(inputs)
    assert (D & (D - 1)) == 0, "Number of inputs must be a power of 2"
    assert len(inputs) == len(masks), "Inputs and masks must have the same length"
    assert len(inputs) == len(lsbs), "Inputs and lsbs must have the same length"

    # Apply logical AND between inputs and masks
    inputs = [inp & mask for inp, mask in zip(inputs, masks)]
    
    # Set reset as the logical OR of the lsbs    
    reset = pyrtl.WireVector(bitwidth=1, name='reset')
    reset <<= reduce(operator.or_, lsbs)

    #reset = lsbs[0]

    while len(inputs) > 1:
        next_level = []
        for i in range(0, len(inputs), 2):
            sum = bs_add(inputs[i], inputs[i + 1], reset)
            next_level.append(sum)
        inputs = next_level

    out = bs_add(inputs[0], threshold, reset)

    return out
