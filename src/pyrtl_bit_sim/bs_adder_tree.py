from bs_add import bs_add

def bs_adder_tree(inputs):
    D = len(inputs)
    assert (D & (D - 1)) == 0, "Number of inputs must be a power of 2"

    while len(inputs) > 1:
        next_level = []
        for i in range(0, len(inputs), 2):
            sum = bs_add(inputs[i], inputs[i + 1])
            next_level.append(sum)
        inputs = next_level

    return inputs[0]
