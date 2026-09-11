from math import log2
import pyrtl
from bs_masked_adder_tree import bs_masked_adder_tree

def int_to_twos_complement_list(value, bit_width):
    if value < 0:
        value = (1 << bit_width) + value
    return [(value >> i) & 1 for i in range(bit_width)]

def twos_complement_list_to_int(bits):
    value = sum(bit << i for i, bit in enumerate(bits))
    if bits[-1] == 1:
        value -= (1 << len(bits))
    return value

def main():
    D = 4  # Number of inputs, must be a power of 2
    bit_width = 8  # Bit width of each input
    latency = int(log2(D))  # Latency of the adder tree
    
    # Test data
    inputs = [63, 32, -122, 127]
    masks = [1, 1, 0, 1]
    expected = sum(inputs)

    serialized_inputs = [int_to_twos_complement_list(x, bit_width) for x in inputs]

    # Initialize PyRTL inputs
    pyrtl_inputs = [pyrtl.Input(bitwidth=1, name=f'in{i}') for i in range(D)]
    pyrtl_masks = [pyrtl.Input(bitwidth=1, name=f'ma{i}') for i in range(D)]

    # Create the adder tree
    sum_out = bs_masked_adder_tree(pyrtl_inputs, pyrtl_masks, lsbs, threshold=0)

    # Connect sum_out to an output
    sum_output = pyrtl.Output(name='sum_out')
    sum_output <<= sum_out

    # Simulate the design
    sim_trace = pyrtl.SimulationTrace()
    sim = pyrtl.Simulation(tracer=sim_trace)

    # Feed serialized data through the adder tree
    for i in range(bit_width+latency):
        if i < bit_width:
            input_bits = {f'in{j}': serialized_inputs[j][i] for j in range(D)}
            input_bits.update({f'ma{j}': masks[j] for j in range(D)})
        else:
            # sign extend
            input_bits = {f'in{j}': serialized_inputs[j][bit_width-1] for j in range(D)}
            input_bits.update({f'ma{j}': masks[j] for j in range(D)})

        sim.step(input_bits)

    # Collect the output bit-stream
    output_bits = [sim_trace.trace['sum_out'][i] for i in range(bit_width+latency)]

    # Deserialize output
    result = twos_complement_list_to_int(output_bits)
    # Render the trace
    sim_trace.render_trace()

    if result != expected:
        print(f"PROBLEM ===>   Result: {result}   Expected: {expected}")
    else: 
        print(f"Result: {result}   Expected: {expected}")


if __name__ == "__main__":
    main()