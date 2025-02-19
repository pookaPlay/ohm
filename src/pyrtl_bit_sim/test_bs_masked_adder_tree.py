from math import log2
import pyrtl
from bs_masked_adder_tree import bs_masked_adder_tree
from bs_util import int_to_twos_complement_list, twos_complement_list_to_int

def main():
    D = 4  # Number of inputs, must be a power of 2
    bit_width = 8  # Bit width of each input
    latency = int(log2(D))  # Latency of the adder tree
    
    # Test data
    inputs = [63, 32, -122, 127]
    masks = [1, 1, 1, 1]
    threshold = 0
    expected = sum(inputs)

    serialized_inputs = [int_to_twos_complement_list(x, bit_width) for x in inputs]
    serialized_threshold = int_to_twos_complement_list(threshold, bit_width)

    # Initialize PyRTL inputs
    pyrtl_inputs = [pyrtl.Input(bitwidth=1, name=f'in{i}') for i in range(D)]
    pyrtl_masks = [pyrtl.Input(bitwidth=1, name=f'ma{i}') for i in range(D)]    
    pyrtl_lsbs = [pyrtl.Input(bitwidth=1, name=f'lsb{i}') for i in range(D)]
    pyrtl_threshold = pyrtl.Input(bitwidth=1, name=f'thresh')

    # Create the adder tree
    sum_out = bs_masked_adder_tree(pyrtl_inputs, pyrtl_masks, pyrtl_lsbs, pyrtl_threshold)
    
    # Simulate the design
    sim_trace = pyrtl.SimulationTrace()
    sim = pyrtl.Simulation(tracer=sim_trace)

    output_bits = []
    # Feed serialized data through the adder tree
    for i in range(bit_width+latency):
        if i < bit_width:
            input_bits = {f'in{j}': serialized_inputs[j][i] for j in range(D)}
            input_bits.update({f'ma{j}': masks[j] for j in range(D)})
            input_bits.update({f'thresh': serialized_threshold[i]})
        else:
            # sign extend
            input_bits = {f'in{j}': serialized_inputs[j][bit_width-1] for j in range(D)}
            input_bits.update({f'ma{j}': masks[j] for j in range(D)})
            input_bits.update({f'thresh': serialized_threshold[bit_width-1]})

        if i == 0:
            input_bits.update({f'lsb{j}': 1 for j in range(D)})
        else:
            input_bits.update({f'lsb{j}': 0 for j in range(D)})
        
        #print(input_bits)
        sim.step(input_bits)
        output_bits.append(sim.inspect(sum_out))

    # Collect the output bit-stream
    other_bits = [sim_trace.trace['sum_out'][i] for i in range(bit_width+latency)]

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