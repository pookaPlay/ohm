from math import log2
import pyrtl
from bs_adder_tree import bs_adder_tree
from bs_util import int_to_twos_complement_list, twos_complement_list_to_int

def generate_dot_file(filename='adder_tree.dot'):
    with open(filename, 'w') as f:
        pyrtl.output_to_graphviz(file=f)

def main():
    D = 4  # Number of inputs, must be a power of 2
    bit_width = 8  # Bit width of each input
    latency = int(log2(D))  # Latency of the adder tree
    
    # Test data
    inputs = [127, 127, -122, 127]
    expected = sum(inputs)

    serialized_inputs = [int_to_twos_complement_list(x, bit_width) for x in inputs]

    # Initialize PyRTL inputs
    pyrtl_inputs = [pyrtl.Input(bitwidth=1, name=f'in{i}') for i in range(D)]

    # Create the adder tree
    sum_out = bs_adder_tree(pyrtl_inputs)

    # Connect sum_out to an output
    sum_output = pyrtl.Output(name='sum_out')
    sum_output <<= sum_out

    # Generate the DOT file
    generate_dot_file()

    # Simulate the design
    sim_trace = pyrtl.SimulationTrace()
    sim = pyrtl.Simulation(tracer=sim_trace)

    # Feed serialized data through the adder tree
    for i in range(bit_width+latency+latency):
        if i < bit_width:
            input_bits = {f'in{j}': serialized_inputs[j][i] for j in range(D)}
        else:
            # sign extend
            input_bits = {f'in{j}': serialized_inputs[j][bit_width-1] for j in range(D)}
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