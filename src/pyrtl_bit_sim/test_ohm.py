import pyrtl
from bs_adder_tree import bs_adder_tree

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

    # Test data
    inputs = [3, -1, 2, -2]
    serialized_inputs = [int_to_twos_complement_list(x, bit_width) for x in inputs]

    # Initialize PyRTL inputs
    pyrtl_inputs = [pyrtl.Input(bitwidth=1, name=f'in{i}') for i in range(D)]

    # Create the adder tree
    sum_out = bs_adder_tree(pyrtl_inputs)

    # Connect sum_out to an output
    sum_output = pyrtl.Output(name='sum_out')
    sum_output <<= sum_out

    # Simulate the design
    sim_trace = pyrtl.SimulationTrace()
    sim = pyrtl.Simulation(tracer=sim_trace)

    # Feed serialized data through the adder tree
    for i in range(bit_width):
        input_bits = {f'in{j}': serialized_inputs[j][i] for j in range(D)}
        sim.step(input_bits)

    # Collect the output bit-stream
    output_bits = [sim_trace.trace['sum_out'][i] for i in range(bit_width)]

    # Deserialize output
    result = twos_complement_list_to_int(output_bits)
    print(f"Result: {result}")

    # Render the trace
    sim_trace.render_trace()

if __name__ == "__main__":
    main()