from math import log2
import pyrtl
from filo import filo
from bs_util import int_to_twos_complement_list, twos_complement_list_to_int

def generate_dot_file(filename='filo.dot'):
    with open(filename, 'w') as f:
        pyrtl.output_to_graphviz(file=f)

def main():
    D = 4  # Number of inputs, must be a power of 2
    bit_width = 8  # Bit width of each input
    latency = int(log2(D))  # Latency of the adder tree
    
    # Test data
    input = 63
    serialized_input = int_to_twos_complement_list(input, bit_width)

    # Initialize PyRTL input
    pyrtl_input = pyrtl.Input(bitwidth=1, name=f'in0')

    # Create the filo
    filo_out = filo(pyrtl_input, bit_width)

    # Connect sum_out to an output
    filo_output = pyrtl.Output(name='sum_out')
    filo_output <<= filo_out

    # Generate the DOT file
    generate_dot_file()

    # Simulate the design
    sim_trace = pyrtl.SimulationTrace()
    sim = pyrtl.Simulation(tracer=sim_trace)

    # Feed serialized data through the adder tree
    for i in range(bit_width+latency+latency):
        if i < bit_width:
            input_bits = {f'in0': serialized_input[i]}
        else:
            # sign extend
            input_bits = {f'in0': serialized_input[bit_width-1]}
        sim.step(input_bits)

    # Collect the output bit-stream
    output_bits = [sim_trace.trace['sum_out'][i] for i in range(latency, bit_width+latency+latency)]

    # Deserialize output
    result = twos_complement_list_to_int(output_bits)
    # Render the trace
    sim_trace.render_trace()


if __name__ == "__main__":
    main()