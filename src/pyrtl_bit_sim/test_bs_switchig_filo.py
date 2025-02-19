import pyrtl
from bs_switching_filo import bs_switching_filo
from bs_util import serialize_int, deserialize_int
from bs_util import generate_dot_file, convert_dot_to_image

pyrtl.set_debug_mode()

def main():
    STACK_DEPTH = 8
    num_bytes = 3
    test_input_int = 1+2+8+32+128  
    bitwidth = STACK_DEPTH  # Bitwidth of the input integer

    # Define the input and output wires
    input = pyrtl.Input(bitwidth=1, name='input')
    switch = pyrtl.Input(bitwidth=1, name='switch')

    # Instantiate the switching_filo component
    output  = bs_switching_filo(input, switch, STACK_DEPTH=STACK_DEPTH)

    # Generate the DOT file
    generate_dot_file('filo.dot')
    convert_dot_to_image('filo.dot', 'filo.png')
    # Simulate the switching_filo component
    sim_trace = pyrtl.SimulationTrace()
    sim = pyrtl.Simulation(tracer=sim_trace)

    # Serialize the input integer to a list of bits
    serialized_input = serialize_int(test_input_int, bitwidth)
    print(f'INPUT {test_input_int}: {serialized_input}')

    reversed_serialized_input = serialized_input[::-1]    

    rint= deserialize_int(reversed_serialized_input)    
    print(f'REVERSED INPUT {rint}: {reversed_serialized_input}')
    
    output_bits = []

    for cycle in range(len(serialized_input) * num_bytes):
        datain = {'input': serialized_input[cycle % len(serialized_input)], 'switch': (cycle + 1) % 8 == 0}
        sim.step(datain)
        output_bits.append(sim.inspect(output))

    # Slice the output bits into num_bytes sequences of bitwidth
    output_slices = [output_bits[i * bitwidth:(i + 1) * bitwidth] for i in range(num_bytes)]
    output_ints = [deserialize_int(bits) for bits in output_slices]

    # Check if each output integer is the same as the input integer
    for i, output_int in enumerate(output_ints):
        print(f"Got {output_int} at byte {i} and expected {rint}")


    # Print the simulation results
    sim_trace.render_trace()

if __name__ == "__main__":
    main()