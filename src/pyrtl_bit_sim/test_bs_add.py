import pyrtl
from bs_add import bs_add
from bs_util import int_to_twos_complement_list, twos_complement_list_to_int

def main():
    bit_width = 8  # Bit width of each input
    latency = 0
    # Test data
    a = 3
    b = 4
    expected = a+b

    sa = int_to_twos_complement_list(a, bit_width)
    sb = int_to_twos_complement_list(b, bit_width)
    # Initialize PyRTL inputs
    pyrtl_a = pyrtl.Input(bitwidth=1, name=f'a')
    pyrtl_b = pyrtl.Input(bitwidth=1, name=f'b')    
    pyrtl_reset = pyrtl.Input(bitwidth=1, name=f'reset')
    pyrtl_sum = pyrtl.Output(bitwidth=1, name='sum')
    # Create the adder tree
    #um_out, carry_out = 
    pyrtl_sum <<= bs_add(pyrtl_a, pyrtl_b, pyrtl_reset)
    #pyrtl_carry <<= carry_out

    # Simulate the design
    sim_trace = pyrtl.SimulationTrace()
    sim = pyrtl.Simulation(tracer=sim_trace)

    output_bits = []
    # Feed serialized data through the adder tree
    for i in range(bit_width+latency):
        if i == 0:
            reset = 1
        else:
            reset = 0
        if i < bit_width:
            input_bits = {f'a': sa[i], f'b': sb[i], f'reset': reset}
        else:
            input_bits = {f'a': sa[bit_width-1], f'b': sb[bit_width-1], f'reset': reset}
        
        #print(input_bits)
        sim.step(input_bits)
        output_bits.append(sim.inspect(pyrtl_sum))

    # Collect the output bit-stream
    #other_bits = [sim_trace.trace['sum'][i] for i in range(bit_width+latency)]

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