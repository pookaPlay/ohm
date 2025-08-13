import pyrtl

def one_bit_add(a, b, carry_in):
    assert len(a) == len(b) == 1  # len returns the bitwidth
    sum = a ^ b ^ carry_in  # operators on WireVectors build the hardware
    carry_out = a & b | a & carry_in | b & carry_in
    return sum, carry_out

def bs_add(a, b, carry_reset):
    
    carry_in = pyrtl.Register(bitwidth=1)
    carry_in_value = pyrtl.mux(carry_reset, carry_in, 0)
        
    sum_out, carry_out = one_bit_add(a, b, carry_in_value)

    carry_in.next <<= carry_out

    return sum_out


