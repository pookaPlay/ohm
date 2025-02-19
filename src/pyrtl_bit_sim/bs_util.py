import os
import pyrtl

def int_to_twos_complement_list(value, bit_width):
    if value < 0:
        value = (1 << bit_width) + value
    return [(value >> i) & 1 for i in range(bit_width)]

def twos_complement_list_to_int(bits):
    value = sum(bit << i for i, bit in enumerate(bits))
    if bits[-1] == 1:
        value -= (1 << len(bits))
    return value

def serialize_int(value, bitwidth):
    return [(value >> i) & 1 for i in range(bitwidth)]

def deserialize_int(bits):
    return sum([bit << i for i, bit in enumerate(bits)])

def generate_dot_file(filename='filo.dot'):
    with open(filename, 'w') as f:
        pyrtl.output_to_graphviz(file=f)

def convert_dot_to_image(dot_filename, output_filename):
    os.system(f'dot -Tpng {dot_filename} -o {output_filename}')
