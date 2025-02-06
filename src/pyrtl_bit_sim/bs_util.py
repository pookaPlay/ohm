def int_to_twos_complement_list(value, bit_width):
    if value < 0:
        value = (1 << bit_width) + value
    return [(value >> i) & 1 for i in range(bit_width)]

def twos_complement_list_to_int(bits):
    value = sum(bit << i for i, bit in enumerate(bits))
    if bits[-1] == 1:
        value -= (1 << len(bits))
    return value
