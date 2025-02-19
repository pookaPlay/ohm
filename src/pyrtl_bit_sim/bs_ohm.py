from bs_switching_filo import bs_switching_filo
from bs_masked_adder_tree import bs_masked_adder_tree
import pyrtl

def bs_ohm(inputs, masks, lsbs, biases, ptf_weights, ptf_threshold, stf_weights, stf_threshold, param):
    
    if 'STACK_DEPTH' not in param:
        param['STACK_DEPTH'] = 8
    
    ptfOut = pyrtl.Output(bitwidth=1, name='ptfOut')
    stfOut = pyrtl.Output(bitwidth=1, name='stfOut')

    sticky_states = [pyrtl.Register(bitwidth=1, name=f'sticky_state_{i}') for i in range(len(inputs)*2)]
    ready_register = pyrtl.Register(bitwidth=1, name=f'ready_register')

    switch = 0
    # masks, lsbs, threshold    
    bats = [bs_masked_adder_tree(inputs, mask, lsbs, bias) for (mask, bias) in zip(masks, biases)]
    
    lsb2msbs = [bs_switching_filo(bat, switch, param['STACK_DEPTH']) for bat in bats]
    
    ptfmasks = [lsb2msb for lsb2msb in lsb2msbs]
    
    mirrored_ptfmasks = ptfmasks + [~mask for mask in ptfmasks]

    ptfOut <<= bs_masked_adder_tree(ptf_weights, mirrored_ptfmasks, lsbs, ptf_threshold)
    
    stfOut <<= bs_masked_adder_tree(stf_weights, sticky_states, lsbs, stf_threshold)
    
    # Update sticky_states 
    for i in range(len(mirrored_ptfmasks)):
        with pyrtl.conditional_assignment:
            with mirrored_ptfmasks[i] != ptfOut:
                sticky_states[i].next |= 1
            with pyrtl.otherwise:
                sticky_states[i].next |= sticky_states[i]

    ready_register.next <<= stfOut

    msb2lsb = bs_switching_filo(inputs, switch, param['STACK_DEPTH'])



def bs_ohm_test():
    D = 4  # Number of inputs, must be a power of 2

    inputs = [pyrtl.Input(bitwidth=1, name=f'in{i}') for i in range(D)]
    # univariate case
    masks = [[pyrtl.Const(1, bitwidth=1) if i == j else pyrtl.Const(0, bitwidth=1) for i in range(D)] for j in range(D)]
    lsbs = [pyrtl.Input(bitwidth=1, name=f'lsb{i}') for i in range(D)]

    biases = [pyrtl.Input(bitwidth=1, name=f'bias{i}') for i in range(D)]
    ptf_weights = [pyrtl.Input(bitwidth=1, name=f'ptf_weight{i}') for i in range(D)]
    ptf_threshold = pyrtl.Input(bitwidth=1, name='ptf_threshold')
    outputs = [pyrtl.Output(bitwidth=1, name=f'out{i}') for i in range(D)]

    param = {'STACK_DEPTH': 8}

    bs_ohm(inputs, masks, lsbs, biases, ptf_weights, ptf_threshold, param)

    # Simulate the circuit
    sim_trace = pyrtl.SimulationTrace()
    sim = pyrtl.Simulation(tracer=sim_trace)

    # Run the simulation for a few cycles
    for cycle in range(10):
        sim.step({})

    # Print the simulation results
    sim_trace.render_trace()

if __name__ == "__main__":
    bs_ohm_test()