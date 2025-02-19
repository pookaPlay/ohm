import pyrtl


def bs_switching_filo(input, switch, STACK_DEPTH=8):
    # This will switch mode the step after switch goes high
    mode = pyrtl.Register(1, 'mode', 0)    
    output = pyrtl.Output(bitwidth=1, name='output')

    with pyrtl.conditional_assignment:
        with switch:
            # Set when switch True.
            mode.next |= 1 - mode        
        with pyrtl.otherwise:
            # Set when switch False.
            mode.next |= mode
        
    # Dualing FILO 
    stack1 = [pyrtl.Register(1) for _ in range(STACK_DEPTH)]
    stack2 = [pyrtl.Register(1) for _ in range(STACK_DEPTH)]    
    with pyrtl.conditional_assignment:
        with mode:
            output |= stack2[0]
            for i in range(0, STACK_DEPTH-1):
                stack2[i].next |= stack2[i + 1]                

            stack1[0].next |= input    
            for i in range(STACK_DEPTH - 1, 0, -1):
                stack1[i].next |= stack1[i - 1]            

        with pyrtl.otherwise:
            output |= stack1[0]
            for i in range(0, STACK_DEPTH-1):
                stack1[i].next |= stack1[i + 1]                

            stack2[0].next |= input
            for i in range(STACK_DEPTH - 1, 0, -1):
                stack2[i].next |= stack2[i - 1]
    
    # Return the output and the contents of stack1 and stack2
    return output
