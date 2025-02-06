import pyrtl


def dual_filo(input, STACK_DEPTH):

    mode = pyrtl.Register(1, 'rw', 0)
    
    #stack1 = [input] + [pyrtl.Register(1) for _ in range(D)]
    stack1 = [pyrtl.Register(1) for _ in range(STACK_DEPTH)]

    if mode == 0:
        for i in range(1, STACK_DEPTH):
            stack1[i].next <<= stack1[i-1]
    
    return stack1[0]


#pyrtl.select(a < 5, truecase=a, falsecase=5)
