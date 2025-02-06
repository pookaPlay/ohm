import pyrtl

def fir(x, bs):
    rwidth = len(x)  # bitwidth of the registers
    ntaps = len(bs) # number of coefficients

    zs = [x] + [pyrtl.Register(rwidth) for _ in range(ntaps-1)]
    for i in range(1,ntaps):
        zs[i].next <<= zs[i-1]

    # produce the final sum of products
    return sum(z*b for z,b in zip(zs, bs))

x = pyrtl.Input(8, 'x')
y = pyrtl.Output(8, 'y')
y <<= fir(x, [0, 1])

sim = pyrtl.Simulation()
sim.step_multiple({'x':[0, 9, 18, 8, 17, 7, 16, 6, 15, 5]})
sim.tracer.render_trace()