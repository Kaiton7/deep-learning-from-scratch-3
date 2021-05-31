if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
# import dezero's simple_core explicitly
import dezero
if not dezero.is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import setup_variable
    setup_variable()


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y

def f(x):
    y  = x**4 - 2*x**2
    return y

def gx2(x):
    return 12*x**2-4

#x0 = Variable(np.array(0.0))
#x1 = Variable(np.array(2.0))
x = Variable(np.array(2.0))

lr = 0.001
iters = 10

for i in range(iters):
    print(x)

    y = f(x)
    x.cleargrad()
    #x1.cleargrad()
    y.backward()

    #x0.data -= lr * x0.grad
    #x1.data -= lr * x1.grad
    x.data -= x.grad/gx2(x.data)