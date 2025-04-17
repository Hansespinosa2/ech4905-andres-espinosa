import numpy as np
from lp_reductions import Variable, Problem, Parameter

### REPLACE BELOW EXAMPLE WITH YOUR PROBLEM ###
# PARAMETERS
A = Parameter(np.array([[2, 1], [1, 3], [3, 1]]))
b = Parameter(np.array([10,18,15]))
c = Parameter(np.array([5,4]))
# VARIABLES
x = Variable(2)
# PROBLEM
problem = Problem({
    'maximize': c.T @ x,
    'subject to': [
        A @ x <= b,
        x >= 0
    ]
})

solution = problem.solve()
print(solution)
