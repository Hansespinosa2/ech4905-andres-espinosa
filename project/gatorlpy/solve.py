import numpy as np
from lp_reductions import Variable, Problem, Parameter

# Defining the way problems are made
A = Parameter(np.array([[1, 2], [4, 0], [0, 4]]))  # Coefficients of the constraints
b = Parameter(np.array([8, 16, 12]))  # Right-hand side of the constraints
c = Parameter(np.array([-3, -5]))  # Coefficients of the objective function

# Defining a feasible starting point
x = Variable(len(c))  # Start with zeros


problem_def = {
    "minimize": c.T @ x,
    "constraints": [
        A @ x == b,
        x >= 0
    ]
}


problem = Problem(problem_def)

problem.solve()