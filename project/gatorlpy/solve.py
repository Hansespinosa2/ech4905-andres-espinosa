import numpy as np
from objects import Variable, Problem, Expression

# Defining the way problems are made
A = np.array([[1, 2], [4, 0], [0, 4]])  # Coefficients of the constraints
b = np.array([8, 16, 12])  # Right-hand side of the constraints
c = np.array([-3, -5])  # Coefficients of the objective function
# Defining a feasible starting point
x0 = np.zeros(len(c))  # Start with zeros


problem_def = {
    "minimize": [c.T @ x0],
    "constraints": [
        A @ x0 == b,
        x0 >= 0
    ]
}

expression = Expression(A @ x0 == b)

# problem = Problem(problem_def)

