import numpy as np
from objects import Problem, Parameter, Variable
import cvxpy as cp


def run_test_0():
    # PARAMETERS
    A_arr = np.array([[2, 1], [1, 3], [3, 1]])
    A = Parameter(A_arr)
    b_arr = np.array([10,18,15])
    b = Parameter(b_arr)
    c_arr = np.array([5,4])
    c = Parameter(c_arr)
    # VARIABLES
    x = Variable(2) 
    # PROBLEM
    problem = Problem({
        'minimize': c.T @ x,
        'subject to': [
            A @ x == b,
            x >= 0
        ]
    })
    solution = problem.solve()

    # VARIABLES
    x_cvx = cp.Variable(2)
    # PROBLEM
    objective = cp.Minimize(c_arr.T @ x_cvx)
    constraints = [
        A_arr @ x_cvx == b_arr,
        x_cvx >= 0
    ]
    problem_cvx = cp.Problem(objective, constraints)

    # SOLVE
    result = problem_cvx.solve()
    print(result)
    print(solution)
    