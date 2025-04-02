import numpy as np
from objects import Problem, Parameter, Variable
import cvxpy as cp


def run_test_0():
    # PARAMETERS
    A_arr = np.c_[np.array([[2, 1], [1, 3], [3, 1]]),np.eye(3)]
    b_arr = np.array([10,18,15])
    c_arr = np.r_[np.array([5,4]),np.zeros(3)]
    A = Parameter(A_arr)
    b = Parameter(b_arr)
    c = Parameter(c_arr)
    # VARIABLES
    x = Variable(5) 
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
    x_cvx = cp.Variable(5)
    # PROBLEM
    objective = cp.Minimize(c_arr @ x_cvx)
    constraints = [
        A_arr @ x_cvx == b_arr,
        x_cvx >= 0
    ]
    problem_cvx = cp.Problem(objective, constraints)

    # SOLVE
    problem_cvx.solve()
    print(np.round(problem_cvx.value,2), np.array([np.round(x.value,2) for x in problem_cvx.variables()]))
    print(solution)
    