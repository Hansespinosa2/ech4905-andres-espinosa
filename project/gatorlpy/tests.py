import numpy as np
import cvxpy as cp
from objects import Problem, Parameter, Variable


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
    if problem_cvx.value != np.inf:
        print(np.round(problem_cvx.value,2), np.array([np.round(x.value,2) if x is not None else 0 for x in problem_cvx.variables() ]))
    else:
        print(problem_cvx.solution)
    print(solution)
    

def run_test_1():
    A_arr = np.hstack((np.array([[2, 1], [1, 3]]),np.eye(2)))
    b_arr = np.array([4, 3])
    c_arr = np.hstack((np.array([3, 2]), np.zeros(2)))
    A = Parameter(A_arr)
    b = Parameter(b_arr)
    c = Parameter(c_arr)
    # VARIABLES
    x = Variable(4) 
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
    x_cvx = cp.Variable(4)
    # PROBLEM
    objective = cp.Minimize(c_arr @ x_cvx)
    constraints = [
        A_arr @ x_cvx == b_arr,
        x_cvx >= 0
    ]
    problem_cvx = cp.Problem(objective, constraints)

    # SOLVE
    problem_cvx.solve()
    if problem_cvx.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        f_star = np.round(problem_cvx.value, 2)
        x_star = np.array([np.round(var.value, 2) if var.value is not None else 0 for var in problem_cvx.variables()])
        feasible = True
    else:
        f_star = None
        x_star = None
        feasible = False

    print(f"f_star: {f_star}, x_star: {x_star}, feasible: {feasible}")
    print(solution)
    # Compare results
    if feasible and solution is not None:
        passed = np.isclose(f_star, solution[0], atol=1e-2) and np.allclose(x_star, solution[1], atol=1e-2)
    else:
        passed = False

    print(f"Test passed: {passed} \n")



if __name__ == "__main__":
    run_test_0()
    run_test_1()
