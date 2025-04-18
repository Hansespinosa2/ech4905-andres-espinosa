import numpy as np
import cvxpy as cp
from lp_reductions import Problem, Parameter, Variable

def get_test_results(glp_prob:Problem, cvxpy_prob:cp.Problem, test_id:str):
    glp_solution = glp_prob.solve()
    cvx_solution = cvxpy_prob.solve()

    if cvxpy_prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        f_star = np.round(cvxpy_prob.value, 2)
        x_star = np.array([np.round(var.value, 2) if var.value is not None else 0 for var in cvxpy_prob.variables()])
        feasible = True
    else:
        f_star = None
        x_star = None
        feasible = False
    # Compare results
    if feasible and cvx_solution is not None:
        passed = np.isclose(f_star, glp_solution[0].reshape(-1,1), atol=1e-2) and np.allclose(x_star, glp_solution[1].reshape(-1,1), atol=1e-2)
    else:
        passed = False
    print(f"Test ID: {test_id}")
    print(f"CVX: {(f_star, x_star, feasible)}")
    print(f"GatORPy: {glp_solution}")
    print(f"Test passed: {passed} \n")


def run_test_0a():
    # PARAMETERS
    A_arr = np.c_[np.array([[2, 1], [1, 3], [3, 1]]),np.eye(3)]
    b_arr = np.array([10,18,15])
    c_arr = np.r_[np.array([5,4]),np.zeros(3)]
    A = Parameter(A_arr)
    b = Parameter(b_arr)
    c = Parameter(c_arr)
    zeros = Parameter(np.zeros_like(c_arr))
    # VARIABLES
    x = Variable(5) 
    # PROBLEM
    problem = Problem({
        'minimize': c.T @ x,
        'subject to': [
            A @ x == b,
            x >= zeros
        ]
    })

    # VARIABLES
    x_cvx = cp.Variable(5)
    # PROBLEM
    objective = cp.Minimize(c_arr @ x_cvx)
    constraints = [
        A_arr @ x_cvx == b_arr,
        x_cvx >= 0
    ]
    problem_cvx = cp.Problem(objective, constraints)

    get_test_results(problem, problem_cvx, "1 matrix constraint, 1 var, n = 5, Slack Form, Minimize")
    

def run_test_1a():
    A_arr = np.hstack((np.array([[2, 1], [1, 3]]),np.eye(2)))
    b_arr = np.array([4, 3])
    c_arr = np.hstack((np.array([3, 2]), np.zeros(2)))
    A = Parameter(A_arr)
    b = Parameter(b_arr)
    c = Parameter(c_arr)
    zeros = Parameter(np.zeros_like(c_arr))
    # VARIABLES
    x = Variable(4) 
    # PROBLEM
    problem = Problem({
        'minimize': c.T @ x,
        'subject to': [
            A @ x == b,
            x >= zeros
        ]
    })
    # VARIABLES
    x_cvx = cp.Variable(4)
    # PROBLEM
    objective = cp.Minimize(c_arr @ x_cvx)
    constraints = [
        A_arr @ x_cvx == b_arr,
        x_cvx >= 0
    ]
    problem_cvx = cp.Problem(objective, constraints)

    get_test_results(problem, problem_cvx, "1 matrix constraint, 1 var, n = 4, Slack Form, Minimize")

def run_test_0b():
    # PARAMETERS
    A_arr = np.c_[np.array([[2, 1], [1, 3], [3, 1]]),np.eye(3)]
    b_arr = np.array([10,18,15])
    c_arr = np.r_[np.array([5,4]),np.zeros(3)]
    A = Parameter(A_arr)
    b = Parameter(b_arr)
    c = Parameter(c_arr)
    zeros = Parameter(np.zeros_like(c_arr))
    # VARIABLES
    x = Variable(5) 
    # PROBLEM
    problem = Problem({
        'maximize': c.T @ x,
        'subject to': [
            A @ x == b,
            x >= zeros
        ]
    })

    # VARIABLES
    x_cvx = cp.Variable(5)
    # PROBLEM
    objective = cp.Maximize(c_arr @ x_cvx)
    constraints = [
        A_arr @ x_cvx == b_arr,
        x_cvx >= 0
    ]
    problem_cvx = cp.Problem(objective, constraints)

    get_test_results(problem, problem_cvx, "1 matrix constraint, 1 var, n = 5, Slack Form, Maximize")
    

def run_test_1b():
    A_arr = np.hstack((np.array([[2, 1], [1, 3]]),np.eye(2)))
    b_arr = np.array([4, 3])
    c_arr = np.hstack((np.array([3, 2]), np.zeros(2)))
    A = Parameter(A_arr)
    b = Parameter(b_arr)
    c = Parameter(c_arr)
    zeros = Parameter(np.zeros_like(c_arr))
    # VARIABLES
    x = Variable(4) 
    # PROBLEM
    problem = Problem({
        'maximize': c.T @ x,
        'subject to': [
            A @ x == b,
            x >= zeros
        ]
    })
    # VARIABLES
    x_cvx = cp.Variable(4)
    # PROBLEM
    objective = cp.Maximize(c_arr @ x_cvx)
    constraints = [
        A_arr @ x_cvx == b_arr,
        x_cvx >= 0
    ]
    problem_cvx = cp.Problem(objective, constraints)

    get_test_results(problem, problem_cvx, "1 matrix constraint, 1 var, n = 4, Slack Form, Maximize")

def run_test_2():
 # PARAMETERS
    A_arr = np.c_[-np.array([[2, 1], [1, 3], [3, 1]]),np.eye(3)]
    b_arr = -np.array([10,18,15])
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

    # VARIABLES
    x_cvx = cp.Variable(5)
    # PROBLEM
    objective = cp.Minimize(c_arr @ x_cvx)
    constraints = [
        A_arr @ x_cvx == b_arr,
        x_cvx >= 0
    ]
    problem_cvx = cp.Problem(objective, constraints)

    get_test_results(problem, problem_cvx, "1 matrix constraint, 1 var, n = 5, Slack Form, Minimize")

def run_test_3():
        # PARAMETERS
    A_arr = np.c_[np.array([[2, 1], [1, 3], [3, 1]])]
    b_arr = np.array([10,18,15])
    c_arr = np.r_[np.array([5,4])]
    A = Parameter(A_arr)
    b = Parameter(b_arr)
    c = Parameter(c_arr)
    # VARIABLES
    x = Variable(2) 
    # PROBLEM
    problem = Problem({
        'minimize': c.T @ x,
        'subject to': [
            A @ x >= b,
            x >= 0
        ]
    })

    # VARIABLES
    x_cvx = cp.Variable(2)
    # PROBLEM
    objective = cp.Minimize(c_arr @ x_cvx)
    constraints = [
        A_arr @ x_cvx >= b_arr,
        x_cvx >= 0
    ]
    problem_cvx = cp.Problem(objective, constraints)

    get_test_results(problem, problem_cvx, "1 matrix constraint, 1 var, n = 2, Standard Form, Minimize")

def run_test_4a():
    # PARAMETERS
    A_arr =np.array([[1, 0], [0, 2]])
    A_eq = np.array([[3,2]])
    b_arr = np.array([4,12])
    b_eq = np.array([18,])
    c_arr = np.array([3,5])
    A = Parameter(A_arr)
    A_e = Parameter(A_eq)
    b = Parameter(b_arr)
    b_e = Parameter(b_eq)
    c = Parameter(c_arr)
    # VARIABLES
    x = Variable(2) 
    # PROBLEM
    problem = Problem({
        'minimize': c.T @ x,
        'subject to': [
            A @ x <= b,
            A_e @ x == b_e,
            x >= 0
        ]
    })

    # VARIABLES
    x_cvx = cp.Variable(2)
    # PROBLEM
    objective = cp.Minimize(c_arr @ x_cvx)
    constraints = [
        A_arr @ x_cvx <= b_arr,
        A_eq @ x_cvx == b_eq, 
        x_cvx >= 0
    ]
    problem_cvx = cp.Problem(objective, constraints)

    get_test_results(problem, problem_cvx, "2 matrix constraint, 1 var, n = 2, Mixed Eq and Leq, Minimize")

def run_test_hw5_1():
    # Parameters
    A_arr = np.array([[1,1],[1.2,0.5]])
    b_arr = np.array([1,1])
    c_arr = np.array([1.2,1])
    A = Parameter(A_arr)
    b = Parameter(b_arr)
    c = Parameter(c_arr)
    # Variables
    y = Variable(2)
    # Problem
    problem = Problem({
        'maximize': c.T @ y,
        'subject to': [
            A @ y <= b,
            y >= 0,
            y <= 1
        ]
    })

    # CVX Vars
    y_cvx = cp.Variable(2)
    # CVX Problem
    objective = cp.Maximize(c_arr @ y_cvx)
    constraints = [
        A_arr @ y_cvx <= b_arr,
        y_cvx >= 0,
        y_cvx <= 1
    ]
    problem_cvx = cp.Problem(objective, constraints)
    
    get_test_results(problem, problem_cvx, "HW5 Problem 1 initial LP relaxation")
    # branch and bound arrays
    I_y_1_arr = np.array([[1,0],[0,0]])
    I_y_2_arr = np.array([[0,0],[0,1]])
    I_y_1 = Parameter(I_y_1_arr)
    I_y_2 = Parameter(I_y_2_arr)
    # LEFT SPLIT
    # Problem
    problem = Problem({
        'maximize': c.T @ y,
        'subject to': [
            A @ y <= b,
            y >= 0,
            y <= 1,
            I_y_1 @ y <= 0,
        ]
    })

    # CVX Vars
    y_cvx = cp.Variable(2)
    # CVX Problem
    objective = cp.Maximize(c_arr @ y_cvx)
    constraints = [
        A_arr @ y_cvx <= b_arr,
        y_cvx >= 0,
        y_cvx <= 1,
        I_y_1_arr @ y_cvx <= 0
    ]
    problem_cvx = cp.Problem(objective, constraints)

    get_test_results(problem, problem_cvx, "HW5 Problem 1 Branch and Bound Left Split y_1 <= 0")
    
    # RIGHT SPLIT
    # Problem
    problem = Problem({
        'maximize': c.T @ y,
        'subject to': [
            A @ y <= b,
            y >= 0,
            y <= 1,
            I_y_1 @ y >= 1,
        ]
    })

    # CVX Vars
    y_cvx = cp.Variable(2)
    # CVX Problem
    objective = cp.Maximize(c_arr @ y_cvx)
    constraints = [
        A_arr @ y_cvx <= b_arr,
        y_cvx >= 0,
        y_cvx <= 1,
        I_y_1_arr @ y_cvx >= 1
    ]
    problem_cvx = cp.Problem(objective, constraints)

    get_test_results(problem, problem_cvx, "HW5 Problem 1 Branch and Bound Right Split y_1 >=1")


if __name__ == "__main__":
    run_test_0a()
    run_test_0b()
    run_test_1a()
    run_test_1b()
    run_test_2() # seems to fail when it is -Ax == -b
    run_test_3()
    run_test_4a()
    run_test_hw5_1()
