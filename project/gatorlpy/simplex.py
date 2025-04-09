import numpy as np

FLOAT_POINT_ROUND = 8
EPSILON = 10**-FLOAT_POINT_ROUND

def zero_if_close_to_zero(x):
    return np.where(np.abs(x) < EPSILON, 0, x)

def pivot(tableau:np.ndarray, row_index:int, col_index:int) -> np.ndarray:
    tableau[row_index, :] /= tableau[row_index, col_index]
    for i in range(tableau.shape[0]):
        if i != row_index:
            tableau[i,:] -= tableau[i, col_index] * tableau[row_index,:]
    return tableau

def find_entering_variable(tableau:np.ndarray,rule:str) -> int:
    if rule == 'blands':
        for col_index in range(tableau.shape[1] -1):
            if tableau[-1, col_index] < -EPSILON:
                return col_index
        return None
    else:
        return None
    
def find_leaving_variable(tableau:np.ndarray, entering_col_index:int) -> int:
    ratios = []
    for row_index in range(tableau.shape[0] -1):
        if tableau[row_index, entering_col_index] > EPSILON:
            ratio = tableau[row_index, -1] / tableau[row_index, entering_col_index]
            ratios.append((ratio, row_index))
    return min(ratios, default=(None,None))[1] if ratios else None


def simplex_phase_1(A:np.ndarray, b:np.ndarray):
    m, n = A.shape
    for i in range(m):
        if b[i] < 0:
            A[i, :] *= -1
            b[i] *= -1
    A_aux = np.hstack((A, np.eye(m)))
    c_aux = np.hstack((np.zeros(n), np.ones(m)))
    tableau = np.hstack((A_aux, b.reshape(-1,1)))
    tableau = np.vstack((tableau, np.hstack((-c_aux, [0]))))
    basis = list(range(n,n+m))

    tableau[-1, :-1] = -np.sum(tableau[:-1, :-1], axis=0)
    tableau[-1,-1] = -np.sum(tableau[:-1,-1])

    while True:
        print("Current Tableau \n")
        print(tableau)
        entering_col_index = find_entering_variable(tableau, rule='blands')
        if entering_col_index is None:
            break
        leaving_row_index = find_leaving_variable(tableau, entering_col_index)
        if leaving_row_index is None:
            raise ValueError("Infeasible Solution in Phase 1")
        tableau = pivot(tableau, leaving_row_index, entering_col_index)
        basis[leaving_row_index] = entering_col_index

    if tableau[-1,-1] > EPSILON:
        return None, None, False # Infeasible Solution to original problem
    print(basis)
    return tableau[:m,:n], tableau[:-1,-1], basis[:m]

def simplex_phase_2(A:np.ndarray, b:np.ndarray, c:np.ndarray, basis:list[int]) -> tuple[np.ndarray|bool]:
    m, n = A.shape
    tableau = np.zeros((m+1, n+1))
    tableau[:-1,:-1] = A
    tableau[:-1, -1] = b
    tableau[-1, :-1] = -c

    for row_index, col_index in enumerate(basis):
        tableau[-1, :-1] += c[col_index] * tableau[row_index, :-1]
        tableau[-1,-1] += c[col_index] * tableau[row_index, -1]

    while True:
        entering_col_index = find_entering_variable(tableau, "blands")
        if entering_col_index is None:
            break
        leaving_row_index = find_leaving_variable(tableau, entering_col_index)
        if leaving_row_index is None:
            return -np.inf, None, False # Unbounded solution
        tableau = pivot(tableau, leaving_row_index, entering_col_index)
        basis[leaving_row_index] = entering_col_index
    
    solution = np.zeros(n)
    solution[basis] = zero_if_close_to_zero(tableau[:-1,-1])
    optimal_value = zero_if_close_to_zero(tableau[-1,-1])

    return optimal_value, solution, True

def two_phase_simplex(A:np.ndarray, b:np.ndarray, c:np.ndarray) -> tuple[np.ndarray|bool]:
    A, b, basis = simplex_phase_1(A,b)
    if A is None:
        return None, None, False # Problem is infeasible
    f_star, x_star, feasible = simplex_phase_2(A,b,-c,basis)
    return f_star, x_star, feasible