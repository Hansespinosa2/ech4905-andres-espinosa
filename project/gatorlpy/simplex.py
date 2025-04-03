import numpy as np

EPSILON = 1e-10

def to_tableau(A:np.ndarray, b:np.ndarray, c:np.ndarray) ->np.ndarray:
    xb = np.hstack([A, b[:, None]])  # Combine A and b (column-wise stack)
    z = np.hstack([c, [0]])  # Append a zero to c
    return np.vstack([xb, z])  # Combine xb and z (row-wise stack)

def can_be_improved(tableau:np.ndarray) -> bool:
    z = tableau[-1]
    return np.any(z[:-1] < 0)

def get_pivot_position(tableau:np.ndarray) -> tuple[int]:
    z = tableau[-1]
    col_index = np.argmin(z[:-1]) # Steepest descent
    ratios = np.array([
        row[-1] / row[col_index] if row[col_index] >= 0 else np.inf
        for row in tableau[:-1]
    ])

    row_index = np.argmin(ratios)
    return row_index, col_index

def pivot_step(tableau:np.ndarray, pivot_position:tuple[int]) ->np.ndarray:
    new_tableau = tableau.copy()
    
    pivot_row_index, pivot_col_index = pivot_position
    pivot_value = tableau[pivot_row_index, pivot_col_index]
    new_tableau[pivot_row_index, :] = tableau[pivot_row_index, :] / pivot_value  
    
    for row_index, row in enumerate(tableau):
        if row_index != pivot_row_index:
            multiplier = new_tableau[pivot_row_index, :] * tableau[row_index, pivot_col_index]
            new_tableau[row_index, :] = tableau[row_index, :] - multiplier
    
    return new_tableau

def is_basic(column:np.ndarray) -> bool:
    return np.sum(column) == 1 and np.count_nonzero(column == 0) == len(column) - 1

def get_solution(tableau:np.ndarray) -> np.ndarray:
    columns = tableau.T
    solutions = []
    for column in columns[:-1]:
        solution = 0
        if is_basic(column):
            one_index = np.where(column == 1)[0][0]  
            solution = columns[-1, one_index]
        solutions.append(solution)
    return np.array(solutions)

def phase_1(A: np.ndarray, b:np.ndarray) -> tuple[np.ndarray]:
    m,n = A.shape
    A_aux = np.hstack([A, np.eye(m)])
    c_aux = np.hstack([np.zeros(n), np.ones(m)])
    tableau = to_tableau(A_aux, b, c_aux)

    while can_be_improved(tableau):
        pivot_position = get_pivot_position(tableau)
        print(pivot_position)
        tableau = pivot_step(tableau, pivot_position)
    if tableau[-1,-1] > 0:
        return None, None
    
    return tableau[:m,:n], tableau[:m,-1]

def simplex(A:np.ndarray, b:np.ndarray, c:np.ndarray) ->tuple[np.ndarray|bool]:
    A, b = phase_1(A,b)
    if A is None:
        return 0,0, False
    tableau = to_tableau(A, b, c)

    while can_be_improved(tableau):
        pivot_position = get_pivot_position(tableau)
        tableau = pivot_step(tableau, pivot_position)

    x_star = get_solution(tableau)
    f_star = c.T @ x_star
    return (f_star, x_star, True)

