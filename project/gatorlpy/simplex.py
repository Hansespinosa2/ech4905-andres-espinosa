import numpy as np


def to_tableau(A:np.ndarray, b:np.ndarray, c:np.ndarray) ->np.ndarray:
    xb = np.hstack([A, b[:, None]])  # Combine A and b (column-wise stack)
    z = np.hstack([c, [0]])  # Append a zero to c
    return np.vstack([xb, z])  # Combine xb and z (row-wise stack)

def can_be_improved(tableau:np.ndarray) -> bool:
    z = tableau[-1]
    return np.any(z[:-1] < 0)

def get_pivot_position(tableau:np.ndarray) -> tuple[int]:
    z = tableau[-1]
    column = np.argmin(z[:-1]) 
    
    ratios = []
    
    restrictions = np.array([
        np.inf if eq[column] <= 0 else eq[-1] / eq[column] 
        for eq in tableau[:-1]
    ])

    row = np.argmin(restrictions)
    return row, column

def pivot_step(tableau:np.ndarray, pivot_position:tuple[int]) ->np.ndarray:
    new_tableau = tableau.copy()
    
    i, j = pivot_position
    pivot_value = tableau[i, j]
    new_tableau[i, :] = tableau[i, :] / pivot_value  
    
    for eq_i, eq in enumerate(tableau):
        if eq_i != i:
            multiplier = new_tableau[i, :] * tableau[eq_i, j]
            new_tableau[eq_i, :] = tableau[eq_i, :] - multiplier
    
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

def simplex(A:np.ndarray, b:np.ndarray, c:np.ndarray) ->tuple[np.ndarray|bool]:
    tableau = to_tableau(A, b, c)

    while can_be_improved(tableau):
        pivot_position = get_pivot_position(tableau)
        tableau = pivot_step(tableau, pivot_position)

    x_star = get_solution(tableau)
    f_star = c.T @ x_star
    return (f_star, x_star, True)

