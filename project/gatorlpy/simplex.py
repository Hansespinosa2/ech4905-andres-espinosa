import numpy as np
c = np.array([1, 1, 0, 0, 0])
A = np.array([
    [-1, 1, 1, 0, 0],
    [ 1, 0, 0, 1, 0],
    [ 0, 1, 0, 0, 1]
])
b = np.array([2, 4, 4])

def to_tableau(A:np.ndarray, b:np.ndarray, c:np.ndarray):
    xb = np.hstack([A, b[:, None]])  # Combine A and b (column-wise stack)
    z = np.hstack([c, [0]])  # Append a zero to c
    return np.vstack([xb, z])  # Combine xb and z (row-wise stack)

def can_be_improved(tableau:np.ndarray):
    z = tableau[-1]
    return np.any(z[:-1] < 0)

def get_pivot_position(tableau:np.ndarray):
    z = tableau[-1]
    column = np.argmin(z[:-1])  # Find the first positive entry in z
    
    ratios = []
    
    restrictions = np.array([
        np.inf if eq[column] <= 0 else eq[-1] / eq[column] 
        for eq in tableau[:-1]
    ])

    row = np.argmin(restrictions)
    return row, column

def pivot_step(tableau:np.ndarray, pivot_position:tuple[int]):
    new_tableau = tableau.copy()
    
    i, j = pivot_position
    pivot_value = tableau[i, j]
    new_tableau[i, :] = tableau[i, :] / pivot_value  # Divide row i by pivot value
    
    for eq_i, eq in enumerate(tableau):
        if eq_i != i:
            multiplier = new_tableau[i, :] * tableau[eq_i, j]
            new_tableau[eq_i, :] = tableau[eq_i, :] - multiplier
    
    return new_tableau

def is_basic(column:np.ndarray):
    return np.sum(column) == 1 and np.count_nonzero(column == 0) == len(column) - 1

def get_solution(tableau:np.ndarray):
    columns = tableau.T
    solutions = []
    for column in columns[:-1]:
        solution = 0
        if is_basic(column):
            one_index = np.where(column == 1)[0][0]  # Index of 1 in column
            solution = columns[-1, one_index]
        solutions.append(solution)
    return solutions

def simplex(A:np.ndarray, b:np.ndarray, c:np.ndarray):
    tableau = to_tableau(A, b, c)

    while can_be_improved(tableau):
        pivot_position = get_pivot_position(tableau)
        tableau = pivot_step(tableau, pivot_position)

    x_star = get_solution(tableau)
    f_star = c.T @ x_star
    return (f_star, x_star, True)

solution = simplex(A, b, c)
print('solution: ', solution)