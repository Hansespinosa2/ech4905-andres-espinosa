import numpy as np
c = np.array([1, 1, 0, 0, 0])
A = np.array([
    [-1, 1, 1, 0, 0],
    [ 1, 0, 0, 1, 0],
    [ 0, 1, 0, 0, 1]
])
b = np.array([2, 4, 4])

def to_tableau(c, A, b):
    xb = np.hstack([A, b[:, None]])  # Combine A and b (column-wise stack)
    z = np.hstack([c, [0]])  # Append a zero to c
    return np.vstack([xb, z])  # Combine xb and z (row-wise stack)

def can_be_improved(tableau):
    z = tableau[-1]
    return np.any(z[:-1] > 0)

def get_pivot_position(tableau):
    z = tableau[-1]
    column = np.argmax(z[:-1] > 0)  # Find the first positive entry in z
    
    restrictions = np.array([
        np.inf if eq[column] <= 0 else eq[-1] / eq[column] 
        for eq in tableau[:-1]
    ])

    row = np.argmin(restrictions)
    return row, column

def pivot_step(tableau, pivot_position):
    new_tableau = tableau.copy()
    
    i, j = pivot_position
    pivot_value = tableau[i, j]
    new_tableau[i, :] = tableau[i, :] / pivot_value  # Divide row i by pivot value
    
    for eq_i, eq in enumerate(tableau):
        if eq_i != i:
            multiplier = new_tableau[i, :] * tableau[eq_i, j]
            new_tableau[eq_i, :] = tableau[eq_i, :] - multiplier
    
    return new_tableau

def is_basic(column):
    return np.sum(column) == 1 and np.count_nonzero(column == 0) == len(column) - 1

def get_solution(tableau):
    columns = tableau.T
    solutions = []
    for column in columns:
        solution = 0
        if is_basic(column):
            one_index = np.where(column == 1)[0][0]  # Index of 1 in column
            solution = columns[-1, one_index]
        solutions.append(solution)
    return solutions

def simplex(c, A, b):
    tableau = to_tableau(c, A, b)

    while can_be_improved(tableau):
        pivot_position = get_pivot_position(tableau)
        tableau = pivot_step(tableau, pivot_position)

    return get_solution(tableau)

solution = simplex(c, A, b)
print('solution: ', solution)