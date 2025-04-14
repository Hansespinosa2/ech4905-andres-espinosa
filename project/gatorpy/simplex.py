import numpy as np

FLOAT_POINT_ROUND = 8
EPSILON = 10**-FLOAT_POINT_ROUND

def zero_if_close_to_zero(x:np.ndarray) -> np.ndarray:
    """ Replaces values in the input array that are close to zero with an exact zero. """
    return np.where(np.abs(x) < EPSILON, 0, x)

def pivot(tableau:np.ndarray, row_index:int, col_index:int) -> np.ndarray:
    """
    Perform a pivot operation on a simplex tableau.
    The pivot operation modifies the tableau to make the element at the 
    specified row and column (pivot element) equal to 1, and all other 
    elements in the pivot column equal to 0. This is a key step in the 
    simplex algorithm for solving linear programming problems.
    Args:
        tableau (np.ndarray): The simplex tableau, represented as a 2D NumPy array.
        row_index (int): The row index of the pivot element.
        col_index (int): The column index of the pivot element.
    Returns:
        np.ndarray: The updated simplex tableau after the pivot operation.
    Notes:
        - The pivot element is located at tableau[row_index, col_index].
        - The row containing the pivot element is scaled so that the pivot 
            element becomes 1.
        - All other rows are updated to make the elements in the pivot column 
            equal to 0, while preserving the tableau's structure.
    """

    tableau[row_index, :] /= tableau[row_index, col_index]
    for i in range(tableau.shape[0]):
        if i != row_index:
            tableau[i,:] -= tableau[i, col_index] * tableau[row_index,:]
    return tableau

def find_entering_variable(tableau:np.ndarray,rule:str) -> int:
    """
    Determines the entering variable for the simplex algorithm based on the specified rule.
    Parameters:
    tableau (np.ndarray): The simplex tableau, where the last row contains the coefficients
                            of the objective function.
    rule (str): The rule to determine the entering variable. Currently supports 'blands'.
    Returns:
    int: The index of the entering variable column if found, or None if no entering variable exists
            or the rule is not supported.
    Notes:
    - If the 'blands' rule is used, 
        the function selects the first column with a negative coefficient
        in the objective function row (last row of the tableau).
    - EPSILON is used as a threshold to account for numerical precision issues.
    """

    if rule == 'blands':
        for col_index in range(tableau.shape[1] -1):
            if tableau[-1, col_index] < -EPSILON:
                return col_index
        return None
    raise ValueError(f"Unsupported rule selected: {rule}")

def find_leaving_variable(tableau:np.ndarray, entering_col_index:int) -> int:
    """
    Determines the leaving variable in the simplex algorithm by calculating the 
    minimum ratio of the right-hand side to the pivot column values.
    Args:
        tableau (np.ndarray): The simplex tableau, where the last column represents 
                                the right-hand side values and the other columns represent 
                                the coefficients of the constraints.
        entering_col_index (int): The index of the entering variable's column in the tableau.
    Returns:
        int: The index of the row corresponding to the leaving variable. Returns None if no 
                valid leaving variable is found (e.g., no positive pivot column values).
    """

    ratios = []
    for row_index in range(tableau.shape[0] -1):
        if tableau[row_index, entering_col_index] > EPSILON:
            ratio = tableau[row_index, -1] / tableau[row_index, entering_col_index]
            ratios.append((ratio, row_index))
    return min(ratios, default=(None,None))[1] if ratios else None


def simplex_phase_1(A:np.ndarray, b:np.ndarray):
    """
    Perform Phase 1 of the Simplex algorithm to find a feasible solution 
    for the given linear programming problem.
    This function attempts to find a basic feasible solution (BFS) for the problem 
    defined by the constraints `Ax = b` 
    and `x >= 0`. If the problem is infeasible, it returns an indication of infeasibility.
    Parameters:
    -----------
    A : np.ndarray
        The constraint matrix of shape (m, n), 
        where `m` is the number of constraints and `n` is the number of variables.
    b : np.ndarray
        The right-hand side vector of shape (m,), representing the constraints.
    Returns:
    --------
    tuple:
        - np.ndarray or None: The reduced constraint matrix of shape (m, n) 
          if a feasible solution is found, otherwise None.
        - np.ndarray or None: The feasible solution vector of shape (m,) 
          if a feasible solution is found, otherwise None.
        - list or None: The list of basic variable indices corresponding to the feasible solution, 
          or None if infeasible.
        - bool: A boolean indicating whether the problem is feasible (True) or infeasible (False).
    Raises:
    -------
    ValueError:
        If the problem is determined to be infeasible during the pivoting process.
    Notes:
    ------
    - This implementation uses Bland's rule to prevent cycling during pivot selection.
    - The function assumes that the input matrix `A` and vector `b` 
      are properly defined and consistent.
    - The auxiliary problem is solved by introducing artificial variables and minimizing their sum.
    """
    # TODO: Fix aux variables not leaving basis bug.
    m, n = A.shape
    A_aux = np.hstack((A, np.eye(m)))
    c_aux = np.hstack((np.zeros(n), np.ones(m)))
    tableau = np.hstack((A_aux, b.reshape(-1,1)))
    tableau = np.vstack((tableau, np.hstack((-c_aux, [0]))))
    basis = list(range(n,n+m))

    tableau[-1, :-1] = -np.sum(tableau[:-1, :-1], axis=0)
    tableau[-1,-1] = -np.sum(tableau[:-1,-1])

    while True:
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
    
    # GPT SOLUTION FOR REMOVING ARTIFICIAL VARIABLES TODO: This still fails the problem but at least runs
    # Remove artificial variables from basis 
    for i in range(len(basis)):
        if basis[i] >= n:
            # Look for a non-artificial variable to pivot in
            for j in range(n):
                if j not in basis and abs(tableau[i, j]) > EPSILON:
                    tableau = pivot(tableau, i, j)
                    basis[i] = j
                    break
            else:
                # If no such variable exists, zero row: leave it as is
                pass

    return tableau[:m, :n], tableau[:-1, -1], basis[:m]

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
    """
    Solves a linear programming problem using the two-phase simplex method.
    This function first attempts to find a feasible solution using Phase 1 of the simplex method.
    If a feasible solution exists, it proceeds to Phase 2 to optimize the objective function.
    Parameters:
    -----------
    A : np.ndarray
        The constraint matrix of the linear programming problem.
    b : np.ndarray
        The right-hand side vector of the constraints.
    c : np.ndarray
        The coefficients of the objective function to be minimized.
    Returns:
    --------
    tuple[np.ndarray | bool]
        A tuple containing:
        - f_star (np.ndarray): The optimal value of the objective function, if feasible.
        - x_star (np.ndarray): The optimal solution vector, if feasible.
        - feasible (bool): A boolean indicating whether the problem is feasible.
    Notes:
    ------
    - If the problem is infeasible, the function returns (None, None, False).
    - The input `c` should represent the coefficients of the objective function to be minimized.
    - The function assumes that the input matrices and vectors are properly formatted 
      and compatible.
    """

    A, b, basis = simplex_phase_1(A,b)
    if A is None:
        return None, None, False # Problem is infeasible
    f_star, x_star, feasible = simplex_phase_2(A,b,-c,basis)
    return f_star, x_star, feasible
