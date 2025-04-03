import numpy as np
from simplex import simplex, two_phase_simplex
class Parameter:
    def __init__(self,array:np.ndarray):
        self.array = array

    def __matmul__(self, other):
        """Overload @ operator for matrix multiplication"""
        if isinstance(other, (np.ndarray, Variable)):
            return Operation(self.array, other, "matmul")
        raise TypeError("Invalid type for matrix multiplication.")

    def __len__(self):
        return len(self.array)

    @property
    def T(self):
        return Parameter(self.array.T)

    def __repr__(self):
        return f"({self.array})"

    def __add__(self, other):
        if isinstance(other, Parameter):
            if len(self) != len(other):
                raise ValueError("Parameters must have the same length for addition.")
            return Parameter(self.array + other.array)

    def __neg__(self):
        return Parameter(-self.array)
    

class Variable:
    """
    Represents a decision variable with a given shape.
    Supports matrix multiplication and constraint creation.
    """
    def __init__(self, shape):
        self.array = np.zeros(shape)

    def __matmul__(self, other):
        """Overload @ operator for matrix multiplication"""
        if isinstance(other, (np.ndarray, Variable)):
            return Operation(other, self.array, "matmul")
        raise TypeError("Invalid type for matrix multiplication.")

    def __ge__(self, b):
        """Overload >= operator for constraints"""
        return Constraint(self, b, ">=")

    def __le__(self, b):
        """Overload <= operator for constraints"""
        return Constraint(self, b, "<=")

    def __eq__(self, b):
        """Overload == operator for constraints"""
        return Constraint(self, b, "==")
    
    def __repr__(self):
        return f"({self.array})"


class Operation:
    """
    Represents a matrix operation (e.g., A @ x).
    """
    def __init__(self, left, right, op):
        self.left = left
        self.right = right
        self.op = op

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"
    
    def __ge__(self, b):
        """Overload >= operator for constraints"""
        return Constraint(self, b, ">=")

    def __le__(self, b):
        """Overload <= operator for constraints"""
        return Constraint(self, b, "<=")

    def __eq__(self, b):
        """Overload == operator for constraints"""
        return Constraint(self, b, "==")


class Constraint:
    """
    Represents a linear constraint: A @ x <= b, >=, or ==.
    """
    def __init__(self, left:np.ndarray|Variable|Operation,right:np.ndarray|Variable,eq_type:str):
        self.left = left
        self.right = right
        self.eq_type = eq_type

    def __repr__(self):
        return f"{self.left} {self.eq_type} {self.right}"
    

            

class Problem:
    def __init__(self, problem_def:dict):
        if 'minimize' in problem_def and 'maximize' not in problem_def:
            self.objective_direction = 'minimize'
            self.objective = problem_def['minimize']
        elif 'maximize' in problem_def and 'minimize' not in problem_def:
            self.objective_direction = 'maximize'
            self.objective = problem_def['maximize']
        else:
            raise ValueError("The problem definition must contain either 'minimize' or 'maximize' as a key.")
        
        if 'subject to' in problem_def and 'constraints' not in problem_def:
            self.constraints = problem_def['subject to']
        elif 'constraints' in problem_def and 'subject to' not in problem_def:
            self.constraints = problem_def['constraints']
        else:
            raise ValueError("The constraint definition must contain either 'subject to' xor 'constraints' as a key.")

    def solve(self)->tuple[np.ndarray,bool]:
        A = self.constraints[0].left.left
        x0 = self.constraints[0].left.right.array
        b = self.constraints[0].right.array
        c = self.objective.left

        f_star, x_star, feasible = two_phase_simplex(A, b, c)
        return f_star, x_star, feasible


