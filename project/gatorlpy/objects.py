import numpy as np
from simplex import two_phase_simplex

class Expression:
    """Base class for variables, parameters, and operations in a computation graph."""
    def __init__(self, expression_type:str, parents:list=[]):
        self.expression_type = expression_type
        self.parents = parents
        
class Parameter(Expression):
    def __init__(self, array:np.ndarray, parents:list=[]):
        super().__init__("param", parents)
        self.array = array
        self.shape = array.shape
                
    @property
    def T(self):
        return Parameter(self.array.T)

    def __matmul__(self, other):
        """Overload @ operator for matrix multiplication"""
        if isinstance(other, Variable):
            return Operation(self, other, "param_matmul_var", [self, other])
        raise TypeError("Invalid type for matrix multiplication.")

    def __len__(self):
        return len(self.array)

    def __repr__(self):
        return f"({self.array})"

    def __add__(self, other):
        if isinstance(other, Parameter):
            if len(self) != len(other):
                raise ValueError("Parameters must have the same length for addition.")
            return Parameter(self.array + other.array, [self, other])
        elif isinstance(other, Variable):
            return Operation(self, other, "param_add_var", [self, other])

    def __neg__(self):
        return Parameter(-self.array, [self])
    

class Variable(Expression):
    """
    Represents a decision variable with a given shape.
    Supports matrix multiplication and constraint creation.
    """
    def __init__(self, shape:int, non_negative:bool=False, parents:list=[]):
        super().__init__("var", parents)
        self.array = np.zeros(shape)
        self.shape = shape
        self.non_negative = non_negative

    def __matmul__(self, other):
        """Overload @ operator for matrix multiplication"""
        if isinstance(other, Parameter):
            return Operation(self, other, "var_matmul_param", [self, other])
        raise TypeError("Invalid type for matrix multiplication.")
    
    def __add__(self, other):
        if isinstance(other, Variable):
            left_op = Operation(Parameter(np.eye(self.shape)), self, "param_matmul_var", [self])
            right_op = Operation(Parameter(np.eye(other.shape)), other, "param_matmul_var", [other])
            return left_op + right_op

    def __ge__(self, other):
        """Overload >= operator for constraints"""
        if isinstance(other, (int, float)):
            other = Parameter(np.full_like(self.array, other))
        
        if np.all(other.array==0):
            self.non_negative = True
            return None
        
        return Constraint(Operation(Parameter(np.eye(self.shape)), self, "param_matmul_var", [self]), other, "geq")

    def __le__(self, other):
        """Overload <= operator for constraints"""
        if isinstance(other, (int, float)):
            other = Parameter(np.full_like(self.array, other))
        return Constraint(Operation(Parameter(np.eye(self.shape)), self, "param_matmul_var", [self]), other, "leq")

    def __eq__(self, other):
        """Overload == operator for constraints"""
        if isinstance(other, (int, float)):
            other = Parameter(np.full_like(self.array, other))
        return Constraint(Operation(Parameter(np.eye(self.shape)), self, "param_matmul_var", [self]), other, "eq")
    
    def __repr__(self):
        return f"({self.array})"


class Operation(Expression):
    """
    Represents a matrix operation (e.g., A @ x).
    possible ops: 
    """
    def __init__(self, left:Expression, right:Expression, op:str, parents:list=[]):
        super().__init__("op", parents)
        self.left = left
        self.right = right
        self.op = op

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"
    
    def __ge__(self, other):
        """Overload >= operator for constraints"""
        return Constraint(self, other, "geq")

    def __le__(self, other):
        """Overload <= operator for constraints"""
        return Constraint(self, other, "leq")

    def __eq__(self, other):
        """Overload == operator for constraints"""
        return Constraint(self, other, "eq")
    
    def __neg__(self):
        return Operation(-self.left, self.right, "param_matmul_var", [self])
    
    def __add__(self, other):
        if isinstance(other, Variable):
            other = Operation(Parameter(np.eye(other.shape)), other, "param_matmul_var", [other])
        A_concat = Parameter(np.hstack(self.left.array, other.left.array), [self.left, other.left])






class Constraint(Expression):
    """
    Represents a linear constraint: A @ x <= b, >=, or ==.
    """
    def __init__(self, left:Operation, right:Expression, eq_type:str, parents:list=[]):
        super().__init__('constraint',parents)
        self.left = left
        self.right = right
        self.eq_type = eq_type
        self.constraint_type = self.left.op + "_" + self.eq_type + "_" + self.right.expression_type
        self.non_negativity = False


        if self.constraint_type == "param_matmul_var_leq_param":
            slack_vars = Variable(self.right.shape,True)
            return Constraint(self.left + slack_vars,self.right, "eq",[self])
        elif self.constraint_type == "param_matmul_var_geq_param":
            return Constraint(-self.left, -self.right, "leq", [self])
        



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
        A = self.constraints[0].left.left.array
        b = self.constraints[0].right.array
        c = self.objective.left.array
        f_star, x_star, feasible = two_phase_simplex(A, b, c)
        return f_star, x_star, feasible


