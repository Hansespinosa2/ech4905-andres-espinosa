import numpy as np
from simplex import two_phase_simplex

class Expression:
    """Base class for variables, parameters, and operations in a computation graph."""
    def __init__(self, expression_type:str, parents:list=[], shape:tuple=None):
        self.expression_type = expression_type
        self.parents = parents
        self.shape = shape

    def convert_to_expression(self, any):
        if isinstance(any, (float, int)):
            return Parameter(np.full(self.shape,any))
        elif isinstance(any, (np.ndarray)):
            return Parameter(any)
        elif isinstance(any, (Expression)):
            return any
        else:
            raise TypeError("Unsupported type for conversion to Expression.")
        
    def expression_to_linear_sum(self):
        if isinstance(self, (Parameter)):
            return Sum([self])
        elif isinstance(self, (Variable)):
            return Sum([self.var_to_lin_op()])
        elif isinstance(self, (LinearOperation)):
            return Sum([self])
        elif isinstance(self, (Sum)):
            sum_list = []
            for item in self.terms:
                if isinstance(item, (Variable)):
                    sum_list.append(item.var_to_lin_op())
                elif isinstance(item, (Parameter, LinearOperation)):
                    sum_list.append(item)
                else:
                    raise TypeError("Unsupported type for conversion to Linear Sum.")
            return Sum(sum_list)

class Sum(Expression):
    def __init__(self, terms: list[Expression]):
        super().__init__("sum", terms)
        self.terms = terms

    def __repr__(self):
        return " + ".join(str(term) for term in self.terms)
    
    def __add__(self, other:Expression):
        other = self.convert_to_expression(other)
        if isinstance(other, Sum):
            return Sum(self.terms + other.terms)
        else: # if it is an LinearOperation, Variable, Parameter, 
            return Sum(self.terms + [other])
        
    def get_terms_of_type(self, term_type: type) -> list:
        """Retrieve all terms of a specific type from the sum."""
        return [term for term in self.terms if isinstance(term, term_type)]
    
    def __neg__(self):
        neg_terms = [-term for term in self.terms]
        return Sum(neg_terms)
    
    def __sub__(self, other):
        return self.__add__(-other)
        
    def split_to_like_terms(self):
        assert self.is_sum_of_type((LinearOperation,Parameter))
        return Sum(self.get_terms_of_type(LinearOperation)), Sum(self.get_terms_of_type(Parameter))
    
    def is_sum_of_type(self, types:tuple[type]) -> bool:
        """Check if the Sum contains only Parameter or LinearOperation objects."""
        return all(isinstance(term, types) for term in self.terms)
    
    def combine_like_params(self):
        assert self.is_sum_of_type(Parameter)
        array = np.array([array for array in self.terms])
        return Parameter(np.sum(array,axis=0),[self])
        

    
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
            return LinearOperation(self, other, "param_matmul_var", [self, other])
        raise TypeError("Invalid type for matrix multiplication.")

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return f"({self.array})"

    def __add__(self, other):
        other = self.convert_to_expression(other)
        if isinstance(other, Parameter):
            if len(self) != len(other):
                raise ValueError("Parameters must have the same length for addition.")
            return Sum([self,other])
        elif isinstance(other, Variable):
            return LinearOperation(self, other, "param_add_var", [self, other])

    def __neg__(self):
        return Parameter(-self.array, [self])
    
    def __sub__(self, other):
        return self.__add__(-other)
    
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

    def __matmul__(self, other): # I think this doesnt need to be implemented
        """Overload @ operator for matrix multiplication"""
        if isinstance(other, Parameter):
            return LinearOperation(self, other, "var_matmul_param", [self, other])
        raise TypeError("Invalid type for matrix multiplication.")
    
    def __add__(self, other:Expression):
        other = self.convert_to_expression(other)
        return Sum([self, other])

    def __ge__(self, other):
        """Overload >= operator for constraints"""
        other = self.convert_to_expression(other)       
        return Constraint(self.var_to_lin_op(), other, "geq")

    def __le__(self, other):
        """Overload <= operator for constraints"""
        other = self.convert_to_expression(other)
        return Constraint(self.var_to_lin_op(), other, "leq")

    def __eq__(self, other):
        """Overload == operator for constraints"""
        other = self.convert_to_expression(other)
        return Constraint(self.var_to_lin_op(), other, "eq")
    
    def __repr__(self):
        return f"({self.array})"
    
    def __len__(self):
        return self.shape[0]
    
    def __neg__(self):
        return self.var_to_lin_op(-1)
    
    def __sub__(self,other):
        return self.__add__(-other)
    
    def var_to_lin_op(self, coeff:int|float=1):
        return LinearOperation(Parameter(coeff * np.eye(self.shape)), self, "param_matmul_var", [self])


class LinearOperation(Expression):
    """
    Represents a matrix operation (e.g., A @ x).
    possible ops: 
    """
    def __init__(self, left:Parameter, right:Variable, op:str, parents:list=[]):
        super().__init__("op", parents)
        self.parameter = left
        self.variable = right
        self.op = op
        self.shape = right.shape

    def __repr__(self):
        return f"({self.parameter} {self.op} {self.variable})"
    
    def __ge__(self, other):
        """Overload >= operator for constraints"""
        other = self.convert_to_expression(other)
        return Constraint(self, other, "geq")

    def __le__(self, other):
        """Overload <= operator for constraints"""
        other = self.convert_to_expression(other)
        return Constraint(self, other, "leq")

    def __eq__(self, other):
        """Overload == operator for constraints"""
        other = self.convert_to_expression(other)
        return Constraint(self, other, "eq")
    
    def __neg__(self):
        return LinearOperation(-self.parameter, self.variable, "param_matmul_var", [self])
    
    def __add__(self, other:Expression):
        other = self.convert_to_expression(other)
        return Sum([self, other])
    
    def __len__(self):
        return self.parameter.shape[0]
    
    def __sub__(self, other:Expression):
        return self.__add__(-other)


class Constraint(Expression):
    """
    Represents a linear constraint: A @ x <= b, >=, or ==.
    """
    def __init__(self, left:Expression, right:Expression, eq_type:str, parents:list=[]):
        super().__init__('constraint', parents)
        left = self.convert_to_expression(left)
        right = self.convert_to_expression(right)
        self.left = left
        self.right = right
        self.eq_type = eq_type
        self.constraint_type = self.left.op + "_" + self.eq_type + "_" + self.right.expression_type

    def to_slack_form(self):
        if self.constraint_type == "param_matmul_var_leq_param":
            slack_vars = Variable(self.right.shape,True)
            return Constraint(self.left + slack_vars,self.right, "eq",[self])
        elif self.constraint_type == "param_matmul_var_geq_param":
            return Constraint(-self.left, -self.right, "leq", [self])
        
    def __repr__(self):
        return f"{self.left} {self.eq_type} {self.right}"

    def any_constraint_to_sum_constraint(self):
        return Constraint(self.left.expression_to_linear_sum(), self.right.expression_to_linear_sum(),self.eq_type,[self])
    
    def sum_constraint_to_linear_constraint(self):
        assert self.is_sum_constraint()
        left_sum = self.left
        right_sum = self.right 
        left_lin_ops, left_params = left_sum.split_to_like_terms()
        right_lin_ops, right_params = right_sum.split_to_like_terms()
        lin_op_sum = left_lin_ops - right_lin_ops
        param_sum = right_params - left_params
        param = param_sum.combine_like_params()

        return Constraint(lin_op_sum, param, self.eq_type, [self])

    def is_sum_constraint(self) -> bool:
        return (isinstance(self.left, Sum) and isinstance(self.right, Sum))
    
    def is_non_negativity_constraint(self) -> bool:
        left_is_var = isinstance(self.left, (Variable))
        right_is_param = isinstance(self.right, (Parameter))
        if right_is_param:
            right_is_zeros = np.all(self.right.array == 0)
        else:
            right_is_zeros = False
        self_is_geq = (self.eq_type == "geq")
        return all([left_is_var, right_is_param, right_is_zeros, self_is_geq])



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
        
        
    def to_slack_form(self):
        # each constraint should be Expression >=,<=,== Expression
        # where Expression is either Sum, Parameter, Variable, or LinearOperation
        # It should turn each constraint into a linear constraint Sum <=, >=, == Parameter
        # It should then make a pass to understand which variables are constrained or not
        # Turn the non_negative ones non_negative and then go to the unconstrained ones
        # then split them and constrain them
        # and then turn it into one stacked constraint Ax == b, x non-negative
        for constraint in self.constraints:
            if constraint.is_non_negativity_constraint():
                constraint.left.non_negative = True
                continue
            constraint = constraint.any_constraint_to_sum_constraint()
            constraint = constraint.sum_constraint_to_linear_constraint()



    def solve(self)->tuple[np.ndarray,bool]:
        A = self.constraints[0].left.parameter.array
        b = self.constraints[0].right.array
        c = self.objective.parameter.array
        f_star, x_star, feasible = two_phase_simplex(A, b, c)
        return f_star, x_star, feasible


