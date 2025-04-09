import numpy as np
from simplex import two_phase_simplex

class Expression:
    """Base class for variables, parameters, and operations in a computation graph."""
    def __init__(self, expression_type:str, parents:list=[], shape:tuple=None):
        self.expression_type = expression_type
        self.parents = parents
        self.shape = shape

class Sum(Expression):
    def __init__(self, terms: list[Expression]):
        super().__init__("sum", terms)
        self.terms = terms
        shapes = [term.shape for term in self.terms]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("All terms in the Sum must have the same shape.")
        self.shape = shapes[0] if self.terms else None


    def __repr__(self):
        return " + ".join(str(term) for term in self.terms)
    
    def __add__(self, other):
        other = convert_to_expression(other,self.shape)
        if isinstance(other, Sum):
            return Sum(self.terms + other.terms)
        elif isinstance(other, (LinearOperation, Variable, Parameter)):
            return Sum(self.terms + [other])
        else:
            raise TypeError("Unsupported type for addition to Sum.")
        
    def get_terms_of_type(self, term_type: type) -> list:
        """Retrieve all terms of a specific type from the sum."""
        return [term for term in self.terms if isinstance(term, term_type)]
    
    def __neg__(self):
        neg_terms = [-term for term in self.terms]
        return Sum(neg_terms)
    
    def __sub__(self, other):
        return self.__add__(-other)
        
    def split_to_like_terms(self) -> tuple:
        assert self.is_sum_of_type((LinearOperation,Parameter))
        return Sum(self.get_terms_of_type(LinearOperation)), Sum(self.get_terms_of_type(Parameter))
    
    def is_sum_of_type(self, types:tuple[type]) -> bool:
        """Check if the Sum contains only Parameter or LinearOperation objects."""
        return all(isinstance(term, types) for term in self.terms)
    
    def combine_like_params(self):
        assert self.is_sum_of_type(Parameter)
        array = np.array([term.array for term in self.terms])
        return Parameter(np.sum(array,axis=0),[self])
    
    def combine_like_vars(self):
        pass # Will likely have to create some function that will take the vars like Ax + Bx and make it (A+B)x
    
    def get_sum_vars(self) -> list:
        assert self.is_sum_of_type(LinearOperation)
        return [term.variable for term in self.terms]
    
    def drop_terms_of_type(self, types:tuple[type]):
        filtered_terms = [term for term in self.terms if not isinstance(term, types)]
        return Sum(filtered_terms)


        

    
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
        other = convert_to_expression(other, self.shape)
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
        other = convert_to_expression(other,self.shape)
        return Sum([self, other])

    def __ge__(self, other):
        """Overload >= operator for constraints"""
        other = convert_to_expression(other,self.shape)       
        return Constraint(var_to_lin_op(self), other, "geq")

    def __le__(self, other):
        """Overload <= operator for constraints"""
        other = convert_to_expression(other,self.shape)
        return Constraint(var_to_lin_op(self), other, "leq")

    def __eq__(self, other):
        """Overload == operator for constraints"""
        other = convert_to_expression(other,self.shape)
        return Constraint(var_to_lin_op(self), other, "eq")
    
    def __repr__(self):
        return f"({self.array})"
    
    def __len__(self):
        return self.shape
    
    def __neg__(self):
        return var_to_lin_op(self,-1)
    
    def __sub__(self,other):
        return self.__add__(-other)
    
    def __hash__(self):
        return id(self)


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
        self.shape = self.parameter.shape[0]

    def __repr__(self):
        return f"({self.parameter} {self.op} {self.variable})"
    
    def __ge__(self, other):
        """Overload >= operator for constraints"""
        other = convert_to_expression(other,self.shape)
        return Constraint(self, other, "geq")

    def __le__(self, other):
        """Overload <= operator for constraints"""
        other = convert_to_expression(other,self.shape)
        return Constraint(self, other, "leq")

    def __eq__(self, other):
        """Overload == operator for constraints"""
        other = convert_to_expression(other,self.shape)
        return Constraint(self, other, "eq")
    
    def __neg__(self):
        return LinearOperation(-self.parameter, self.variable, "param_matmul_var", [self])
    
    def __add__(self, other:Expression):
        other = convert_to_expression(other,self.shape)
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
        left = convert_to_expression(left)
        right = convert_to_expression(right)
        self.left = left
        self.right = right
        self.eq_type = eq_type
        
    def __repr__(self):
        return f"{self.left} {self.eq_type} {self.right}"

    def any_constraint_to_sum_constraint(self):
        return Constraint(expression_to_linear_sum(self.left), expression_to_linear_sum(self.right),self.eq_type,[self])
    
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
        left_is_lin_op = isinstance(self.left, LinearOperation)
        if left_is_lin_op:
            lin_op_param_is_square = (self.left.parameter.array.shape[0] == self.left.parameter.array.shape[1])
        else:
            lin_op_param_is_square = False
        if lin_op_param_is_square:
            lin_op_param_is_diag = np.allclose(self.left.parameter.array, np.diag(np.full(self.left.parameter.shape[0], self.left.parameter.array[0, 0])))
        else:
            lin_op_param_is_diag = False
        right_is_param = isinstance(self.right, (Parameter))
        if right_is_param:
            right_is_zeros = np.all(self.right.array == 0)
        else:
            right_is_zeros = False
        self_is_geq = (self.eq_type == "geq")
        return all([lin_op_param_is_diag, right_is_zeros, self_is_geq])

    def is_linear_constraint(self) -> bool: # this is not correct
        return (isinstance(self.left, Sum) and isinstance(self.right, Parameter) and self.left.is_sum_of_type(LinearOperation))
    
    def fix_unbounded_linear_constraint(self, ub_b_map:dict):
        assert self.is_linear_constraint()
        lin_terms = []
        for lin_op_term in self.left.terms:
            var = lin_op_term.variable
            if var in ub_b_map.keys():
                var_plus = ub_b_map[var][0]
                var_neg = ub_b_map[var][1]
                lin_op_var_plus = LinearOperation(lin_op_term.parameter, var_plus, "param_matmul_var", [lin_op_term])
                lin_op_var_neg = LinearOperation(-lin_op_term.parameter, var_neg, "param_matmul_var", [lin_op_term])
                lin_terms.append(lin_op_var_plus)
                lin_terms.append(lin_op_var_neg)
            else:
                lin_terms.append(lin_op_term)
        
        return Constraint(Sum(lin_terms), self.right, self.eq_type, [self])
    
    def turn_linear_constraint_to_equality(self) -> tuple:
        assert self.is_linear_constraint()
        if self.eq_type == "eq":
            return self, []
        elif self.eq_type == "leq":
            slack_var = Variable(self.right.shape[0])
            return Constraint(self.left + var_to_lin_op(slack_var), self.right, "eq", [self]), [slack_var]
        elif self.eq_type == "geq":
            slack_var = Variable(self.right.shape[0])
            return Constraint(-self.left + var_to_lin_op(slack_var), -self.right, "eq", [self]), [slack_var]
        else:
            raise ValueError(f"Unsupported eq_type: {self.eq_type}")

    
    def get_variables(self) -> list:
        assert self.is_linear_constraint()
        return self.left.get_sum_vars()
    
    def linear_equality_to_matrix_equality(self, problem_var_map:dict, x_big):
        equality_matrix = np.zeros(shape=(self.right.shape[0], x_big.shape))
        param_list = []
        for term in self.left.terms:
            start_index, end_index = problem_var_map[term.variable]
            equality_matrix[:,start_index:end_index] += term.parameter.array
            param_list.append(term.parameter)
        A_constraint = Parameter(equality_matrix,param_list)
        big_lin_op = LinearOperation(A_constraint,x_big,"param_matmul_var",self.left.terms)
        return Constraint(big_lin_op, self.right, "eq", [self])









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
        self.unbounded_vars = []
        self.bounded_vars = []
        self.ub_b_map = {}
        self.problem_var_map = {}
        self.total_var_length = 0
        self.A_big = None
        self.b_big = None
        self.c_big = None
        self.x_big = None
        
        
    def to_slack_form(self):
        # each constraint should be Expression >=,<=,== Expression
        # where Expression is either Sum, Parameter, Variable, or LinearOperation
        # It should turn each constraint into a linear constraint Sum <=, >=, == Parameter
        # It should then make a pass to understand which variables are constrained or not
        # Turn the non_negative ones non_negative and then go to the unconstrained ones
        # then split them and constrain them
        # and then turn it into one stacked constraint Ax == b, x non-negative
        self.add_bounded_vars_to_list()
        self.remove_non_negativity_constraints()
        self.turn_all_constraints_to_linear_form()
        self.add_unbounded_vars_to_list()
        self.turn_all_constraints_to_equalities()
        self.add_unbounded_to_bounded_dict()
        self.fix_non_negative_variables()
        self.get_problem_var_map()
        self.get_final_variable()
        self.turn_all_equalities_to_matrix_equalities()
        self.standard_form_constraint_params()
        self.standard_form_objective()
        
        
    def remove_non_negativity_constraints(self):
        updated_constraints = []
        for constraint in self.constraints:  # Iterate over a copy of the list to allow removal
            if not constraint.is_non_negativity_constraint():
                updated_constraints.append(constraint)
        self.constraints = updated_constraints
    
    def add_bounded_vars_to_list(self):
        for constraint in self.constraints:  # Iterate over a copy of the list to allow removal
            if constraint.is_non_negativity_constraint():
                var = constraint.left.variable
                var.non_negative = True
                if var not in self.bounded_vars:
                    self.bounded_vars.append(var) 

    def add_unbounded_vars_to_list(self):
        for constraint in self.constraints:
            constraint_vars = constraint.get_variables()
            for var in constraint_vars:
                if var not in self.unbounded_vars + self.bounded_vars:
                    self.unbounded_vars.append(var)
            
    def fix_non_negative_variables(self):
        updated_constraints = []
        for constraint in self.constraints:
            assert not constraint.is_non_negativity_constraint()
            constraint = constraint.fix_unbounded_linear_constraint(self.ub_b_map)
            updated_constraints.append(constraint)
        self.constraints = updated_constraints

    def add_unbounded_to_bounded_dict(self):
        unbounded_bounded_map = {}
        for var in self.unbounded_vars:
            var_plus = Variable(var.shape, True, [var])
            self.bounded_vars.append(var_plus)
            var_neg = Variable(var.shape, True, [var])
            self.bounded_vars.append(var_neg)
            unbounded_bounded_map[var] = [var_plus, var_neg]
        self.ub_b_map = unbounded_bounded_map
    
    def turn_all_constraints_to_linear_form(self):
        updated_constraints = []
        for constraint in self.constraints:
            constraint = constraint.any_constraint_to_sum_constraint()
            constraint = constraint.sum_constraint_to_linear_constraint()
            updated_constraints.append(constraint)
        self.constraints = updated_constraints
    
    def turn_all_constraints_to_equalities(self):
        updated_constraints = []
        for constraint in self.constraints:
            constraint, slack_var = constraint.turn_linear_constraint_to_equality()
            if slack_var:
                self.bounded_vars.extend(slack_var)
            updated_constraints.append(constraint)
        self.constraints = updated_constraints

    def get_problem_var_map(self) -> int:
        start_index = 0
        end_index = 0
        for var in self.bounded_vars:
           end_index = var.shape + start_index
           self.problem_var_map[var] =  (start_index, end_index)
           start_index = end_index
        self.total_var_length = end_index

    def get_final_variable(self):
        self.x_big = Variable(sum([var.shape for var in self.bounded_vars]),True, self.bounded_vars)

    def turn_all_equalities_to_matrix_equalities(self):
        updated_constraints = []
        for constraint in self.constraints:
            constraint = constraint.linear_equality_to_matrix_equality(self.problem_var_map, self.x_big)
            updated_constraints.append(constraint)
        self.constraints = updated_constraints
        
    def standard_form_constraint_params(self):
        self.A_big = np.vstack([constraint.left.parameter.array for constraint in self.constraints])
        self.b_big = np.vstack([constraint.right.array.flatten() for constraint in self.constraints]).flatten()
            
    def standard_form_objective(self): 
        objective = expression_to_linear_sum(self.objective) #This could still be a problem since there could be feasibility problems
        objective = objective.drop_terms_of_type(Parameter)
        objective_const = Constraint(objective, Parameter(np.zeros(1)), "obj")
        self.c_big = objective_const.linear_equality_to_matrix_equality(self.problem_var_map, self.x_big).left.parameter.array.flatten()

    def solve(self)->tuple[np.ndarray,bool]:
        self.to_slack_form()
        f_star, x_star, feasible = two_phase_simplex(self.A_big, self.b_big, self.c_big)
        return f_star, x_star, feasible




def expression_to_linear_sum(exp:Expression) -> Sum:
    if isinstance(exp, (Parameter)):
        return Sum([exp])
    elif isinstance(exp, (Variable)):
        return Sum([var_to_lin_op(exp)])
    elif isinstance(exp, (LinearOperation)):
        return Sum([exp])
    elif isinstance(exp, (Sum)):
        sum_list = []
        for item in exp.terms:
            if isinstance(item, (Variable)):
                sum_list.append(var_to_lin_op(item))
            elif isinstance(item, (Parameter, LinearOperation)):
                sum_list.append(item)
            else:
                raise TypeError("Unsupported type for conversion to Linear Sum.")
        return Sum(sum_list)
    
def convert_to_expression(any, shape_hint:tuple|int=None) -> Expression:
    
    if isinstance(any, (float, int)):
        if shape_hint is None:
            raise ValueError("Must provide a shape_hint if the input is a float or int.")
        return Parameter(np.full(shape_hint, any))
    elif isinstance(any, (np.ndarray)):
        return Parameter(any)
    elif isinstance(any, (Expression)):
        return any
    else:
        raise TypeError("Unsupported type for conversion to Expression.")
    
def var_to_lin_op(var:Variable, coeff:int|float=1) -> LinearOperation:
    return LinearOperation(Parameter(coeff * np.eye(var.shape)), var, "param_matmul_var", [var])