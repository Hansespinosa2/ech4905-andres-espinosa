from __future__ import annotations
import numpy as np
from simplex import two_phase_simplex

class Expression:
    """
    Represents a base class for variables, parameters, and operations in a computation graph.

    Attributes:
        expression_type (str): The type of the expression (e.g., variable, parameter, operation).
        parents (list): A list of parent Expression objects that this expression depends on.
                        Defaults to an empty list if not provided.
        shape (tuple): The shape of the expression, typically used for tensors or arrays.
                       Defaults to None if not specified.

    Args:
        expression_type (str): The type of the expression.
        parents (list, optional): A list of parent Expression objects. Defaults to None.
        shape (tuple, optional): The shape of the expression. Defaults to None.
    """
    def __init__(self, expression_type:str, parents:list=None, shape:tuple=None):
        self.expression_type = expression_type
        if parents is None:
            self.parents = []
        else:
            self.parents = parents
        self.shape = shape

class Sum(Expression):
    """
    Represents a summation of multiple Expression objects. The Sum class ensures
    that all terms have the same shape and provides methods for manipulating and
    analyzing the summation.
    Attributes:
        terms (list[Expression]): A list of Expression objects that make up the sum.
        shape (tuple or None): The shape of the terms in the sum, or None if there are no terms.
    Methods:
        __repr__():
            Returns a string representation of the Sum object as a summation of its terms.
        __add__(other):
            Adds another Expression or Sum to the current Sum. Returns a new Sum object.
        get_terms_of_type(term_type: type) -> list:
            Retrieves all terms of a specific type from the sum.
        __neg__():
            Negates all terms in the Sum and returns a new Sum object.
        __sub__(other):
            Subtracts another Expression or Sum from the current Sum. Returns a new Sum object.
        split_to_like_terms() -> tuple:
            Splits the Sum into two separate Sums: one containing LinearOperation terms
            and the other containing Parameter terms. Returns a tuple of these two Sums.
        is_sum_of_type(types: tuple[type]) -> bool:
            Checks if the Sum contains only terms of the specified types.
        combine_like_params():
            Combines all Parameter terms in the Sum into a single Parameter object.
            Returns the combined Parameter.
        combine_like_vars():
            Placeholder method for combining Variable terms in the Sum. Intended to
            simplify terms like Ax + Bx into (A+B)x.
        get_sum_vars() -> list:
            Retrieves the variables from all LinearOperation terms in the Sum.
        drop_terms_of_type(types: tuple[type]):
            Removes all terms of the specified types from the Sum and returns a new Sum object.
    """

    def __init__(self, terms:list[Expression]):
        super().__init__("sum", terms)
        self.terms = terms
        shapes = [term.shape for term in self.terms]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("All terms in the Sum must have the same shape.")
        self.shape = shapes[0] if self.terms else None


    def __repr__(self) -> str:
        return " + ".join(str(term) for term in self.terms)

    def __add__(self, other) -> Sum:
        other = convert_to_expression(other, self.shape)
        if isinstance(other, Sum):
            return Sum(self.terms + other.terms)
        if isinstance(other, (LinearOperation, Variable, Parameter)):
            return Sum(self.terms + [other])
        raise TypeError("Unsupported type for addition to Sum.")

    def get_terms_of_type(self, term_type:type) -> list:
        """Retrieve all terms of a specific type from the sum."""
        return [term for term in self.terms if isinstance(term, term_type)]

    def __neg__(self) -> Sum:
        neg_terms = [-term for term in self.terms]
        return Sum(neg_terms)

    def __sub__(self, other) -> Sum:
        return self.__add__(-other)

    def split_to_like_terms(self) -> tuple:
        """
        Splits the current object into two separate sums based on the type of terms.
        Returns:
            tuple: A tuple containing two `Sum` objects:
                - The first `Sum` contains all terms of type `LinearOperation`.
                - The second `Sum` contains all terms of type `Parameter`.
        Raises:
            AssertionError: If the current object is not a sum of terms of type 
                            `LinearOperation` or `Parameter`.
        """
        if not self.is_sum_of_type((LinearOperation, Parameter)):
            raise TypeError("The Sum must contain only LinearOperation or Parameter terms.")
        return Sum(self.get_terms_of_type(LinearOperation)), Sum(self.get_terms_of_type(Parameter))

    def is_sum_of_type(self, types:tuple[type]) -> bool:
        """
        Checks if all terms in the current object are instances of the specified types.
        Args:
            types (tuple[type]): A tuple of types to check against.
        Returns:
            bool: True if all terms are instances of the specified types, False otherwise.
        """
        return all(isinstance(term, types) for term in self.terms)

    def combine_like_params(self) -> Parameter:
        """
        Combines terms of the current Sum object that are of type `Parameter` 
        into a single `Parameter` object by summing their arrays.
        Returns:
            Parameter: A new `Parameter` object with the combined array 
            and a reference to the current Sum object as its parent.
        Raises:
            TypeError: If the Sum object contains terms that are not of type `Parameter`.
        """

        if not self.is_sum_of_type((Parameter,)):
            raise TypeError("The Sum must contain only Parameter terms to combine them.")
        array = np.array([term.array for term in self.terms])
        return Parameter(np.sum(array, axis=0), [self])

    def combine_like_vars(self):
        pass # Will likely have to create some function that will take the vars like Ax + Bx and make it (A+B)x

    def get_sum_vars(self) -> list:
        """
        Retrieve the list of variables from the terms of the sum.
        Returns:
            list: A list of variables extracted from the terms of the sum.
        Raises:
            TypeError: If the sum does not exclusively contain terms of type `LinearOperation`.
        """

        if not self.is_sum_of_type((LinearOperation,)):
            raise TypeError("The Sum must contain only LinearOperation terms to retrieve variables.")
        return [term.variable for term in self.terms]

    def drop_terms_of_type(self, types:tuple[type]) -> Sum:
        """
        Removes terms from the current Sum object that are of the specified types.
        Args:
            types (tuple[type]): A tuple of types to filter out from the terms.
        Returns:
            Sum: A new Sum object containing only the terms that are not instances
            of the specified types.
        """
        filtered_terms = [term for term in self.terms if not isinstance(term, types)]
        return Sum(filtered_terms)


class Parameter(Expression):
    """
    Represents a parameter in a linear programming expression. 
    This class extends the `Expression` class and provides functionality for 
    matrix operations, addition, subtraction, and transposition.
    Attributes:
        array (np.ndarray): The numerical array representing the parameter's values.
        shape (tuple): The shape of the parameter's array.
    Methods:
        T:
            Returns the transpose of the parameter as a new `Parameter` object.
        __matmul__(other):
            Overloads the `@` operator for matrix multiplication. 
            Supports multiplication with `Variable` objects.
            Raises:
                TypeError: If the other operand is not of a valid type.
        __len__():
            Returns the number of rows in the parameter's array.
        __repr__():
            Returns a string representation of the parameter's array.
        __add__(other):
            Overloads the `+` operator for addition. 
            Supports addition with other `Parameter` or `Variable` objects.
            Raises:
                ValueError: If the lengths of the parameters do not match for addition.
        __neg__():
            Returns the negation of the parameter as a new `Parameter` object.
        __sub__(other):
            Overloads the `-` operator for subtraction. 
            Equivalent to adding the negation of the other operand.
    """
    def __init__(self, array:np.ndarray, parents:list=None):
        super().__init__("param", parents)
        self.array = array
        self.shape = array.shape

    @property
    def T(self) -> Parameter:
        """Returns the transpose of the parameter as a new Parameter object."""
        return Parameter(self.array.T)

    def __matmul__(self, other:Variable) -> LinearOperation:
        """Overload @ operator for matrix multiplication"""
        if isinstance(other, Variable):
            return LinearOperation(self, other, "lin_op", [self, other])
        raise TypeError("Invalid type for matrix multiplication.")

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return f"({self.array})"

    def __add__(self, other) -> Sum:
        other = convert_to_expression(other, self.shape)
        if isinstance(other, Parameter):
            if len(self) != len(other):
                raise ValueError("Parameters must have the same length for addition.")
            return Sum([self,other])
        if isinstance(other, Variable):
            return Sum([self, other])
        raise TypeError("Unsupported type for addition with Parameter.")

    def __neg__(self) -> Parameter:
        return Parameter(-self.array, [self])

    def __sub__(self, other) -> Sum:
        return self.__add__(-other)

class Variable(Expression):
    """
    Represents a variable in a linear programming model.
    Attributes:
        shape (int): The size or dimensionality of the variable.
        non_negative (bool): Indicates whether the variable is constrained to be non-negative.
        array (numpy.ndarray): An array initialized with zeros representing the variable's values.
        parents (list): A list of parent expressions associated with this variable.
    Methods:
        __matmul__(other):
            Overloads the @ operator for matrix multiplication with a Parameter object.
            Raises:
                TypeError: If the other operand is not of type Parameter.
        __add__(other):
            Overloads the + operator to add this variable to another Expression.
            Converts the other operand to an Expression if necessary.
        __ge__(other):
            Overloads the >= operator to create a "greater than or equal to" constraint.
            Converts the other operand to an Expression if necessary.
        __le__(other):
            Overloads the <= operator to create a "less than or equal to" constraint.
            Converts the other operand to an Expression if necessary.
        __eq__(other):
            Overloads the == operator to create an equality constraint.
            Converts the other operand to an Expression if necessary.
        __repr__():
            Returns a string representation of the variable's array.
        __len__():
            Returns the shape (size) of the variable.
        __neg__():
            Overloads the unary - operator to negate the variable.
        __sub__(other):
            Overloads the - operator to subtract another Expression from this variable.
        __hash__():
            Returns a unique hash value for the variable based on its ID.
    """

    def __init__(self, shape:tuple|int, non_negative:bool=False, parents:list=None): #TODO shape should be tuple not int
        super().__init__("var", parents)
        self.array = np.zeros(shape)
        if isinstance(shape, int):
            self.shape = (shape,)
        elif isinstance(shape, tuple):
            if len(shape) > 1:
                raise ValueError("Each Variable must be a vector")
            self.shape = shape
        else:
            raise TypeError("The shape input of each Variable must be an integer or singular tuple")
        self.non_negative = non_negative

    def __matmul__(self, other): # I think this doesnt need to be implemented
        """Overload @ operator for matrix multiplication"""
        if isinstance(other, Parameter):
            return LinearOperation(self, other, "var_matmul_param", [self, other])
        raise TypeError("Invalid type for matrix multiplication.")

    def __add__(self, other) -> Sum:
        other = convert_to_expression(other,self.shape)
        return Sum([self, other])

    def __ge__(self, other) -> Constraint:
        """Overload >= operator for constraints"""
        other = convert_to_expression(other,self.shape)
        return Constraint(var_to_lin_op(self), other, "geq")

    def __le__(self, other) -> Constraint:
        """Overload <= operator for constraints"""
        other = convert_to_expression(other,self.shape)
        return Constraint(var_to_lin_op(self), other, "leq")

    def __eq__(self, other) -> Constraint:
        """Overload == operator for constraints"""
        other = convert_to_expression(other,self.shape)
        return Constraint(var_to_lin_op(self), other, "eq")

    def __repr__(self) -> str:
        return f"({self.array})"

    def __len__(self) -> int:
        return self.shape[0]

    def __neg__(self) -> LinearOperation:
        return var_to_lin_op(self,-1)

    def __sub__(self,other) -> Sum:
        return self.__add__(-other)

    def __hash__(self) -> int:
        return id(self)


class LinearOperation(Expression):
    """
    Represents a linear operation between a parameter and a variable, with support for 
    various mathematical operations and constraints.
    Attributes:
        parameter (Parameter): The left-hand side parameter of the operation.
        variable (Variable): The right-hand side variable of the operation.
        op (str): The operation type (e.g., "lin_op").
        shape (int): The shape of the parameter (assumes 1D shape for now).
    Methods:
        __repr__():
            Returns a string representation of the linear operation.
        __ge__(other):
            Overloads the >= operator to create a "greater than or equal to" constraint.
        __le__(other):
            Overloads the <= operator to create a "less than or equal to" constraint.
        __eq__(other):
            Overloads the == operator to create an equality constraint.
        __neg__():
            Returns the negation of the linear operation.
        __add__(other):
            Overloads the + operator to add another expression to the linear operation.
        __len__():
            Returns the length of the parameter (number of rows in its shape).
        __sub__(other):
            Overloads the - operator to subtract another expression from the linear operation.
    """

    def __init__(self, left:Parameter, right:Variable, op:str, parents:list=None): #TODO: get rid of op, it is always linear
        super().__init__("op", parents)
        self.parameter = left
        self.variable = right
        self.op = op
        self.shape = self.parameter.shape[0]

    def __repr__(self) -> str:
        return f"({self.parameter} {self.op} {self.variable})"

    def __ge__(self, other) -> Constraint:
        """Overload >= operator for constraints"""
        other = convert_to_expression(other,self.shape)
        return Constraint(self, other, "geq")

    def __le__(self, other) -> Constraint:
        """Overload <= operator for constraints"""
        other = convert_to_expression(other,self.shape)
        return Constraint(self, other, "leq")

    def __eq__(self, other) -> Constraint:
        """Overload == operator for constraints"""
        other = convert_to_expression(other,self.shape)
        return Constraint(self, other, "eq")

    def __neg__(self) -> LinearOperation:
        return LinearOperation(-self.parameter, self.variable, "lin_op", [self])

    def __add__(self, other:Expression) -> Sum:
        other = convert_to_expression(other,self.shape)
        return Sum([self, other])

    def __len__(self) -> int:
        return self.parameter.shape[0]

    def __sub__(self, other:Expression) -> Sum:
        return self.__add__(-other)

class Constraint(Expression):
    """
    A class representing a mathematical constraint in the form of an expression.
    This class is used to define constraints in optimization problems, where a constraint
    is represented as a relationship between two expressions (left-hand side and right-hand side)
    with a specified equality type (e.g., "eq", "leq", "geq").
    Attributes:
        left (Expression): The left-hand side of the constraint.
        right (Expression): The right-hand side of the constraint.
        eq_type (str): The type of equality for the constraint. Supported types are:
            - "eq": Equality constraint.
            - "leq": Less-than-or-equal-to constraint.
            - "geq": Greater-than-or-equal-to constraint.
        parents (list, optional): A list of parent constraints or expressions that this
            constraint depends on.
    Methods:
        __repr__() -> str:
            Returns a string representation of the constraint.
        any_constraint_to_sum_constraint() -> Constraint:
            Converts any constraint into a sum constraint by transforming both sides
            into linear sum expressions while preserving the equality type.
        sum_constraint_to_linear_constraint() -> Constraint:
            Converts a sum constraint into a linear constraint by combining like terms
            and simplifying the expressions. Raises a TypeError if the constraint is not
            a sum constraint.
        is_sum_constraint() -> bool:
            Checks if the constraint is a sum constraint (both sides are instances of Sum).
        is_non_negativity_constraint() -> bool:
            Checks if the constraint is a non-negativity constraint, where the left-hand
            side is a diagonal linear operation, the right-hand side is zero, and the
            equality type is "geq".
        is_linear_constraint() -> bool:
            Checks if the constraint is a linear constraint, where the left-hand side is
            a sum of linear operations and the right-hand side is a parameter.
        fix_unbounded_linear_constraint(ub_b_map: dict):
            Fixes unbounded variables in a linear constraint by introducing new variables
            for upper and lower bounds.
        turn_linear_constraint_to_equality() -> tuple:
            Converts a linear constraint into an equality constraint by introducing slack
            variables if necessary. Returns the new constraint and a list of slack variables.
        get_variables() -> list:
            Retrieves the variables involved in a linear constraint.
        linear_equality_to_matrix_equality(problem_var_map: dict, x_big):
            Converts a linear equality constraint into a matrix equality representation
            suitable for optimization solvers.
    """

    def __init__(self, left:Expression, right:Expression, eq_type:str, parents:list=None):
        super().__init__('constraint', parents)
        left = convert_to_expression(left)
        right = convert_to_expression(right)
        self.left = left
        self.right = right
        self.eq_type = eq_type

    def __repr__(self) -> str:
        return f"{self.left} {self.eq_type} {self.right}"

    def any_constraint_to_sum_constraint(self) -> Constraint:
        """
        Converts any constraint into a sum constraint.
        This method transforms the left-hand side and right-hand side expressions
        of the constraint into Sum expressions. It retains the equality type
        and includes the original constraint as part of the resulting constraint.
        Returns:
            Constraint: A new constraint object with both sides converted to linear
            Sum expressions, preserving the equality type and referencing the original
            constraint.
        """
        left_linear_sum = expression_to_linear_sum(self.left)
        right_linear_sum = expression_to_linear_sum(self.right)
        return Constraint(left_linear_sum, right_linear_sum, self.eq_type, [self])

    def sum_constraint_to_linear_constraint(self) -> Constraint:
        """
        Converts a sum constraint into a linear constraint.
        This method transforms a constraint that is represented as a sum of terms
        into a linear constraint by separating the linear operators and parameters
        on both sides of the equation, combining like terms, and constructing a 
        new linear constraint.
        Returns:
            Constraint: A new linear constraint derived from the sum constraint.
        Raises:
            TypeError: If the constraint is not a sum constraint.
        """

        if not self.is_sum_constraint():
            raise TypeError("The constraint must be a sum constraint to convert it to linear")
        left_sum = self.left
        right_sum = self.right
        left_lin_ops, left_params = left_sum.split_to_like_terms()
        right_lin_ops, right_params = right_sum.split_to_like_terms()
        lin_op_sum = left_lin_ops - right_lin_ops
        param_sum = right_params - left_params
        param = param_sum.combine_like_params()

        return Constraint(lin_op_sum, param, self.eq_type, [self])

    def is_sum_constraint(self) -> bool:
        """
        Determines if the current constraint is a sum constraint.
        A sum constraint is defined as a constraint where both the left-hand 
        side and the right-hand side are instances of the `Sum` class.
        Returns:
            bool: True if both sides of the constraint are instances of `Sum`, 
                  False otherwise.
        """

        return (isinstance(self.left, Sum) and isinstance(self.right, Sum))

    def is_non_negativity_constraint(self) -> bool:
        """
        Determines if the current constraint represents a non-negativity constraint.
        A non-negativity constraint is defined as:
        - The left-hand side is a linear operation with a square parameter matrix.
        - The parameter matrix of the linear operation is diagonal with identical diagonal entries.
        - The right-hand side is a parameter with all zero entries.
        - The constraint type is "greater than or equal to" (geq).
        Returns:
            bool: True if the constraint satisfies all the conditions 
            of a non-negativity constraint, False otherwise.
        """

        left_is_lin_op = isinstance(self.left, LinearOperation)
        if left_is_lin_op:
            left_param_m_rows = self.left.parameter.array.shape[0]
            left_param_n_cols = self.left.parameter.array.shape[1]
            lin_op_param_is_square = (left_param_m_rows == left_param_n_cols)
        else:
            lin_op_param_is_square = False
        if lin_op_param_is_square:
            left_param = self.left.parameter
            left_param_diagonalized = np.diag(np.full(left_param.shape[0], left_param.array[0, 0]))
            lin_op_param_is_diag = np.allclose(left_param.array, left_param_diagonalized)
        else:
            lin_op_param_is_diag = False
        right_is_param = isinstance(self.right, (Parameter))
        if right_is_param:
            right_is_zeros = np.all(self.right.array == 0)
        else:
            right_is_zeros = False
        self_is_geq = (self.eq_type == "geq")
        return all([lin_op_param_is_diag, right_is_zeros, self_is_geq])

    def is_linear_constraint(self) -> bool:
        """
        Determines if the current constraint is a linear constraint.
        A constraint is considered linear if:
        - The left-hand side (`self.left`) is an instance of `Sum`.
        - The right-hand side (`self.right`) is an instance of `Parameter`.
        - The left-hand side is a sum composed entirely of `LinearOperation` types.
        Returns:
            bool: True if the constraint is linear, False otherwise.
        """
        if isinstance(self.left, Sum):
            left_is_lin_op_sum = self.left.is_sum_of_type(LinearOperation)
        else:
            left_is_lin_op_sum = False
        right_is_param = isinstance(self.right, Parameter)
        return (left_is_lin_op_sum and right_is_param)

    def fix_unbounded_linear_constraint(self, ub_b_map:dict):
        """
        Adjusts a linear constraint to handle unbounded variables by replacing them
        with bounded auxiliary variables.
        This method modifies the linear constraint by substituting unbounded variables
        with two auxiliary variables (positive and negative) that are bounded. The 
        substitution is based on the mapping provided in `ub_b_map`.
        Args:
            ub_b_map (dict): A dictionary mapping unbounded variables to a tuple of 
                             two auxiliary variables (positive and negative). 
                             The structure is {variable: (var_plus, var_neg)}.
        Returns:
            Constraint: A new Constraint object where unbounded variables have been 
                        replaced with their corresponding bounded auxiliary variables.
        Raises:
            AssertionError: If the constraint is not a linear constraint.
        """

        if not self.is_linear_constraint():
            raise TypeError("Must be a linear constraint to fix unbounded variables.")

        lin_terms = []
        for lin_op_term in self.left.terms:
            param = lin_op_term.parameter
            var = lin_op_term.variable
            if var in ub_b_map.keys():
                var_plus = ub_b_map[var][0]
                var_neg = ub_b_map[var][1]
                lin_op_var_plus = LinearOperation(param, var_plus, "lin_op", [lin_op_term])
                lin_op_var_neg = LinearOperation(-param, var_neg, "lin_op", [lin_op_term])
                lin_terms.append(lin_op_var_plus)
                lin_terms.append(lin_op_var_neg)
            else:
                lin_terms.append(lin_op_term)

        return Constraint(Sum(lin_terms), self.right, self.eq_type, [self])

    def turn_linear_constraint_to_equality(self) -> tuple[Constraint,list]:
        """
        Converts a linear constraint into an equality constraint by introducing slack variables.
        Returns:
            tuple: A tuple containing:
                - A new `Constraint` object representing the equality constraint.
                - A list of slack variables introduced during the conversion.
        Raises:
            TypeError: If the constraint is not a linear constraint.
            ValueError: If the `eq_type` of the constraint is unsupported.
        Notes:
            - If the constraint is already an equality constraint (`eq_type == "eq"`), 
              it is returned unchanged along with an empty list of slack variables.
            - For a less-than-or-equal-to constraint (`eq_type == "leq"`), a slack variable 
              is added to convert it into an equality constraint.
            - For a greater-than-or-equal-to constraint (`eq_type == "geq"`), a slack variable 
              is added after negating the constraint to convert it into an equality constraint.
        """

        if not self.is_linear_constraint():
            raise TypeError("Must be a linear constraint to turn to equality.")
        if self.eq_type == "eq":
            return self, []
        if self.eq_type == "leq":
            slack_var = Variable(self.right.shape[0])
            slack_lin_op = var_to_lin_op(slack_var)
            return Constraint(self.left + slack_lin_op, self.right, "eq", [self]), [slack_var]
        if self.eq_type == "geq":
            slack_var = Variable(self.right.shape[0])
            slack_lin_op = var_to_lin_op(slack_var)
            return Constraint(-self.left + slack_lin_op, -self.right, "eq", [self]), [slack_var]
        raise ValueError(f"Unsupported eq_type: {self.eq_type}")


    def get_variables(self) -> list:
        """
        Retrieves the list of variables involved in the linear constraint.
        Returns:
            list: A list of variables present in the left-hand side of the linear constraint.
        Raises:
            TypeError: If the constraint is not a linear constraint.
        """

        if not self.is_linear_constraint():
            raise TypeError("Must be a linear constraint to get vars.")
        return self.left.get_sum_vars()

    def linear_equality_to_matrix_equality(self, problem_var_map:dict, x_big:Variable):
        """
        Converts a linear equality constraint into a matrix equality representation.
        This method transforms a linear equality constraint, represented by terms 
        on the left-hand side and a right-hand side, into a matrix equality form 
        suitable for optimization problems. It constructs a matrix representation 
        of the equality constraint and returns it as a `Constraint` object.
        Args:
            problem_var_map (dict): A mapping of variables to their corresponding 
                index ranges in the larger variable vector `x_big`. The keys are 
                variable names, and the values are tuples of the form 
                (start_index, end_index).
            x_big (Variable): The larger variable vector that includes all variables 
                in the optimization problem.
        Returns:
            Constraint: A `Constraint` object representing the equality constraint 
            in matrix form. This includes the linear operation, the right-hand side, 
            and metadata about the constraint type.
        Raises:
            KeyError: If a variable in `self.left.terms` is not found in 
                `problem_var_map`.
        Notes:
            - The method assumes that `self.left.terms` contains terms with 
              `variable` and `parameter` attributes.
            - The `Parameter` and `LinearOperation` classes are used to construct 
              the matrix representation of the equality constraint.
        """

        equality_matrix = np.zeros(shape=(self.right.shape[0], x_big.shape[0]))
        param_list = []
        for term in self.left.terms:
            start_index, end_index = problem_var_map[term.variable]
            equality_matrix[:,start_index:end_index] += term.parameter.array
            param_list.append(term.parameter)
        A_constraint = Parameter(equality_matrix,param_list)
        big_lin_op = LinearOperation(A_constraint,x_big, "lin_op",self.left.terms)
        return Constraint(big_lin_op, self.right, "eq", [self])


class Problem:
    """
    Class Problem
    Represents a linear programming problem and provides methods to convert it into 
    a standard form suitable for solving using the simplex algorithm.
    Attributes:
        objective_direction (str): Direction of optimization, either 'minimize' or 'maximize'.
        objective (Expression): The objective function of the problem.
        constraints (list): List of constraints for the problem.
        unbounded_vars (list): List of variables that are unbounded.
        bounded_vars (list): List of variables that are bounded (non-negative).
        ub_b_map (dict): Mapping of unbounded variables to their corresponding bounded variables.
        problem_var_map (dict): Mapping of variables to their indices in the final variable vector.
        total_var_length (int): Total length of the variable vector.
        A_big (np.ndarray): Coefficient matrix for the constraints in standard form.
        b_big (np.ndarray): Right-hand side vector for the constraints in standard form.
        c_big (np.ndarray): Coefficient vector for the objective function in standard form.
        x_big (Variable): Combined variable vector in standard form.
    Methods:
        __init__(problem_def: dict):
            Initializes the Problem instance with the given problem definition.
        to_slack_form():
            Converts the problem into slack form by transforming constraints and variables.
        remove_non_negativity_constraints():
            Removes non-negativity constraints from the list of constraints.
        add_bounded_vars_to_list():
            Adds bounded (non-negative) variables to the bounded_vars list.
        add_unbounded_vars_to_list():
            Adds unbounded variables to the unbounded_vars list.
        fix_non_negative_variables():
            Fixes unbounded variables by replacing them with bounded variables.
        add_unbounded_to_bounded_dict():
            Maps unbounded variables to their corresponding bounded variables.
        turn_all_constraints_to_linear_form():
            Converts all constraints into linear form.
        turn_all_constraints_to_equalities():
            Converts all constraints into equalities by introducing slack variables.
        get_problem_var_map() -> int:
            Generates a mapping of variables to their indices in the final variable vector.
        get_final_variable():
            Combines all bounded variables into a single variable vector.
        turn_all_equalities_to_matrix_equalities():
            Converts all equality constraints into matrix equalities.
        standard_form_constraint_params():
            Constructs the standard form constraint parameters (A_big and b_big).
        standard_form_objective():
            Converts the objective function into standard form (c_big).
        to_min_problem():
            Converts the problem into a minimization problem if it is a maximization problem.
        solve() -> tuple[np.ndarray, bool]:
            Solves the linear programming problem using the two-phase simplex method.
            Returns the optimal value, the optimal solution vector, and a feasibility flag.
    """

    def __init__(self, problem_def:dict):
        if 'minimize' in problem_def and 'maximize' not in problem_def:
            self.objective_direction = 'minimize'
            self.objective = problem_def['minimize']
        elif 'maximize' in problem_def and 'minimize' not in problem_def:
            self.objective_direction = 'maximize'
            self.objective = problem_def['maximize']
        else:
            raise ValueError("Problem must contain either 'minimize' or 'maximize' as a key.")

        if 'subject to' in problem_def and 'constraints' not in problem_def:
            self.constraints = problem_def['subject to']
        elif 'constraints' in problem_def and 'subject to' not in problem_def:
            self.constraints = problem_def['constraints']
        else:
            raise ValueError("Problem must contain either 'subject to' xor 'constraints' as a key.")
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
        """
        Converts the current linear programming problem into slack form.
        Slack form is a representation of a linear programming problem where all 
        constraints are expressed as equalities, and all variables are non-negative. 
        This method performs the following steps to achieve this transformation:
        1. Adds bounded variables to an internal list for tracking.
        2. Removes non-negativity constraints from the problem.
        3. Converts all constraints into linear form.
        4. Identifies and adds unbounded variables to an internal list.
        5. Converts all constraints into equalities by introducing slack variables.
        6. Maps unbounded variables to bounded variables for consistency.
        7. Ensures all non-negative variables are properly handled.
        8. Generates a mapping of problem variables for internal representation.
        9. Finalizes the variable representation for the problem.
        10. Converts all equality constraints into matrix equality form (Ax = b).
        11. Standardizes the constraint parameters for consistency.
        12. Standardizes the objective function for compatibility with slack form.
        This method prepares the problem for further processing in algorithms that 
        require the problem to be in slack form, such as the simplex method.
        """

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
        """
        Removes non-negativity constraints from the list of constraints.
        This method iterates through the current list of constraints and filters
        out any constraints that are identified as non-negativity constraints.
        The remaining constraints are updated in the `self.constraints` attribute.
        Non-negativity constraints are determined using the 
        `is_non_negativity_constraint` method of each constraint object.
        Returns:
            None
        """

        updated_constraints = []
        for constraint in self.constraints:  # Iterate over a copy of the list to allow removal
            if not constraint.is_non_negativity_constraint():
                updated_constraints.append(constraint)
        self.constraints = updated_constraints

    def add_bounded_vars_to_list(self):
        """
        Adds variables with non-negativity constraints to the list of bounded variables.
        This method iterates through the constraints of the object, identifies those
        that are non-negativity constraints, and marks the associated variable as 
        non-negative. If the variable is not already in the list of bounded variables, 
        it is appended to the list.
        Returns:
            None
        """

        for constraint in self.constraints:  # Iterate over a copy of the list to allow removal
            if constraint.is_non_negativity_constraint():
                var = constraint.left.variable
                var.non_negative = True
                if var not in self.bounded_vars:
                    self.bounded_vars.append(var)

    def add_unbounded_vars_to_list(self):
        """
        Adds variables from the constraints to the list of unbounded variables.
        This method iterates through all the constraints and retrieves their variables.
        If a variable is not already present in the combined list of unbounded and 
        bounded variables, it is added to the list of unbounded variables.
        Returns:
            None
        """

        for constraint in self.constraints:
            constraint_vars = constraint.get_variables()
            for var in constraint_vars:
                if var not in self.unbounded_vars + self.bounded_vars:
                    self.unbounded_vars.append(var)
            
    def fix_non_negative_variables(self):
        """
        Adjusts the constraints of the linear programming problem to handle 
        non-negative variables by fixing unbounded linear constraints.
        This method iterates through the list of constraints, ensuring that 
        none of them are non-negativity constraints. It then applies a fix 
        to unbounded linear constraints using the `fix_unbounded_linear_constraint` 
        method and updates the constraints list accordingly.
        Raises:
            AssertionError: If any constraint is a non-negativity constraint.
        Attributes:
            self.constraints (list): A list of constraints in the linear 
                programming problem.
            self.ub_b_map (dict): A mapping used to fix unbounded linear constraints.
        """

        updated_constraints = []
        for constraint in self.constraints:
            assert not constraint.is_non_negativity_constraint()
            constraint = constraint.fix_unbounded_linear_constraint(self.ub_b_map)
            updated_constraints.append(constraint)
        self.constraints = updated_constraints

    def add_unbounded_to_bounded_dict(self):
        """
        Converts unbounded variables into bounded variables and updates the mapping.
        This method iterates over the list of unbounded variables (`self.unbounded_vars`) 
        and creates two new bounded variables for each unbounded variable: one representing 
        the positive part (`var_plus`) and one representing the negative part (`var_neg`). 
        These new bounded variables are added to the `self.bounded_vars` list. Additionally, 
        a mapping from the original unbounded variable to the two new bounded variables is 
        created and stored in the `self.ub_b_map` dictionary.
        Attributes:
            self.unbounded_vars (list): A list of unbounded variables to be converted.
            self.bounded_vars (list): A list where the newly created bounded variables are appended.
            self.ub_b_map (dict): A dictionary mapping each unbounded variable to its corresponding 
                                  list of two bounded variables (`var_plus` and `var_neg`).
        Raises:
            AttributeError: If `self.unbounded_vars` or `self.bounded_vars` is not defined.
        """

        unbounded_bounded_map = {}
        for var in self.unbounded_vars:
            var_plus = Variable(var.shape[0], True, [var])
            self.bounded_vars.append(var_plus)
            var_neg = Variable(var.shape[0], True, [var])
            self.bounded_vars.append(var_neg)
            unbounded_bounded_map[var] = [var_plus, var_neg]
        self.ub_b_map = unbounded_bounded_map

    def turn_all_constraints_to_linear_form(self):
        """
        Converts all constraints in the current object to their linear form.
        This method iterates through the list of constraints, applying two 
        transformations to each constraint:
        1. Converts any type of constraint into a sum constraint using 
           `any_constraint_to_sum_constraint`.
        2. Converts the resulting sum constraint into a linear constraint using 
           `sum_constraint_to_linear_constraint`.
        The updated constraints are then stored back in the `self.constraints` attribute.
        """

        updated_constraints = []
        for constraint in self.constraints:
            constraint = constraint.any_constraint_to_sum_constraint()
            constraint = constraint.sum_constraint_to_linear_constraint()
            updated_constraints.append(constraint)
        self.constraints = updated_constraints

    def turn_all_constraints_to_equalities(self):
        """
        Converts all constraints in the linear programming problem to equalities.
        This method iterates through the list of constraints and transforms each 
        constraint into an equality by introducing slack or surplus variables 
        where necessary. The slack or surplus variables are added to the list of 
        bounded variables. The updated constraints are then stored back in the 
        constraints list.
        Returns:
            None
        """

        updated_constraints = []
        for constraint in self.constraints:
            constraint, slack_var = constraint.turn_linear_constraint_to_equality()
            if slack_var:
                self.bounded_vars.extend(slack_var)
            updated_constraints.append(constraint)
        self.constraints = updated_constraints

    def get_problem_var_map(self) -> int:
        """
        Maps each variable in `self.bounded_vars` to its corresponding index range 
        within the problem's variable space and calculates the total variable length.
        This method iterates through the variables in `self.bounded_vars`, determines 
        their shape, and assigns a tuple `(start_index, end_index)` to each variable 
        in `self.problem_var_map`. The `start_index` and `end_index` represent the 
        range of indices occupied by the variable in the problem's variable space. 
        The `start_index` is updated for the next variable, and the total length of 
        all variables is stored in `self.total_var_length`.
        Returns:
            int: The total length of all variables in the problem.
        """

        start_index = 0
        end_index = 0
        for var in self.bounded_vars:
           end_index = var.shape[0] + start_index
           self.problem_var_map[var] =  (start_index, end_index)
           start_index = end_index
        self.total_var_length = end_index

    def get_final_variable(self):
        """
        Constructs the final variable by combining all bounded variables.
        This method calculates the total size of all bounded variables, creates a 
        new variable with the combined size, and stores it in `self.x_big`.
        Attributes:
            self.bounded_vars (list): A list of variables, each having a `shape` attribute 
                                      that defines its size.
        Returns:
            None
        """
        final_var_shape = sum(var.shape[0] for var in self.bounded_vars)
        self.x_big = Variable(final_var_shape, True, self.bounded_vars)

    def turn_all_equalities_to_matrix_equalities(self):
        """
        Converts all linear equality constraints in the problem to matrix equality constraints.
        This method iterates through the list of constraints in the problem, 
        transforms each linear equality constraint into a matrix equality 
        representation using the `linear_equality_to_matrix_equality` method, 
        and updates the constraints list with the transformed constraints.
        Attributes:
            self.constraints (list): A list of constraints in the problem.
            self.problem_var_map (dict): A mapping of problem variables.
            self.x_big (any): A parameter used in the transformation process.
        Returns:
            None: The method updates the `self.constraints` attribute in place.
        """

        updated_constraints = []
        for constraint in self.constraints:
            constraint = constraint.linear_equality_to_matrix_equality(self.problem_var_map, self.x_big)
            updated_constraints.append(constraint)
        self.constraints = updated_constraints
        
    def standard_form_constraint_params(self):
        """
        Converts the constraints of a linear programming problem into standard form.
        This method processes the constraints of the problem by stacking the left-hand 
        side parameters into a single matrix `A_big` and the right-hand side values into 
        a single vector `b_big`. If any right-hand side value in `b_big` is negative, 
        the corresponding row in `A_big` and the value in `b_big` are multiplied by -1 
        to ensure all constraints are in standard form.
        Attributes:
            A_big (numpy.ndarray): A matrix containing the stacked left-hand side 
                parameters of all constraints.
            b_big (numpy.ndarray): A vector containing the stacked right-hand side 
                values of all constraints.
        Notes:
            - The method assumes that `self.constraints` is a list of constraint objects, 
              each having `left.parameter.array` and `right.array` attributes.
            - The constraints are modified such that all right-hand side values in `b_big` 
              are non-negative.
        """

        self.A_big = np.vstack([constraint.left.parameter.array for constraint in self.constraints])
        reshaped_b_params = [constraint.right.array.flatten().reshape(-1,1) for constraint in self.constraints]
        self.b_big = np.vstack(reshaped_b_params).flatten()
        for i in range(self.A_big.shape[0]):
            if self.b_big[i] < 0:
                self.A_big[i, :] *= -1
                self.b_big[i] *= -1


    def standard_form_objective(self):
        """
        Converts the objective function of the problem into its standard form.
        This method processes the objective function by:
        1. Converting it into a linear sum representation using `expression_to_linear_sum`.
        2. Dropping terms of type `Parameter` from the objective.
        3. Creating a `Constraint` object for the objective function.
        4. Transforming the objective constraint into a matrix equality representation
           and extracting the coefficient vector `c_big`.
        Attributes:
            self.objective: The original objective expression of the problem.
            self.problem_var_map: A mapping of problem variables to their indices.
            self.x_big: A vector representing the decision variables.
        Raises:
            ValueError: If there are feasibility issues with the objective expression.
        Notes:
            - The method assumes that the objective can be represented as a linear sum.
            - The resulting `c_big` vector is stored as an attribute of the object.
        """

        objective = expression_to_linear_sum(self.objective) #This could still be a problem since there could be feasibility problems
        objective = objective.drop_terms_of_type(Parameter)
        objective_const = Constraint(objective, Parameter(np.zeros(1)), "obj")
        matrix_objective = objective_const.linear_equality_to_matrix_equality(self.problem_var_map, self.x_big)
        self.c_big = matrix_objective.left.parameter.array.flatten()

    def to_min_problem(self):
        """
        Converts the linear programming problem to a minimization problem.
        If the current objective direction is "maximize", this method negates
        the objective function coefficients and the constant term `c_big` to
        transform the problem into a minimization problem.
        Attributes:
            objective_direction (str): The direction of the objective function,
                                       either "maximize" or "minimize".
            objective (numpy.ndarray or similar): The coefficients of the 
                                                  objective function.
            c_big (float): A constant term used in the objective function.
        Side Effects:
            Modifies the `objective` and `c_big` attributes of the instance
            if the problem is currently a maximization problem.
        """

        if self.objective_direction == "maximize":
            self.objective = -self.objective
            self.c_big = -self.c_big


    def solve(self)->tuple[np.ndarray,bool]:
        """
        Solves the linear programming problem using the two-phase simplex method.
        This method first converts the problem into slack form and ensures it is a 
        minimization problem. It then applies the two-phase simplex algorithm to 
        find the optimal solution.
        Returns:
            tuple[np.ndarray, bool]: A tuple containing:
                - f_star (np.ndarray): The optimal value of the objective function.
                - x_star (np.ndarray): The optimal solution vector.
                - feasible (bool): A boolean indicating whether the problem is feasible.
        """

        self.to_slack_form()
        self.to_min_problem()
        f_star, x_star, feasible = two_phase_simplex(self.A_big, self.b_big, self.c_big)
        return f_star, x_star, feasible




def expression_to_linear_sum(exp:Expression) -> Sum:
    """
    Converts a given expression into a linear sum representation.
    This function takes an expression and ensures it is represented as a `Sum` object,
    which is a collection of terms that can include `Parameter`, `Variable`, or 
    `LinearOperation` objects. If the input expression is already a `Sum`, it processes 
    its terms to ensure all variables are converted to linear operations.
    Args:
        exp (Expression): The input expression to be converted. It can be of type 
                            `Parameter`, `Variable`, `LinearOperation`, or `Sum`.
    Returns:
        Sum: A `Sum` object representing the linear sum of the input expression.
    Raises:
        TypeError: If the input expression contains unsupported types that cannot 
                    be converted to a linear sum.
    """

    if isinstance(exp, (Parameter)):
        return Sum([exp])
    if isinstance(exp, (Variable)):
        return Sum([var_to_lin_op(exp)])
    if isinstance(exp, (LinearOperation)):
        return Sum([exp])
    if isinstance(exp, (Sum)):
        sum_list = []
        for item in exp.terms:
            if isinstance(item, (Variable)):
                sum_list.append(var_to_lin_op(item))
            elif isinstance(item, (Parameter, LinearOperation)):
                sum_list.append(item)
            else:
                raise TypeError("Unsupported type for conversion to Linear Sum.")
        return Sum(sum_list)
    raise TypeError("Unsupported type for conversion to Linear Sum.")

def convert_to_expression(any_obj:float|int|np.ndarray|Expression,
                          shape_hint:tuple|int=None) -> Expression:
    """
    Converts the input into an Expression object.
    Parameters:
    -----------
    any : float, int, np.ndarray, or Expression
        The input to be converted into an Expression. Supported types are:
        - float or int: A scalar value. Requires `shape_hint` to specify the shape.
        - np.ndarray: A NumPy array that will be directly converted into an Expression.
        - Expression: An existing Expression object, which will be returned as-is.
    shape_hint : tuple or int, optional
        The shape of the resulting Expression if the input is a scalar (float or int).
        Must be provided when `any` is a scalar. Ignored for other input types.
    Returns:
        Expression object of the input
    --------
    Expression
        The converted Expression object.
    Raises:
    -------
    ValueError
        If `any` is a scalar (float or int) and `shape_hint` is not provided.
    TypeError
        If `any` is of an unsupported type.
    Examples:
    ---------
    >>> convert_to_expression(5, shape_hint=(2, 2))
    Parameter()
    >>> convert_to_expression(np.array([1, 2, 3]))
    Parameter()
    >>> expr = Expression(...)
    >>> convert_to_expression(expr)
    Expression()
    """

    if isinstance(any_obj, (float, int)):
        if shape_hint is None:
            raise ValueError("Must provide a shape_hint if the input is a float or int.")
        return Parameter(np.full(shape_hint, any_obj))
    if isinstance(any_obj, (np.ndarray)):
        return Parameter(any_obj)
    if isinstance(any_obj, (Expression)):
        return any_obj
    raise TypeError("Unsupported type for conversion to Expression.")

def var_to_lin_op(var:Variable, coeff:int|float=1) -> LinearOperation:
    """
    Converts a variable into a linear operation by applying a scalar coefficient.
    Args:
        var (Variable): The variable to be converted into a linear operation.
        coeff (int | float, optional): The scalar coefficient to multiply with the identity matrix. 
            Defaults to 1.
    Returns:
        LinearOperation: A linear operation object representing the parameterized 
        matrix multiplication of the variable.
    """

    return LinearOperation(Parameter(coeff * np.eye(var.shape[0])), var, "lin_op", [var])
