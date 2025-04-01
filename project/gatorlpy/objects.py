class Variable:
    def __init__(self,shape):
        pass

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
            raise ValueError("The constraint definition must contain either 'subject to' or 'constraints' as a key.")
        


class Expression:
    def __init__(self, expression):
        if '==' in expression:
            self.equality_type = 'e'
            self.left_side, self.right_side = map(str.strip, expression.split('=='))
        elif '<=' in expression:
            self.equality_type = 'l'
            self.left_side, self.right_side = map(str.strip, expression.split('<='))
        elif '>=' in expression:
            self.equality_type = 'g'
            self.left_side, self.right_side = map(str.strip, expression.split('>='))
        else:
            raise ValueError("Expression must contain one of the following operators: '==', '<=', '>='")
