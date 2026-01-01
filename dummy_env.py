import random

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.val = None
    
    def evaluate(self, x):
        return 0

class Operation(Node):
    def __init__(self, left, right, op_char):
        super().__init__()
        self.left = left
        self.right = right
        self.op = op_char
    
    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"

class Terminal(Node):
    def __init__(self, value):
        super().__init__()
        self.val = value
    
    def __repr__(self):
        return str(self.val)

def generate_random_tree(depth=3):
    """Creates a fake random tree structure."""
    if depth == 0 or random.random() < 0.3:
        # Return a leaf (x or a number)
        return Terminal(random.choice(['x', 1, 2, 3, 5]))
    else:
        # Return an operator (+, -, *)
        op = random.choice(['+', '-', '*'])
        left = generate_random_tree(depth - 1)
        right = generate_random_tree(depth - 1)
        return Operation(left, right, op)

def calculate_fitness(tree):
    """
    Returns a FAKE error score. 
    Lower is better. 0 is perfect.
    In the real version, this will run the code against test cases.
    """

    score = random.randint(0, 100) 
    return score