import random

# ==========================================
# 1. THE DNA (NODE CLASSES)
# ==========================================

class Node:
    """Base class for all nodes in the tree."""
    def __init__(self):
        self.left = None
        self.right = None
    
    def evaluate(self, x):
        """Abstract method to be overridden."""
        raise NotImplementedError

class Terminal(Node):
    """Leaf node: Represents a number (1, 5) or variable (x)."""
    def __init__(self, value):
        super().__init__()
        self.value = value

    def evaluate(self, x):
        # If the value is 'x', return the actual input number
        if self.value == 'x':
            return x
        # Otherwise, return the constant number
        return self.value

    def __repr__(self):
        return str(self.value)

class Operation(Node):
    """Internal node: Represents a math function (+, -, *)."""
    def __init__(self, left, right, op_char):
        super().__init__()
        self.left = left
        self.right = right
        self.op = op_char

    def evaluate(self, x):
        # RECURSION: We call evaluate() on children before doing math here.
        left_val = self.left.evaluate(x)
        right_val = self.right.evaluate(x)

        if self.op == '+': return left_val + right_val
        if self.op == '-': return left_val - right_val
        if self.op == '*': return left_val * right_val
        
        # Protected Division (Prevents crash if dividing by zero)
        if self.op == '/': 
            return 1 if right_val == 0 else left_val / right_val
        
        return 0

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"

# ==========================================
# 2. THE FACTORY (RANDOM TREE GENERATOR)
# ==========================================

def generate_random_tree(depth=4):
    """
    Recursively builds a random expression tree.
    Depth 0 = Must be a Terminal (Leaf).
    Depth > 0 = Can be an Operation or Terminal.
    """
    
    # If we reached max depth, OR we get a random coin flip, stop growing
    if depth == 0 or (random.random() < 0.3 and depth < 4):
        # Return a number (1-10) or 'x'
        choice = random.choice(['x', 'x', 1, 2, 3, 5, 10]) 
        return Terminal(choice)
    
    else:
        # Create an operator (+, -, *)
        op = random.choice(['+', '-', '*'])
        
        # Recursively build left and right children
        left_child = generate_random_tree(depth - 1)
        right_child = generate_random_tree(depth - 1)
        
        return Operation(left_child, right_child, op)