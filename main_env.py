import gp_engine as gp
import random

# Holds the current "Secret" formula object
HIDDEN_TRUTH_TREE = None
TRAINING_DATA = []

def generate_new_environment(difficulty_depth=3):
    """
    1. Invents a random formula (The Teacher).
    2. Generates the Answer Key (X, Y) data.
    """
    global HIDDEN_TRUTH_TREE, TRAINING_DATA
    
    # 1. The Teacher invents a rule
    # We use the same random generator to create the 'Truth'
    HIDDEN_TRUTH_TREE = gp.generate_random_tree(depth=difficulty_depth)
    
    # 2. Generate Data Points
    TRAINING_DATA = []
    # We test on integers from -5 to 5
    for x in range(-5, 6):
        try:
            target_y = HIDDEN_TRUTH_TREE.evaluate(x)
            TRAINING_DATA.append((x, target_y))
        except:
            # If the random formula creates a divide by zero, just skip that point
            pass
            
    # For debugging/grading, print what the secret is (User shouldn't see this!)
    print(f"\n[ORACLE] I have invented a secret formula.")
    print(f"[ORACLE] It looks like: {HIDDEN_TRUTH_TREE}")
    print(f"[ORACLE] Difficulty Depth: {difficulty_depth}")

def calculate_fitness(student_tree):
    """
    Compares the Student's logic against the Teacher's Data.
    """
    total_error = 0
    
    if not TRAINING_DATA:
        return 9999 # Error if data generation failed
        
    for x_input, expected_y in TRAINING_DATA:
        try:
            student_output = student_tree.evaluate(x_input)
            
            # Absolute Error
            error = abs(expected_y - student_output)
            
            # Cap huge errors
            if error > 10000: error = 10000
            
            total_error += error
            
        except Exception:
            return 10000 # Crash penalty
            
    return total_error