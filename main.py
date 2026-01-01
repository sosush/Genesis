import random
import copy
import dummy_env as env  # <--- WHEN TEAMMATE 1 IS DONE, CHANGE THIS TO: import real_env as env

# ==========================================
# CONFIGURATION
# ==========================================
POPULATION_SIZE = 100
GENERATIONS = 20
TOURNAMENT_SIZE = 5
CROSSOVER_RATE = 0.8  # 80% chance to breed
MUTATION_RATE = 0.2   # 20% chance to mutate

# ==========================================
# HELPER: TREE MANIPULATION
# ==========================================

def get_node_list(node, parent=None, side=None):
    """
    Flattens the tree into a list of tuples: (Node, Parent, Side)
    Side: 'left' means this node is the left child of Parent.
    This makes it easy to pick a random node and swap it.
    """
    nodes = []
    # Add the current node info
    nodes.append({
        'node': node,
        'parent': parent,
        'side': side
    })
    
    # Recursively get children
    if hasattr(node, 'left') and node.left:
        nodes.extend(get_node_list(node.left, node, 'left'))
    if hasattr(node, 'right') and node.right:
        nodes.extend(get_node_list(node.right, node, 'right'))
        
    return nodes

def swap_node(target_info, new_subtree):
    """
    Takes the info from get_node_list and replaces the node with new_subtree.
    """
    parent = target_info['parent']
    side = target_info['side']
    
    # If parent is None, we are replacing the ROOT.
    if parent is None:
        return new_subtree
    
    # Otherwise, update the parent's pointer
    if side == 'left':
        parent.left = new_subtree
    elif side == 'right':
        parent.right = new_subtree
        
    return None # We didn't change the root, so return None

# ==========================================
# CORE GENETIC ALGORITHMS
# ==========================================

def tournament_selection(population):
    """
    Picks K random programs and returns the best one.
    """
    # 1. Pick random contestants
    contestants = random.sample(population, TOURNAMENT_SIZE)
    
    # 2. Sort them by fitness (Lower score is better)
    contestants.sort(key=lambda x: x['score'])
    
    # 3. Return the winner (the tree object)
    return contestants[0]['tree']

def crossover(parent1, parent2):
    """
    Swaps a random subtree from Parent 2 into Parent 1.
    """
    # 1. Create a Deep Copy of Parent 1 (The Base)
    child = copy.deepcopy(parent1)
    
    # 2. List all nodes in Child and Parent 2
    nodes_in_child = get_node_list(child)
    nodes_in_donor = get_node_list(parent2)
    
    # 3. Pick a random spot in the Child to replace
    target_spot = random.choice(nodes_in_child)
    
    # 4. Pick a random piece of logic from the Donor
    donor_spot = random.choice(nodes_in_donor)
    donor_subtree = copy.deepcopy(donor_spot['node']) # Copy needed!
    
    # 5. Perform the Swap
    new_root = swap_node(target_spot, donor_subtree)
    
    # If we replaced the root, return the new root. Otherwise return modified child.
    if new_root:
        return new_root
    return child

def mutation(tree):
    """
    Replaces a random node with a brand new random tree.
    """
    mutated_tree = copy.deepcopy(tree)
    
    # 1. Pick a random spot
    nodes = get_node_list(mutated_tree)
    target_spot = random.choice(nodes)
    
    # 2. Generate fresh random code (Alien DNA)
    new_random_subtree = env.generate_random_tree(depth=2)
    
    # 3. Swap
    new_root = swap_node(target_spot, new_random_subtree)
    
    if new_root:
        return new_root
    return mutated_tree

# ==========================================
# MAIN EVOLUTION LOOP
# ==========================================

def run_evolution():
    print("--- GENESIS AI STARTED ---")
    
    # 1. INITIALIZATION: Create Primordial Soup
    population = []
    for _ in range(POPULATION_SIZE):
        tree = env.generate_random_tree(depth=4)
        score = env.calculate_fitness(tree)
        population.append({'tree': tree, 'score': score})

    # 2. GENERATIONAL LOOP
    for gen in range(1, GENERATIONS + 1):
        
        # Sort population (Best First)
        population.sort(key=lambda x: x['score'])
        
        # Print Stats
        best_score = population[0]['score']
        print(f"Generation {gen}: Best Score = {best_score}")
        
        # Check for solution (Score 0 is perfect)
        if best_score == 0:
            print(f"\nSUCCESS! Solution found in Gen {gen}")
            print("Code:", population[0]['tree'])
            break
            
        # ELITISM: Keep the top 5% best programs automatically
        next_gen = population[:5] 
        
        # BREEDING LOOP
        while len(next_gen) < POPULATION_SIZE:
            # Select Parents
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            
            # Crossover
            if random.random() < CROSSOVER_RATE:
                child_tree = crossover(p1, p2)
            else:
                child_tree = copy.deepcopy(p1)
                
            # Mutation
            if random.random() < MUTATION_RATE:
                child_tree = mutation(child_tree)
                
            # Calculate Fitness for new child
            child_score = env.calculate_fitness(child_tree)
            next_gen.append({'tree': child_tree, 'score': child_score})
            
        population = next_gen

    print("\n--- EVOLUTION COMPLETE ---")
    print("Best Program Found:")
    print(population[0]['tree'])

if __name__ == "__main__":
    run_evolution()