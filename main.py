import random
import copy
import dummy_env as env  # <--- TODO: CHANGE THIS TO: import real_env as env

# Check if matplotlib is installed for graphing (Optional but recommended)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Graphs will not be generated.")

# ==========================================
# CONFIGURATION
# ==========================================
POPULATION_SIZE = 100
GENERATIONS = 30
TOURNAMENT_SIZE = 5
BASE_CROSSOVER_RATE = 0.8
BASE_MUTATION_RATE = 0.1

# ==========================================
# HELPER: TREE ANALYSIS
# ==========================================

def count_nodes(node):
    """
    Recursively counts nodes to punish 'Bloat' (overly long code).
    """
    if node is None:
        return 0
    # Check if the node is an Operation (has children) or Terminal
    # We check attributes to be safe with different Node implementations
    count = 1
    if hasattr(node, 'left') and node.left:
        count += count_nodes(node.left)
    if hasattr(node, 'right') and node.right:
        count += count_nodes(node.right)
    return count

def get_node_list(node, parent=None, side=None):
    """
    Flattens the tree so we can pick random nodes easily.
    """
    nodes = []
    nodes.append({'node': node, 'parent': parent, 'side': side})
    
    if hasattr(node, 'left') and node.left:
        nodes.extend(get_node_list(node.left, node, 'left'))
    if hasattr(node, 'right') and node.right:
        nodes.extend(get_node_list(node.right, node, 'right'))
    return nodes

def swap_node(target_info, new_subtree):
    """
    Performs the transplant of logic.
    """
    parent = target_info['parent']
    side = target_info['side']
    
    if parent is None:
        return new_subtree # Replaced Root
    
    if side == 'left':
        parent.left = new_subtree
    elif side == 'right':
        parent.right = new_subtree
        
    return None

# ==========================================
# GENETIC OPERATORS
# ==========================================

def tournament_selection(population):
    """
    Picks K candidates and chooses the best one (lowest score).
    """
    candidates = random.sample(population, TOURNAMENT_SIZE)
    # Sort by Score first, then by Size (Parismony Pressure)
    candidates.sort(key=lambda x: (x['score'], x['size']))
    return candidates[0]['tree']

def crossover(parent1, parent2):
    child = copy.deepcopy(parent1)
    
    nodes_child = get_node_list(child)
    nodes_donor = get_node_list(parent2)
    
    # Safety check: if trees are empty/broken
    if not nodes_child or not nodes_donor:
        return child

    target = random.choice(nodes_child)
    donor = random.choice(nodes_donor)
    donor_subtree = copy.deepcopy(donor['node'])
    
    new_root = swap_node(target, donor_subtree)
    return new_root if new_root else child

def mutation(tree):
    mutated_tree = copy.deepcopy(tree)
    nodes = get_node_list(mutated_tree)
    
    if not nodes: 
        return mutated_tree

    target = random.choice(nodes)
    # Generate a small random subtree (Depth 2 ensures we don't just add single variables)
    new_subtree = env.generate_random_tree(depth=2)
    
    new_root = swap_node(target, new_subtree)
    return new_root if new_root else mutated_tree

# ==========================================
# VISUALIZATION
# ==========================================
def plot_stats(best_scores, avg_scores):
    if not HAS_MATPLOTLIB:
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(best_scores, label='Best Error (Lower is better)', color='red', linewidth=2)
    plt.plot(avg_scores, label='Average Population Error', color='blue', linestyle='--', alpha=0.6)
    
    plt.title('Genetic Algorithm Convergence')
    plt.xlabel('Generations')
    plt.ylabel('Fitness Score (Error)')
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================================
# MAIN EVOLUTION LOOP
# ==========================================

def run_evolution():
    print("--- GENESIS AI: ADVANCED MODE ---")
    
    # 1. INITIALIZATION
    population = []
    print(f"Initializing Population ({POPULATION_SIZE} programs)...")
    for _ in range(POPULATION_SIZE):
        tree = env.generate_random_tree(depth=4)
        score = env.calculate_fitness(tree)
        size = count_nodes(tree)
        population.append({'tree': tree, 'score': score, 'size': size})

    # Stats Tracking
    history_best = []
    history_avg = []
    
    # Adaptive Logic Tracking
    stagnation_counter = 0
    last_best_score = float('inf')
    current_mutation_rate = BASE_MUTATION_RATE

    # 2. GENERATIONAL LOOP
    for gen in range(1, GENERATIONS + 1):
        
        # Sort: Primary Key = Score (Ascending), Secondary Key = Size (Ascending)
        # This implements "Bloat Control" - prefer shorter code if scores are tied
        population.sort(key=lambda x: (x['score'], x['size']))
        
        best_program = population[0]
        avg_score = sum(p['score'] for p in population) / len(population)
        
        # Log Stats
        history_best.append(best_program['score'])
        history_avg.append(avg_score)
        
        print(f"Gen {gen}: Best Score = {best_program['score']} | Avg Score = {avg_score:.2f} | Size = {best_program['size']}")

        # CHECK SUCCESS
        if best_program['score'] == 0:
            print(f"\n>>> PERFECT SOLUTION FOUND IN GEN {gen} <<<")
            print("Code:", best_program['tree'])
            break

        # ADAPTIVE MUTATION LOGIC
        # If we haven't improved for 3 generations, panic and increase mutation
        if best_program['score'] == last_best_score:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            last_best_score = best_program['score']
            current_mutation_rate = BASE_MUTATION_RATE # Reset to normal

        if stagnation_counter >= 3:
            current_mutation_rate = 0.4 # Boost mutation to 40%
            print(f"   [!] Stagnation detected. Boosting mutation rate to {current_mutation_rate}")

        # PREPARE NEXT GENERATION
        next_gen = []
        
        # Elitism: Keep the top 5 programs (Survivors)
        next_gen.extend(population[:5])
        
        while len(next_gen) < POPULATION_SIZE:
            # Selection
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            
            # Crossover
            if random.random() < BASE_CROSSOVER_RATE:
                child_tree = crossover(p1, p2)
            else:
                child_tree = copy.deepcopy(p1)
            
            # Mutation (Dynamic Rate)
            if random.random() < current_mutation_rate:
                child_tree = mutation(child_tree)
            
            # Evaluate Child
            score = env.calculate_fitness(child_tree)
            size = count_nodes(child_tree)
            next_gen.append({'tree': child_tree, 'score': score, 'size': size})
            
        population = next_gen

    print("\n--- EVOLUTION COMPLETE ---")
    print("Final Best Code:")
    print(population[0]['tree'])
    
    # Plot results
    plot_stats(history_best, history_avg)

if __name__ == "__main__":
    run_evolution()