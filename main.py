import random
import copy
import gp_engine as gp       # The DNA/Tree Factory
import main_env as problem   # The Oracle/Teacher (Infinite Generator)

# Check if matplotlib is installed for graphing
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Graphs will not be generated.")

# ==========================================
# CONFIGURATION
# ==========================================
POPULATION_SIZE = 150   
GENERATIONS = 50        
TOURNAMENT_SIZE = 7
BASE_CROSSOVER_RATE = 0.8
BASE_MUTATION_RATE = 0.15

# ==========================================
# HELPER: TREE ANALYSIS
# ==========================================

def count_nodes(node):
    """Recursively counts nodes to punish 'Bloat'."""
    if node is None: return 0
    count = 1
    if hasattr(node, 'child'): 
        count += count_nodes(node.child)
    elif hasattr(node, 'left'): 
        count += count_nodes(node.left)
        count += count_nodes(node.right)
    return count

def get_node_list(node, parent=None, side=None):
    """Flattens the tree so we can pick random nodes easily."""
    nodes = [{'node': node, 'parent': parent, 'side': side}]
    if hasattr(node, 'child'): 
        nodes.extend(get_node_list(node.child, node, 'child'))
    elif hasattr(node, 'left'): 
        if node.left: nodes.extend(get_node_list(node.left, node, 'left'))
        if node.right: nodes.extend(get_node_list(node.right, node, 'right'))
    return nodes

def swap_node(target_info, new_subtree):
    """Performs the transplant of logic."""
    parent = target_info['parent']
    side = target_info['side']
    if parent is None: return new_subtree 
    if side == 'left': parent.left = new_subtree
    elif side == 'right': parent.right = new_subtree
    elif side == 'child': parent.child = new_subtree 
    return None

# ==========================================
# GENETIC OPERATORS
# ==========================================

def tournament_selection(population):
    candidates = random.sample(population, TOURNAMENT_SIZE)
    candidates.sort(key=lambda x: (x['score'], x['size']))
    return candidates[0]['tree']

def crossover(parent1, parent2):
    child = copy.deepcopy(parent1)
    nodes_child = get_node_list(child)
    nodes_donor = get_node_list(parent2)
    if not nodes_child or not nodes_donor: return child

    target = random.choice(nodes_child)
    donor = random.choice(nodes_donor)
    donor_subtree = copy.deepcopy(donor['node'])
    
    new_root = swap_node(target, donor_subtree)
    return new_root if new_root else child

def mutation(tree):
    mutated_tree = copy.deepcopy(tree)
    nodes = get_node_list(mutated_tree)
    if not nodes: return mutated_tree

    target = random.choice(nodes)
    new_subtree = gp.generate_random_tree(depth=2)
    
    new_root = swap_node(target, new_subtree)
    return new_root if new_root else mutated_tree

# ==========================================
# REPORTING & VISUALIZATION
# ==========================================

def print_final_report(session_data):
    """Generates a neat ASCII table of the session."""
    print("\n" + "="*100)
    print(f"{'GENESIS AI: SESSION SUMMARY':^100}")
    print("="*100)
    print(f"{'LVL':<4} | {'TARGET (ORACLE FORMULA)':<35} | {'AI DISCOVERED FORMULA':<35} | {'GENS':<5} | {'STATUS'}")
    print("-" * 100)
    
    for entry in session_data:
        target = (str(entry['target'])[:32] + '..') if len(str(entry['target'])) > 32 else str(entry['target'])
        solution = (str(entry['solution'])[:32] + '..') if len(str(entry['solution'])) > 32 else str(entry['solution'])
        status_icon = "✅" if entry['status'] == "SOLVED" else "❌"
        print(f"{entry['level']:<4} | {target:<35} | {solution:<35} | {entry['gens']:<5} | {status_icon}")
        
    print("="*100 + "\n")

def plot_session_summary(all_histories):
    """Plots all levels on one graph at the very end."""
    if not HAS_MATPLOTLIB or not all_histories: return
    
    plt.figure(figsize=(12, 7))
    
    # Loop through saved histories and plot them
    for level, history in all_histories.items():
        plt.plot(history, label=f'Level {level}', linewidth=2, alpha=0.8)
    
    plt.title('GENESIS AI: Performance Comparison Across Levels')
    plt.xlabel('Generations')
    plt.ylabel('Error Rate (Log Scale recommended if errors vary widely)')
    plt.yscale('log') # Log scale makes it easier to see convergence to 0
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    print(">>> Displaying Cumulative Performance Graph...")
    plt.show()

# ==========================================
# CORE EVOLUTION ENGINE
# ==========================================

def evolve_solution(level_num):
    """Runs the genetic algorithm for a single level."""
    population = []
    for _ in range(POPULATION_SIZE):
        tree = gp.generate_random_tree(depth=4)
        score = problem.calculate_fitness(tree)
        population.append({'tree': tree, 'score': score, 'size': count_nodes(tree)})

    history_best = []
    solved = False
    best_tree_found = None
    generations_used = GENERATIONS
    
    for gen in range(1, GENERATIONS + 1):
        population.sort(key=lambda x: (x['score'], x['size']))
        best = population[0]
        history_best.append(best['score'])
        
        best_tree_found = best['tree']
        generations_used = gen
        
        if gen % 10 == 0 or gen == 1:
            print(f"   Gen {gen}: Error = {best['score']:.4f}")

        if best['score'] < 0.1:
            print(f"   >>> SOLVED in Gen {gen}!")
            print(f"   >>> AI Solution: {best['tree']}")
            solved = True
            break

        next_gen = population[:10]
        while len(next_gen) < POPULATION_SIZE:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            child = crossover(p1, p2) if random.random() < BASE_CROSSOVER_RATE else copy.deepcopy(p1)
            if random.random() < BASE_MUTATION_RATE: child = mutation(child)
            score = problem.calculate_fitness(child)
            next_gen.append({'tree': child, 'score': score, 'size': count_nodes(child)})
        population = next_gen

    return solved, history_best, best_tree_found, generations_used

# ==========================================
# INFINITE MODE CONTROLLER
# ==========================================

def run_infinite_mode():
    print("\n==========================================")
    print(" GENESIS: INFINITE DISCOVERY MODE ")
    print(" The AI will face endlessly generated formulas.")
    print("==========================================\n")
    
    level = 1
    difficulty = 2 
    session_data = [] 
    all_histories = {} # Store graph data here
    
    while True:
        print(f"\n--- LEVEL {level} (Complexity Depth: {difficulty}) ---")
        problem.generate_new_environment(difficulty_depth=difficulty)
        
        # Run AI
        success, history, best_tree, gens = evolve_solution(level)
        
        # Store Data
        status_str = "SOLVED" if success else "FAILED"
        session_data.append({
            'level': level,
            'target': str(problem.HIDDEN_TRUTH_TREE),
            'solution': str(best_tree),
            'gens': gens,
            'status': status_str
        })
        all_histories[level] = history # Save for final graph
        
        # Interaction Logic
        if success:
            print(f"   [+] Level {level} Complete.")
            choice = input(f"   >>> Continue to Level {level+1}? (y/n): ")
            if choice.lower() == 'y':
                level += 1
                if level % 3 == 0:
                    difficulty += 1
                    print("   [!] WARNING: Difficulty Increasing!")
            else:
                break
        else:
            print("   [-] AI Failed.")
            choice = input("   >>> Try again? (y/n): ")
            if choice.lower() != 'y':
                break
    
    # FINAL OUTPUT
    print_final_report(session_data)
    plot_session_summary(all_histories)

if __name__ == "__main__":
    run_infinite_mode()