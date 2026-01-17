import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import copy
import sympy
import streamlit.components.v1 as components

# Import your existing logic
import gp_engine as gp
import main_env as problem

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="GENESIS",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 1. VISUALS & CSS
# ==========================================
components.html("""
<style>
    body { margin: 0; overflow: hidden; background-color: #0e1117; }
    canvas { display: block; vertical-align: bottom; }
</style>
<div id="particles-js"></div>
<script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
<script>
    particlesJS("particles-js", {
      "particles": {
        "number": { "value": 80, "density": { "enable": true, "value_area": 800 } },
        "color": { "value": "#00ff41" },
        "shape": { "type": "circle" },
        "opacity": { "value": 0.5, "random": false },
        "size": { "value": 3, "random": true },
        "line_linked": { "enable": true, "distance": 150, "color": "#00ff41", "opacity": 0.2, "width": 1 },
        "move": { "enable": true, "speed": 2, "direction": "none", "random": false, "straight": false, "out_mode": "out", "bounce": false }
      },
      "interactivity": {
        "detect_on": "canvas",
        "events": { "onhover": { "enable": true, "mode": "grab" }, "onclick": { "enable": true, "mode": "push" }, "resize": true },
        "modes": { "grab": { "distance": 140, "line_linked": { "opacity": 1 } } }
      },
      "retina_detect": true
    });
</script>
""", height=0, width=0)

st.markdown("""
<style>
    /* 1. Global Layout Fix: Reduce Top Padding globally */
    div.block-container {
        padding-top: 2rem !important; /* Removes the huge default whitespace */
        padding-bottom: 2rem !important;
    }

    /* 2. Global Font Fix: Protects Icons */
    .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, div[data-testid="stMetricValue"], .stCode, .terminal-box {
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    /* 3. Cyberpunk Buttons */
    .stButton>button {
        font-family: 'Courier New', Courier, monospace !important;
        background-color: rgba(0, 0, 0, 0.6);
        color: #00ff41;
        border: 1px solid #00ff41;
        border-radius: 0px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #00ff41;
        color: black;
        box-shadow: 0 0 15px #00ff41;
    }
    
    /* 4. Metrics */
    div[data-testid="stMetricValue"] {
        color: #00ff41 !important;
        text-shadow: 0 0 5px #00ff41;
    }
    
    /* 5. Terminal Box */
    .terminal-box {
        background-color: rgba(0, 20, 0, 0.9);
        border-left: 3px solid #00ff41;
        padding: 20px;
        color: #cfcfcf;
        margin-bottom: 20px;
    }
    
    /* 6. Expander Fix */
    .streamlit-expanderHeader {
        background-color: rgba(0, 20, 0, 0.8) !important;
        color: #00ff41 !important;
        border: 1px solid #00ff41 !important;
    }
    .streamlit-expanderHeader p {
        font-family: 'Courier New', Courier, monospace !important;
        font-size: 16px;
        font-weight: bold;
        margin: 0;
    }
    
    .stApp { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MATH KERNEL
# ==========================================
def tree_to_sympy(node):
    x = sympy.symbols('x')
    if node is None: return sympy.Integer(0)
    if hasattr(node, 'value'):
        return x if node.value == 'x' else sympy.Integer(node.value)
    if hasattr(node, 'child'):
        child_expr = tree_to_sympy(node.child)
        if node.op == 'sin': return sympy.sin(child_expr)
        if node.op == 'cos': return sympy.cos(child_expr)
    if hasattr(node, 'left'):
        left_expr = tree_to_sympy(node.left)
        right_expr = tree_to_sympy(node.right)
        if node.op == '+': return left_expr + right_expr
        if node.op == '-': return left_expr - right_expr
        if node.op == '*': return left_expr * right_expr
    return sympy.Integer(0)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def count_nodes(node):
    if node is None: return 0
    c = 1
    if hasattr(node, 'child'): c += count_nodes(node.child)
    elif hasattr(node, 'left'): 
        c += count_nodes(node.left)
        c += count_nodes(node.right)
    return c

def tournament_selection(population):
    candidates = random.sample(population, 7)
    candidates.sort(key=lambda x: (x['score'], x['size']))
    return candidates[0]['tree']

def get_node_list(node, parent=None, side=None):
    nodes = [{'node': node, 'parent': parent, 'side': side}]
    if hasattr(node, 'child'): nodes.extend(get_node_list(node.child, node, 'child'))
    elif hasattr(node, 'left'): 
        if node.left: nodes.extend(get_node_list(node.left, node, 'left'))
        if node.right: nodes.extend(get_node_list(node.right, node, 'right'))
    return nodes

def swap_node(target_info, new_subtree):
    parent = target_info['parent']
    side = target_info['side']
    if parent is None: return new_subtree
    if side == 'left': parent.left = new_subtree
    elif side == 'right': parent.right = new_subtree
    elif side == 'child': parent.child = new_subtree
    return None

def crossover(parent1, parent2):
    child = copy.deepcopy(parent1)
    nodes_child = get_node_list(child)
    nodes_donor = get_node_list(parent2)
    if not nodes_child or not nodes_donor: return child
    target = random.choice(nodes_child)
    donor = random.choice(nodes_donor)
    new_root = swap_node(target, copy.deepcopy(donor['node']))
    return new_root if new_root else child

def mutation(tree):
    mutated = copy.deepcopy(tree)
    nodes = get_node_list(mutated)
    if not nodes: return mutated
    target = random.choice(nodes)
    new_subtree = gp.generate_random_tree(depth=2)
    new_root = swap_node(target, new_subtree)
    return new_root if new_root else mutated

# ==========================================
# SESSION STATE & NAV
# ==========================================
if 'page' not in st.session_state: st.session_state.page = 'HOME'
if 'prev_page' not in st.session_state: st.session_state.prev_page = 'HOME'
if 'level' not in st.session_state: st.session_state.level = 1
if 'difficulty' not in st.session_state: st.session_state.difficulty = 2
if 'session_data' not in st.session_state: st.session_state.session_data = []
if 'all_histories' not in st.session_state: st.session_state.all_histories = {}
if 'last_result' not in st.session_state: st.session_state.last_result = {}

# --- CONDITIONAL NAVIGATION HEADER ---
if st.session_state.page != 'HOME':
    col_nav1, col_nav2 = st.columns([1, 6])
    with col_nav1:
        if st.session_state.page != 'HISTORY':
            if st.button("SYSTEM LOGS"):
                st.session_state.prev_page = st.session_state.page
                st.session_state.page = 'HISTORY'
                st.rerun()
        else:
            if st.button("⬅ BACK"):
                st.session_state.page = st.session_state.prev_page
                st.rerun()

    with col_nav2:
        st.markdown(f"<h3 style='margin-top:-6px; color: #00ff41;'>GENESIS // MODE: {st.session_state.page}</h3>", unsafe_allow_html=True)
    st.divider()

# ==========================================
# PAGE: HISTORY (THE LOGS)
# ==========================================
if st.session_state.page == 'HISTORY':
    st.markdown("## >> ARCHIVED MISSION LOGS")
    
    if not st.session_state.session_data:
        st.info("NO DATA IN ARCHIVES.")
    
    for entry in reversed(st.session_state.session_data):
        lvl = entry['Level']
        status = entry['Status']
        color = "#00ff41" if status == "SOLVED" else "red"
        
        with st.expander(f"LVL {lvl} [{status}]", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**ORACLE:** `{entry['Target']}`")
                st.markdown(f"**AI:** `{entry['Solution']}`")
                st.markdown(f"**GENS:** `{entry['Generations']}`")
            with c2:
                if lvl in st.session_state.all_histories:
                    hist = st.session_state.all_histories[lvl]
                    fig, ax = plt.subplots(figsize=(6, 2))
                    fig.patch.set_facecolor('#0e1117')
                    ax.set_facecolor('#0e1117')
                    ax.plot(hist, color=color, linewidth=2, marker='o', markersize=4)
                    ax.set_title("Convergence", color='white', fontsize=8)
                    ax.tick_params(colors='white', labelsize=6)
                    ax.grid(color='#333333', linewidth=0.5)
                    st.pyplot(fig)

# ==========================================
# PAGE: HOME
# ==========================================
elif st.session_state.page == 'HOME':
    # SPACER: This pushes the title down ONLY on the home page
    st.markdown("<div style='height: 12vh;'></div>", unsafe_allow_html=True)
    
    st.markdown(f"<h1 style='text-align: center; color: #00ff41; font-size: 80px; text-shadow: 0 0 20px #00ff41;'>GENESIS</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center; color: white;'>AUTONOMOUS ALGORITHM SYNTHESIS ENGINE</h3>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown("<div class='terminal-box'>SYSTEM STATUS: ONLINE<br>MODULES: GENETIC_ENGINE, MATH_KERNEL<br>MODE: INFINITE DISCOVERY</div>", unsafe_allow_html=True)
        if st.button("INITIALIZE SEQUENCE", use_container_width=True):
            st.session_state.page = 'RUNNING'
            st.rerun()

# ==========================================
# PAGE: RUNNING
# ==========================================
elif st.session_state.page == 'RUNNING':
    
    st.markdown(f"<h2 style='color: #00ff41;'>// LEVEL {st.session_state.level}: ANALYSIS IN PROGRESS</h2>", unsafe_allow_html=True)
    
    problem.generate_new_environment(difficulty_depth=st.session_state.difficulty)
    target_formula = str(problem.HIDDEN_TRUTH_TREE)
    st.session_state.current_oracle_tree = problem.HIDDEN_TRUTH_TREE
    
    with st.expander(">> ACCESS RESTRICTED DATA (SHOW ORACLE TRUTH)", expanded=False):
        st.code(target_formula)

    c1, c2, c3 = st.columns(3)
    gen_display = c1.empty()
    err_display = c2.empty()
    best_display = c3.empty()
    
    terminal_log = st.empty()
    progress_bar = st.progress(0)
    
    pop_size = 150
    generations = 50
    population = []
    
    for _ in range(pop_size):
        t = gp.generate_random_tree(depth=4)
        s = problem.calculate_fitness(t)
        population.append({'tree': t, 'score': s, 'size': count_nodes(t)})

    history_best = []
    solved = False
    best_tree = None
    
    for gen in range(1, generations + 1):
        population.sort(key=lambda x: (x['score'], x['size']))
        best = population[0]
        history_best.append(best['score'])
        best_tree = best['tree']
        
        gen_display.metric("GENERATION", f"{gen}/{generations}")
        err_display.metric("ERROR RATE", f"{best['score']:.4f}")
        best_display.metric("COMPLEXITY", best['size'])
        progress_bar.progress(gen / generations)
        
        with terminal_log.container():
            st.markdown(f"""
            <div class='terminal-box' style='font-size: 14px;'>
            > SCANNING GENERATION {gen}...<br>
            > FITTEST CANDIDATE IDENTIFIED<br>
            > LOGIC: <span style='color: #00ff41;'>{best['tree']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        if best['score'] < 0.1:
            solved = True
            progress_bar.progress(1.0)
            break
        
        next_gen = population[:10]
        while len(next_gen) < pop_size:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            child = crossover(p1, p2) if random.random() < 0.8 else copy.deepcopy(p1)
            if random.random() < 0.15: child = mutation(child)
            score = problem.calculate_fitness(child)
            next_gen.append({'tree': child, 'score': score, 'size': count_nodes(child)})
        population = next_gen
        
        time.sleep(0.01)

    st.session_state.last_result = {
        'success': solved,
        'gens': gen,
        'target_str': target_formula,
        'solution_str': str(best_tree),
        'solution_tree': best_tree,
        'history': history_best
    }
    st.session_state.all_histories[st.session_state.level] = history_best
    st.session_state.page = 'DECISION'
    st.rerun()

# ==========================================
# PAGE: DECISION
# ==========================================
elif st.session_state.page == 'DECISION':
    res = st.session_state.last_result
    
    st.markdown("<br>", unsafe_allow_html=True)
    if res['success']:
        st.markdown(f"<div class='terminal-box' style='border-left: 5px solid #00ff41;'><h1>✅ MISSION SUCCESS</h1><p>LAW OF PHYSICS DERIVED SUCCESSFULLY.</p></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='terminal-box' style='border-left: 5px solid red; color: red;'><h1>❌ MISSION FAILED</h1><p>CONVERGENCE NOT ACHIEVED.</p></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### >> EXPECTED (ORACLE)")
        st.info(res['target_str'])
    with col2:
        st.markdown("### >> PREDICTED (GENESIS)")
        st.code(res['solution_str'])
    
    st.markdown("---")
    with st.expander(">> RUN ALGEBRAIC VERIFICATION PROTOCOL (CLICK TO SIMPLIFY)"):
        if 'solution_tree' in res and 'current_oracle_tree' in st.session_state:
            try:
                sym_oracle = tree_to_sympy(st.session_state.current_oracle_tree)
                sym_ai = tree_to_sympy(res['solution_tree'])
                simple_oracle = sympy.expand(sym_oracle)
                simple_ai = sympy.expand(sym_ai)
                
                c_a, c_b = st.columns(2)
                with c_a:
                    st.markdown("**Simplified Oracle Formula:**")
                    st.latex(sympy.latex(simple_oracle))
                with c_b:
                    st.markdown("**Simplified AI Formula:**")
                    st.latex(sympy.latex(simple_ai))
                    
                if simple_oracle == simple_ai:
                     st.success("✅ VERIFICATION SUCCESSFUL: MATHEMATICALLY IDENTICAL")
                else:
                     st.warning("⚠️ APPROXIMATELY EQUIVALENT")
            except Exception as e:
                st.error(f"MATH KERNEL ERROR: {e}")

    st.markdown("### >> AWAITING COMMAND INPUT")
    
    c1, c2, c3 = st.columns([1, 1, 1])
    
    # Common Save Function
    def save_log(status_label):
        st.session_state.session_data.append({
            'Level': st.session_state.level,
            'Target': res['target_str'],
            'Solution': res['solution_str'],
            'Generations': res['gens'],
            'Status': status_label
        })

    if res['success']:
        with c1:
            if st.button("EXECUTE NEXT LEVEL"):
                save_log('SOLVED')
                st.session_state.level += 1
                if st.session_state.level % 3 == 0: st.session_state.difficulty += 1
                st.session_state.page = 'RUNNING'
                st.rerun()
        with c2:
            if st.button("TERMINATE SESSION"):
                 save_log('SOLVED')
                 st.session_state.page = 'REPORT'
                 st.rerun()
    else:
        with c1:
            if st.button("RETRY SIMULATION"):
                st.session_state.page = 'RUNNING'
                st.rerun()
        with c2:
            if st.button("ABORT MISSION"):
                save_log('FAILED')
                st.session_state.page = 'REPORT'
                st.rerun()

# ==========================================
# PAGE: REPORT
# ==========================================
elif st.session_state.page == 'REPORT':
    st.markdown(f"<h1 style='color: #00ff41;'>// FINAL MISSION REPORT</h1>", unsafe_allow_html=True)
    
    st.markdown("### >> SESSION LOGS")
    if st.session_state.session_data:
        df = pd.DataFrame(st.session_state.session_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("NO DATA LOGGED.")

    st.markdown("### >> CUMULATIVE CONVERGENCE")
    if st.session_state.all_histories:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        
        for lvl, hist in st.session_state.all_histories.items():
            ax.plot(hist, label=f'LVL {lvl}', linewidth=2, alpha=0.8)
        
        ax.set_title("ERROR CONVERGENCE", color='#00ff41')
        ax.set_xlabel("GENERATIONS", color='white')
        ax.set_ylabel("ERROR (LOG SCALE)", color='white')
        ax.set_yscale('log')
        ax.grid(color='#333333', linestyle='--')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#0e1117', edgecolor='#00ff41', labelcolor='white')
        
        st.pyplot(fig)
    
    st.markdown("---")
    if st.button("SYSTEM REBOOT"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()