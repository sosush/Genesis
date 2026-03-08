import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import graphviz
import time
import random
import copy
import sympy
import gp_engine as gp

# ==========================================
# CONFIG & PURPLE THEME
# ==========================================
st.set_page_config(page_title="GENESIS", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0b0b15; color: #e0e0e0; }
    .glass-card {
        background: rgba(124, 77, 255, 0.05);
        border: 1px solid rgba(124, 77, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    h1, h2, h3 { color: #b388ff !important; font-family: 'Segoe UI', sans-serif; }
    .stButton>button {
        background: linear-gradient(90deg, #651fff, #b388ff);
        color: white; font-weight: bold; border: none; border-radius: 8px; height: 50px; font-size: 18px;
    }
    [data-testid="stSidebar"] { background-color: #12121f; border-right: 1px solid #2a2a40; }
    .log-list { margin-left: 10px; font-family: monospace; font-size: 13px; color: #b0b0c0; }
    .stProgress > div > div > div > div { background-color: #b388ff; }
    .wrapped-code {
        background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d;
        font-family: 'Courier New', monospace; color: #00e676; word-wrap: break-word; white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# STATE MANAGEMENT
# ==========================================
if 'page' not in st.session_state: st.session_state.page = "LANDING"
if 'data' not in st.session_state: st.session_state.data = [] # Replaces 'patients'
if 'features' not in st.session_state: st.session_state.features = []
if 'target_col' not in st.session_state: st.session_state.target_col = "Safe_Dose"
if 'best_tree' not in st.session_state: st.session_state.best_tree = None
if 'best_score' not in st.session_state: st.session_state.best_score = 9999
if 'history_logs' not in st.session_state: st.session_state.history_logs = []
if 'csv_mode' not in st.session_state: st.session_state.csv_mode = False

SCENARIOS = {
    "Linear Dose": { "formula": "(Weight * 5) + (Age * 2)", "func": lambda w, a: (w * 5) + (a * 2) },
    "Quadratic Risk": { "formula": "(Weight * Weight) / 50 + Age", "func": lambda w, a: (w**2 / 50) + a },
    "Geriatric Decay": { "formula": "1000 - (Age * 10) + Weight", "func": lambda w, a: 1000 - (a * 10) + w }
}
if 'current_scenario' not in st.session_state: st.session_state.current_scenario = "Linear Dose"

# ==========================================
# LOGIC ENGINE
# ==========================================
def generate_simulation_data():
    """Generates synthetic medical data."""
    func = SCENARIOS[st.session_state.current_scenario]["func"]
    st.session_state.data = []
    st.session_state.features = ['Weight', 'Age']
    st.session_state.target_col = "Safe_Dose"
    
    for i in range(5):
        w = random.randint(50, 100)
        a = random.randint(20, 80)
        dose = int(func(w, a))
        st.session_state.data.append({'ID': f"P-{100+i}", 'Weight': w, 'Age': a, 'Safe_Dose': dose})

def calculate_fitness(tree):
    err = 0
    if not st.session_state.data: return 9999
    
    for row in st.session_state.data:
        try:
            pred = tree.evaluate(row)
            actual = row[st.session_state.target_col]
            err += abs(actual - pred)
        except: return 9999
    return err / len(st.session_state.data)

def perform_crossover(p1, p2):
    child = copy.deepcopy(p1)
    nodes_c = gp.get_subtrees(child)
    nodes_d = gp.get_subtrees(p2)
    if not nodes_c or not nodes_d: return child
    target = random.choice(nodes_c)
    donor = random.choice(nodes_d)
    return gp.replace_subtree(child, target, copy.deepcopy(donor))

def tree_to_sympy(node):
    if node is None: return sympy.Integer(0)
    if hasattr(node, 'value'):
        if isinstance(node.value, str): return sympy.Symbol(node.value)
        return sympy.Integer(node.value) if isinstance(node.value, int) else sympy.Float(node.value)
    if hasattr(node, 'left') and hasattr(node, 'right'):
        l = tree_to_sympy(node.left)
        r = tree_to_sympy(node.right)
        if node.op == '+': return l + r
        if node.op == '-': return l - r
        if node.op == '*': return l * r
        if node.op == '/': return l / r
    return sympy.Integer(0)

# ==========================================
# VISUALIZERS
# ==========================================
def get_3d_grids(tree=None):
    # Only works for 2-variable problems (Weight/Age)
    if 'Weight' not in st.session_state.features or 'Age' not in st.session_state.features:
        return None, None, None, None
        
    x = np.linspace(40, 100, 20)
    y = np.linspace(20, 80, 20)
    X, Y = np.meshgrid(x, y)
    
    # Check if we are in Sim mode to show "Real" surface
    Z_oracle = np.zeros_like(X)
    if not st.session_state.csv_mode:
        func = SCENARIOS[st.session_state.current_scenario]["func"]
        Z_oracle = func(X, Y)
        
    Z_ai = np.zeros_like(X)
    if tree:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    val = tree.evaluate({'Weight': X[i][j], 'Age': Y[i][j]})
                    Z_ai[i][j] = min(max(val, 0), 2000)
                except: Z_ai[i][j] = 0
    return X, Y, Z_oracle, Z_ai

def tree_to_dot(node):
    dot = graphviz.Digraph()
    dot.attr(bgcolor='transparent')
    dot.attr('node', style='filled', fontname='Helvetica', fontcolor='white')
    dot.attr('edge', color='white')
    node_count = 0
    def add(n, pid=None):
        nonlocal node_count
        cid = str(node_count)
        node_count += 1
        if hasattr(n, 'value'):
            lbl = str(n.value)
            fill = '#00e676' if isinstance(n.value, str) else '#ffea00'
            shape = 'box'
            font = 'black'
        else:
            lbl = str(n.op)
            fill = '#651fff'
            shape = 'circle'
            font = 'white'
        dot.node(cid, lbl, shape=shape, fillcolor=fill, fontcolor=font, color='white')
        if pid: dot.edge(pid, cid, color='white')
        if hasattr(n, 'left') and n.left: add(n.left, cid)
        if hasattr(n, 'right') and n.right: add(n.right, cid)
    if node: add(node)
    return dot

# ==========================================
# PAGE: LANDING
# ==========================================
if st.session_state.page == "LANDING":
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<h1 style='text-align: center; font-size: 80px; color: #b388ff;'>GENESIS</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #9575cd;'>Autonomous Logic Synthesis Engine</h3>", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("Begin Discovery", use_container_width=True):
                st.session_state.page = "DASHBOARD"
                st.rerun()


# ==========================================
# PAGE: DASHBOARD (Unified)
# ==========================================
elif st.session_state.page == "DASHBOARD":
    
    with st.sidebar:
        if st.button("⬅️ HOME"): st.session_state.page = "LANDING"; st.rerun()
        st.header("1. Input Pipeline")
        
        # --- NEW: CSV UPLOAD FEATURE ---
        mode = st.radio("Data Source:", ["Clinical Simulation", "Upload CSV File"])
        
        if mode == "Upload CSV File":
            st.session_state.csv_mode = True
            uploaded_file = st.file_uploader("Upload Dataset (.csv)", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df.to_dict('records')
                # Auto-detect columns
                cols = list(df.columns)
                target = st.selectbox("Select Target Column (Y):", cols, index=len(cols)-1)
                st.session_state.target_col = target
                st.session_state.features = [c for c in cols if c != target and c != "ID"]
                st.success(f"Loaded {len(df)} rows.")
        else:
            st.session_state.csv_mode = False
            st.subheader("Simulation Config")
            new_scen = st.radio("Target Formula:", list(SCENARIOS.keys()))
            if new_scen != st.session_state.current_scenario or not st.session_state.data:
                st.session_state.current_scenario = new_scen
                generate_simulation_data()
            
            if st.button("Regenerate Data"): 
                generate_simulation_data()
                st.rerun()

    # MAIN TITLE
    mode_title = "External Dataset Analysis" if st.session_state.csv_mode else f"Simulation: {st.session_state.current_scenario}"
    st.title(f"🔍 {mode_title}")
    
    # 1. SHOW DATA
    if st.session_state.data:
        st.markdown(f"### Step 1: Input Data ({len(st.session_state.data)} samples)")
        st.dataframe(pd.DataFrame(st.session_state.data), use_container_width=True)
    else:
        st.warning("Waiting for data...")
        st.stop()

    # 2. EVOLUTION
    st.markdown("### Step 2: Evolutionary Analysis")
    if st.button("🧬 INITIATE GENESIS ENGINE", type="primary", use_container_width=True):
        prog = st.progress(0)
        status = st.empty()
        st.session_state.history_logs = []
        
        # INIT
        pop = []
        for _ in range(600):
            t = gp.generate_random_tree(depth=5, feature_names=st.session_state.features)
            pop.append({'tree': t, 'score': calculate_fitness(t)})
            
        # LOOP
        for gen in range(1, 31): 
            pop.sort(key=lambda x: x['score'])
            best = pop[0]
            
            # LOGS
            survivors = [f"• {str(x['tree'])} (Err: {x['score']:.1f})" for x in pop[:3]]
            dead = [f"• {str(x['tree'])} (Err: {x['score']:.1f})" for x in pop[-3:]]
            breeding_events = []
            
            next_gen = pop[:30] # Elitism
            while len(next_gen) < 600:
                p1 = random.choice(pop[:100])['tree']
                p2 = random.choice(pop[:100])['tree']
                child = perform_crossover(p1, p2)
                if random.random() < 0.3:
                    child = gp.replace_subtree(child, random.choice(gp.get_subtrees(child)), 
                                               gp.generate_random_tree(depth=2, feature_names=st.session_state.features))
                
                if len(breeding_events) < 3: breeding_events.append(f"• Parent A: [{p1}] + Parent B: [{p2}] -> Child: [{child}]")
                next_gen.append({'tree': child, 'score': calculate_fitness(child)})
            pop = next_gen
            
            st.session_state.history_logs.append({"gen": gen, "survivors": survivors, "dead": dead, "breeding": breeding_events})
            status.markdown(f"<div class='glass-card'><b>Gen {gen}/30</b><br>Best: <code style='color:#b388ff'>{best['tree']}</code> | Error: <b style='color:#ff4b4b'>{best['score']:.2f}</b></div>", unsafe_allow_html=True)
            prog.progress(gen/30)
            
            if best['score'] < 0.1:
                prog.progress(1.0)
                st.success("✅ CONVERGENCE ACHIEVED")
                break
            time.sleep(0.01)
            
        st.session_state.best_tree = best['tree']
        st.session_state.best_score = best['score']

    # 3. RESULTS
    if st.session_state.best_tree:
        st.markdown("### Step 3: Verification")
        c1, c2 = st.columns(2)
        with c1: 
            if st.session_state.csv_mode:
                st.info("Target: **Hidden / Unknown (CSV)**")
            else:
                st.info(f"Target: {SCENARIOS[st.session_state.current_scenario]['formula']}")
        with c2: st.success(f"AI Result: {st.session_state.best_tree}")

        # MATH KERNEL
        with st.expander("🧮 MATH KERNEL (Algebraic Verification)"):
            try:
                raw_sympy = tree_to_sympy(st.session_state.best_tree)
                simplified = sympy.simplify(raw_sympy)
                st.markdown("**1. Raw Logic:**")
                st.markdown(f"<div class='wrapped-code'>{str(st.session_state.best_tree)}</div>", unsafe_allow_html=True)
                st.markdown("**2. Simplified Equation:**")
                st.markdown(f"<div class='wrapped-code'>{str(simplified)}</div>", unsafe_allow_html=True)
            except: st.error("Simplification Engine requires SymPy.")

        # VISUALS
        with st.expander("🔬 VISUALS: 3D Manifold & Tree Structure"):
            t1, t2 = st.tabs(["🧊 3D Visualization", "🌳 Logic Tree"])
            with t1:
                X, Y, Z_real, Z_ai = get_3d_grids(st.session_state.best_tree)
                if X is not None:
                    fig = go.Figure()
                    if not st.session_state.csv_mode:
                        fig.add_trace(go.Surface(z=Z_real, x=X, y=Y, colorscale='Reds', opacity=0.5, name='Truth', showscale=False))
                    fig.add_trace(go.Surface(z=Z_ai, x=X, y=Y, colorscale='Viridis', opacity=0.8, name='AI Logic'))
                    fig.update_layout(height=500, margin=dict(l=0,r=0,b=0,t=0), scene=dict(xaxis_title='Weight', yaxis_title='Age', zaxis_title='Target'))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("3D Visualization requires exactly 2 input variables (e.g. Weight & Age).")
            with t2:
                try: st.graphviz_chart(tree_to_dot(st.session_state.best_tree))
                except: st.error("Graphviz not found.")

        # LOGS
        with st.expander("📜 EVOLUTIONARY LOGS"):
            if st.session_state.history_logs:
                tabs = st.tabs([f"Gen {log['gen']}" for log in st.session_state.history_logs])
                for i, tab in enumerate(tabs):
                    log = st.session_state.history_logs[i]
                    with tab:
                        c_s, c_d = st.columns(2)
                        with c_s:
                            st.markdown("**🏆 Top Survivors**")
                            for s in log['survivors']: st.markdown(f"<span class='log-list' style='color:#00e676'>{s}</span>", unsafe_allow_html=True)
                        with c_d:
                            st.markdown("**💀 Eliminated**")
                            for d in log['dead']: st.markdown(f"<span class='log-list' style='color:#ff1744'>{d}</span>", unsafe_allow_html=True)
                        st.markdown("**🧬 Breeding Events**")
                        for b in log['breeding']: st.markdown(f"<span class='log-list' style='color:#ffea00'>{b}</span>", unsafe_allow_html=True)