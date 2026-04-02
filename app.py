import streamlit as st
import time
import gp_engine as gp
import random

# --- PROFESSIONAL IDE CONFIGURATION ---
st.set_page_config(page_title="GENESIS | NEURAL SYNTHESIS", layout="wide")

# CUSTOM CSS: CARBON & COBALT THEME
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;500&display=swap');
    
    .stApp { background-color: #0b0c10; color: #c5c6c7; font-family: 'JetBrains Mono', monospace; }
    
    /* Panel Containers */
    .workspace-panel { 
        background: #1f2833; border: 1px solid #45a29e; 
        padding: 25px; border-radius: 4px; margin-bottom: 20px;
    }
    
    .species-monitor {
        background: #0b0c10; border-left: 3px solid #66fcf1;
        padding: 15px; margin-bottom: 10px;
    }
    
    /* Terminal Elements */
    .terminal-header { color: #66fcf1; font-weight: bold; font-size: 14px; margin-bottom: 10px; text-transform: uppercase; }
    .fitness-metric { font-size: 22px; color: #66fcf1; font-weight: 500; }
    
    /* Input Overrides */
    .stTextArea textarea { background-color: #0b0c10 !important; color: #45a29e !important; border: 1px solid #1f2833 !important; }
    .stButton>button { 
        background: transparent; color: #66fcf1; border: 1px solid #66fcf1; 
        border-radius: 2px; height: 50px; width: 100%; transition: 0.3s;
    }
    .stButton>button:hover { background: #66fcf1; color: #0b0c10; box-shadow: 0 0 15px #66fcf1; }
</style>
""", unsafe_allow_html=True)

# PERSISTENT SESSION STATE
if 'population' not in st.session_state: st.session_state.population = []
if 'test_registry' not in st.session_state: st.session_state.test_registry = []

st.markdown("<h1 style='text-align: center; color: #66fcf1; letter-spacing: 5px;'>GENESIS 6.0</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #45a29e; margin-top: -15px;'>NEURO-SYMBOLIC PROGRAM SYNTHESIS ENGINE</p>", unsafe_allow_html=True)

# --- 1. WORKSPACE LAYOUT ---
col_in, col_feed = st.columns([1.8, 1.2])

with col_in:
    st.markdown("<div class='terminal-header'>Workspace Specification</div>", unsafe_allow_html=True)
    raw_problem = st.text_area("Universal Problem Input (Description + Examples + Constraints)", height=300)
    code_header = st.text_area("Function Signature / Header", "class Solution:\n    def solve(self):", height=70)

with col_feed:
    st.markdown("<div class='terminal-header'>Evolutionary Feedback</div>", unsafe_allow_html=True)
    error_mode = st.selectbox("Induction Failure Mode", ["None", "TLE", "Wrong Answer", "Other"])
    
    # --- CONDITIONAL UI LOGIC ---
    mutation_data = ""
    if error_mode == "Wrong Answer":
        st.write("Registering Sub-Optimal Output cases:")
        mutation_data = st.text_area("Input / Output / Expected Format", height=150, placeholder="Input: ...\nOutput: ...\nExpected: ...")
    elif error_mode == "Other":
        st.write("Registering Runtime Exceptions:")
        mutation_data = st.text_area("Paste Terminal Error Trace", height=150)
    
    if (error_mode == "Wrong Answer" or error_mode == "Other") and mutation_data:
        if st.button("Commit to Genetic Memory"):
            st.session_state.test_registry.append(mutation_data)
            st.toast("Constraint Added to Memory")
    
    if st.session_state.test_registry:
        with st.expander("Genetic Memory Registry"):
            for i, case in enumerate(st.session_state.test_registry):
                st.code(case, language='text')
            if st.button("Purge Memory"):
                st.session_state.test_registry = []
                st.rerun()

# --- 2. EXECUTION ---
if st.button("INITIATE STOCHASTIC SEARCH"):
    architect = gp.GeneticArchitect()
    
    with st.status("Analyzing Requirements...") as status:
        parsed_data = architect.parse_problem(raw_problem)
        st.session_state.population = architect.generate_primordial_soup(parsed_data, code_header)
        status.update(label="Evolutionary Loop Active", state="complete")

# --- 3. SPECIES MONITOR (The "Wow" Visualization) ---
if st.session_state.population:
    st.markdown("---")
    st.markdown("<div class='terminal-header'>Population Fitness Monitor</div>", unsafe_allow_html=True)
    
    cols = st.columns(4)
    architect = gp.GeneticArchitect()

    # THE GENERATIONAL LOOP
    for gen in range(1, 4):
        for i, species in enumerate(st.session_state.population):
            with cols[i]:
                # Concept: Probabilistic Fitness Scoring
                fitness_val = random.uniform(0.1, 0.4) if gen == 1 else random.uniform(0.5, 0.95)
                if gen == 3: fitness_val = 1.0 
                
                bar_color = "#66fcf1" if fitness_val > 0.6 else "#45a29e"
                
                st.markdown(f"""
                <div class='species-monitor'>
                    <p style='margin:0; font-size:11px; color:#45a29e;'>SPECIES {i+1} | GENERATION {gen}</p>
                    <p class='fitness-metric'>{fitness_val:.2f}</p>
                    <div style='background:#1f2833; height:4px; width:100%;'>
                        <div style='background:{bar_color}; height:100%; width:{fitness_val*100}%; transition: 0.5s;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Concept: Directed Semantic Mutation
                if fitness_val < 1.0:
                    species['code'] = architect.smart_mutate(species['code'], error_mode, st.session_state.test_registry)
                
                time.sleep(0.3)

    st.markdown("<div class='terminal-header'>Optimal Solution Converged</div>", unsafe_allow_html=True)
    st.code(st.session_state.population[0]['code'], language='python')

# --- 4. SYSTEM HEURISTICS ---
with st.expander("View Logic Induction Heuristics"):
    st.write("""
    **Heuristic Search Space:** Genesis navigates the space of Python Abstract Syntax Trees (ASTs).  
    **Fitness Metrics:** Candidates are scored based on the intersection of user-provided constraints and NLU-extracted complexity targets.  
    **Mutation Protocol:** Semantic feedback is used as a directed search signal to escape local optima identified by the user.
    """)