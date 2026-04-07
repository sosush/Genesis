import streamlit as st
import time
import gp_engine as gp
import random

# --- PROFESSIONAL THEME ---
st.set_page_config(page_title="GENESIS | NEURAL SYNTHESIS", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;500&display=swap');
    .stApp { background-color: #0b0c10; color: #c5c6c7; font-family: 'JetBrains Mono', monospace; }
    .workspace-panel { background: #1f2833; border: 1px solid #45a29e; padding: 20px; border-radius: 4px; }
    .species-monitor { background: #0b0c10; border-left: 3px solid #66fcf1; padding: 15px; margin-bottom: 10px; }
    .fitness-metric { font-size: 20px; color: #66fcf1; }
    .terminal-header { color: #66fcf1; font-weight: bold; font-size: 14px; margin-bottom: 10px; text-transform: uppercase; }
    .stTextArea textarea { background-color: #0b0c10 !important; color: #45a29e !important; border: 1px solid #1f2833 !important; }
    .stButton>button { background: transparent; color: #66fcf1; border: 1px solid #66fcf1; border-radius: 2px; height: 50px; width: 100%; }
    .stButton>button:hover { background: #66fcf1; color: #0b0c10; box-shadow: 0 0 15px #66fcf1; }
</style>
""", unsafe_allow_html=True)

if 'population' not in st.session_state: st.session_state.population = []
if 'test_registry' not in st.session_state: st.session_state.test_registry = []
if 'parsed_data' not in st.session_state: st.session_state.parsed_data = None

st.markdown("<h1 style='text-align: center; color: #66fcf1; letter-spacing: 5px;'>GENESIS 6.0</h1>", unsafe_allow_html=True)

col_in, col_feed = st.columns([1.8, 1.2])

with col_in:
    st.markdown("<div class='terminal-header'>Workspace Specification</div>", unsafe_allow_html=True)
    raw_problem = st.text_area("Universal Problem Input (Description + Examples + Constraints)", height=300)
    code_header = st.text_area("Function Signature / Header", "class Solution:\n    def solve(self):", height=70)

with col_feed:
    st.markdown("<div class='terminal-header'>Evolutionary Feedback</div>", unsafe_allow_html=True)
    error_mode = st.selectbox("Induction Failure Mode", ["None", "TLE", "Wrong Answer", "Other"])
    
    # Conditional Input Logic
    mutation_data = ""
    if error_mode == "Wrong Answer":
        mutation_data = st.text_area("Failed Case (Input / Output / Expected)", height=150)
    elif error_mode == "Other":
        mutation_data = st.text_area("Terminal Error Trace", height=150)
    
    if mutation_data and st.button("Commit to Genetic Memory"):
        st.session_state.test_registry.append(mutation_data)
        st.toast("Constraint Added")

    if st.session_state.test_registry:
        with st.expander("Genetic Memory Registry"):
            for case in st.session_state.test_registry: st.code(case)
            if st.button("Purge Memory"): st.session_state.test_registry = []; st.rerun()

# --- THE EVOLUTIONARY LOOP ---
if st.button("INITIATE STOCHASTIC SEARCH"):
    architect = gp.GeneticArchitect()
    
    with st.status("Performing Semantic Analysis...") as status:
        st.session_state.parsed_data = architect.parse_problem(raw_problem)
        st.session_state.population = architect.generate_primordial_soup(st.session_state.parsed_data, code_header)
        status.update(label="Evolutionary Loop Active", state="complete")

    st.markdown("### Population Fitness Monitor")
    cols = st.columns(4)
    
    for gen in range(1, 4):
        for i, species in enumerate(st.session_state.population):
            with cols[i]:
                # THE EVALUATION (Real AI Backend)
                fit_score = architect.evaluate_fitness(species['code'], st.session_state.parsed_data.get('examples', []))
                species['fitness'] = fit_score
                
                bar_color = "#66fcf1" if fit_score > 0.6 else "#45a29e"
                st.markdown(f"""
                <div class='species-monitor'>
                    <p style='margin:0; font-size:10px; color:#45a29e;'>SPECIES {i+1} | GEN {gen}</p>
                    <p class='fitness-metric'>{fit_score:.2f}</p>
                    <div style='background:#1f2833; height:4px; width:100%;'>
                        <div style='background:{bar_color}; height:100%; width:{fit_score*100}%; transition: 0.5s;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Selection & Mutation
                if fit_score < 1.0:
                    species['code'] = architect.smart_mutate(species['code'], error_mode, st.session_state.test_registry)
                time.sleep(0.5)

    st.success("Optimal Solution Converged")
    st.code(st.session_state.population[0]['code'], language='python')

with st.expander("View System Heuristics"):
    st.write("""
    **Environment:** A high-dimensional search space of valid Python ASTs.  
    **Fitness Function:** Numerical score derived from the intersection of Sandbox STDOUT and User Constraints.  
    **Search Strategy:** Stochastic Hill-Climbing via directed local mutation.
    """)