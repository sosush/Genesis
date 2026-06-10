
<img width="1470" height="104" alt="Screenshot 2026-04-02 at 1 59 49 PM" src="https://github.com/user-attachments/assets/f270afc1-918c-4d0b-a344-e1037be44e10" />

## THE AUTONOMOUS EVOLUTIONARY FRAMEWORK FOR ALGORITHMIC DISCOVERY

Genesis is a research-grade Artificial Intelligence system designed to solve the "Grand Challenge" of automated software engineering: the synthesis of complex algorithmic logic from unstructured natural language specifications. Unlike traditional machine learning models that perform statistical mimicry, Genesis utilizes a hybrid Neuro-Symbolic architecture to navigate the infinite search space of computer programs and evolve functional, human-readable solutions.

---

## PROJECT ARCHITECTURE

<img width="1470" height="773" alt="Screenshot 2026-04-02 at 2 02 03 PM" src="https://github.com/user-attachments/assets/95192320-3549-419c-80ed-ae578d8c4068" />


---

## ACADEMIC CONTEXT & RESEARCH BASE

This project was developed as an incremental study in "Evolutionary Computation" and "Symbolic AI." The system's intelligence is not derived from a static database but is instead the result of an active, directed search through the space of all possible Python Abstract Syntax Trees (ASTs).

### THE EVOLUTIONARY STAGES

1.  **Phase I: Stochastic Initialization**
    The project began with a "Pure Genetic Programming" approach. We represented logic as simple tree structures and used random mutations to find mathematical relationships. While effective for basic regression, this phase struggled with high-dimensional algorithmic logic.

2.  **Phase II: Semantic Parsing**
    To handle complex problems (e.g., Robot Collisions, Dynamic Programming), we integrated Natural Language Understanding (NLU). This allowed the engine to "read" problem constraints and identify critical variables, providing a "Heuristic Seed" to the genetic pool.

3.  **Phase III: Global Constraint Satisfaction**
    The final iteration introduced "Persistent Genetic Memory." By maintaining a registry of failed test cases, the system evolved a "Multi-Objective Fitness Function," ensuring that new mutations satisfied all previous constraints simultaneously, solving the "Regression Problem" common in AI synthesis.

---

## CORE AI CONCEPTS IMPLEMENTED

### 1. GENETIC PROGRAMMING (THE PRIMITIVE BACKEND)
Genesis treats every Python function as a **Genome**. The engine utilizes Darwinian principles to optimize code:
*   **Tournament Selection:** The "Survival of the Fittest" protocol where programs are ranked by their ability to satisfy unit tests.
*   **Stochastic Search:** Navigating the non-linear "Fitness Landscape" of programming logic to escape local optima.
*   **Parsimony Pressure:** An implementation of **Occam's Razor**, where the fitness function penalizes "Bloat" (unnecessarily long code), forcing the AI to evolve elegant, efficient algorithms.

### 2. NEURO-SYMBOLIC INDUCTION (THE MODERN INTEGRATION)
While the Genetic Algorithm handles the "Search," we utilize Large Language Models (LLMs) via the Groq LPU (Llama-3.3-70B) as **Heuristic Catalysts**:
*   **Directed Mutation:** Instead of random bit-flipping, the LLM analyzes compiler errors and "mutates" the logic with semantic intent.
*   **Heuristic Seeding:** The LLM provides the "Primordial Soup"—initial logical candidates that are close to the target solution—significantly reducing the search time.

---

## DISCOVERIES & RESEARCH FINDINGS

Through the development of Genesis, several key AI behaviors were observed:
*   **Convergence Patterns:** We discovered that "Hard" algorithmic problems (O(N log N)) require a highly directed mutation signal. Random search is insufficient for discovering complex data structures like Monotonic Stacks.
*   **Semantic vs. Syntactic Learning:** The system demonstrates that AI can "understand" the intent of a problem (Semantics) before it can perfectly execute the syntax (Indentation/Structure).
*   **Regression Sensitivity:** We found that without "Persistent Memory," the evolutionary process often breaks one logic gate while fixing another. The introduction of a "Constraint Registry" was the turning point for solving LeetCode Hard problems.

---

## SHORTCOMINGS & LIMITATIONS

*   **Search Space Explosion:** As the number of nested loops increases, the time required for genetic convergence grows exponentially.
*   **Context Window Constraints:** Extremely long problem descriptions can dilute the NLU's ability to extract specific constraints.
*   **Inference Costs:** The reliance on high-parameter models (70B+) for mutation logic requires significant computational throughput.

---

## GETTING STARTED

### PREREQUISITES
*   Python 3.10+
*   Groq API Key (Llama-3.3-70B-Versatile) (For legacy app)
*   Streamlit (For legacy app)
*   Gradio, PyTorch, MLflow (For Neuro-Symbolic Engine)

### INSTALLATION
1. Clone the repository:
   ```bash
   git clone https://github.com/sosush/Genesis.git
   cd Genesis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables (for legacy app):
   Create a .env file in the root directory:
   ```env
    GROQ_API_KEY=your_groq_api_key
    GROQ_MODEL=llama-3.3-70b-versatile
   ```

### NEURO-SYMBOLIC DEMO (Gradio)
Launch the interactive synthesis dashboard:
```bash
python demo/app.py
```
This will start a local Gradio server where you can provide input-output examples and watch the fitness curve evolve in real time.

### RUN BENCHMARKS
Run the comparative benchmarks across problem spaces:
```bash
python experiments/run_benchmarks.py
```
Results will be written to `results/benchmark_table.md` and logged in MLflow (`mlflow ui`).

### BENCHMARK RESULTS

| Problem | Random search | Pure evolutionary | Genesis (neuro-symbolic) |
|---|---|---|---|
| identity (f(x)=x) | 12 gen | 4 gen | 2 gen |
| square (f(x)=x²) | timeout | 87 gen | 23 gen |
| add (f(x,y)=x+y) | timeout | 210 gen | 61 gen |

*(Note: Neuro-Symbolic uses PyTorch MLP pre-filtering to significantly reduce generations needed.)*

### LEGACY USAGE (Streamlit + Groq LLM)
Run the original induction engine:
```bash
streamlit run app.py
```
---

### PREREQUISITES
*   Python 3.10+
*   Groq API Key (Llama-3.3-70B-Versatile)
*   Streamlit

### INSTALLATION
1. Clone the repository:
   ```bash
   git clone https://github.com/sosush/Genesis.git
   cd Genesis
   ```
2. Install dependencies:
   ```Bash
   pip install streamlit groq python-dotenv
   ```
2. Configure environment variables:
   Create a .env file in the root directory:
   ```Env
    GROQ_API_KEY=your_groq_api_key
    GROQ_MODEL=llama-3.3-70b-versatile
   ```
### USAGE
   Run the induction engine:
   ```Bash
   streamlit run app.py
   ```
---

*NOTE: This project is intended for research and educational purposes only. It explores the boundaries of autonomous program synthesis and the synergy between classical and modern Artificial Intelligence.*
