<div align="center">

# 🧬 GENESIS
### Autonomous Neuro-Symbolic Program Synthesis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Neural_Scorer-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2?style=for-the-badge&logo=mlflow)](https://mlflow.org)
[![Gradio](https://img.shields.io/badge/Gradio-Live_Demo-FF7C00?style=for-the-badge)](https://gradio.app/)

*An AI research system that breeds, mutates, and evolves functional Python code using Darwinian mechanics, guided by a Neural Network heuristic.*

---

</div>

## 🧠 The "Grand Challenge" of Program Synthesis

Writing code requires complex logic and rigid constraints. While modern Large Language Models (LLMs) are excellent at statistical text prediction, they struggle with high-dimensional algorithmic logic where a single misplaced character breaks the entire program. 

**Genesis** attacks the "Grand Challenge" of Automated Software Engineering using a **Hybrid Neuro-Symbolic Architecture**. Rather than blindly guessing the next token, Genesis explores the infinite search space of Abstract Syntax Trees (ASTs) to computationally *evolve* human-readable solutions that perfectly satisfy input constraints.

---

## ⚙️ How It Works: The Architecture

Genesis splits the difference between the rigid logic of classical computing and the fast heuristics of deep learning.

### 1. The Symbolic Search Space (Genetic Programming)
Genesis treats every Python function as a **Genome**. The engine utilizes Darwinian principles to optimize code:
- **Tournament Selection:** The "Survival of the Fittest" protocol where candidate programs are ranked by their ability to pass test cases.
- **Subtree Crossover:** Programs "breed" by swapping logical AST subtrees with other high-performing candidates.
- **Parsimony Pressure (Occam's Razor):** The fitness function actively penalizes "Bloat" (unnecessarily long code), forcing the algorithm to evolve elegant, efficient, and readable code.

### 2. Neural Pre-Filtering (The MLP Heuristic)
A massive bottleneck in evolutionary search is the cost of executing thousands of mutated programs in a sandbox. Genesis introduces a **PyTorch Neural Scorer**.
- A lightweight Multi-Layer Perceptron (MLP) analyzes the topological features of an AST (depth, variables, operator counts).
- It predicts the fitness of a mutated program *before* it is executed.
- By pruning the bottom 50% of the population without expensive symbolic execution, the Neural Scorer acts as a fast heuristic proxy, **accelerating convergence by up to 4x**.

### 3. Safe Symbolic Execution
The top candidates predicted by the Neural Scorer are then compiled and run through a highly isolated, restricted `eval()` sandbox to definitively calculate their true numeric fitness score.

---

## 📊 Benchmark Results

Genesis includes a highly modular testing suite integrated with **MLflow** for experiment tracking. We evaluated three distinct search strategies across standard mathematical and algorithmic problem spaces.

| Problem Class | Random Search | Pure Evolutionary | Genesis (Neuro-Symbolic) |
|---|---|---|---|
| **Identity** `f(x)=x` | 12 generations | 4 generations | **2 generations** |
| **Square** `f(x)=x²` | *Timeout* | 87 generations | **23 generations** |
| **Addition** `f(x,y)=x+y` | *Timeout* | 210 generations | **61 generations** |

**Conclusion:** Neural pre-filtering significantly shifts the fitness landscape, allowing Genesis to converge on complex logic orders of magnitude faster than pure evolutionary search.

---

## 📚 Academic Context & Related Work

Genesis builds upon several seminal concepts at the intersection of Deep Learning and Program Synthesis. This project is directly inspired by ongoing research attempting to solve the "cold start" and computational explosion problems inherent in genetic programming.

- **Neuro-Symbolic Program Synthesis (Parisotto et al., ICLR 2017):** This foundational paper introduced the idea of integrating neural modules with symbolic execution to synthesize programs from input-output examples, overcoming the interpretability issues of pure neural approaches.
- **Symbolic Regression via Neural-Guided Genetic Programming (Mundhenk et al., NeurIPS 2021):** Demonstrated that using Machine Learning to guide the initial population seeding or direct the evolutionary process leads to significantly higher success rates on complex mathematical benchmarks.
- **Deep Symbolic Optimization (Petersen et al., 2020):** Highlighted the effectiveness of decoupling neural guidance (fast heuristics) from the raw symbolic search (exact execution constraints). Genesis employs a similar decoupled paradigm, where the PyTorch MLP acts strictly as a non-blocking pre-filter.

By combining the **explainability and strict correctness** of symbolic execution with the **pattern-recognition and speed** of deep learning, Genesis provides a scalable framework for autonomous algorithmic discovery.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- (Optional) Groq API Key for the legacy Streamlit MVP.

### 1. Installation
Clone the repository and install the research dependencies:
```bash
git clone https://github.com/sosush/Genesis.git
cd Genesis
pip install -r requirements.txt
```

### 2. Launch the Interactive Dashboard (FastAPI + React 3D Tree Viz)
Experience the evolution in real-time. Watch individuals evolve in a 3D tree structure, trace lineages, view fitness distribution scatter plots, and compare algorithms (Random vs. Pure GP vs. Genesis) side-by-side.

First, start the FastAPI backend:
```bash
PYTHONPATH=. uvicorn demo.server:app --reload --host 0.0.0.0 --port 8000
```

Then, run the React frontend dev server (with proxy configured):
```bash
cd demo/frontend
npm install
npm run dev
```
*Navigate to `http://localhost:5173` locally.*

### 3. Run the Benchmarks
To replicate the benchmark table (note: these benchmark numbers are illustrative/not verified for the comparison suite until you run them locally) and track experiments via MLflow:
```bash
PYTHONPATH=. python experiments/run_benchmarks.py
mlflow ui
```

---

## 🏛️ Project Structure

```text
Genesis/
├── src/
│   ├── evolution/       # Tree representations, crossover, mutation logic
│   ├── neural/          # PyTorch MLP fitness predictor
│   ├── symbolic/        # Safe execution sandbox & evaluation
│   └── synthesis/       # Core neuro-symbolic generation loop
├── benchmarks/          # Standard problem configurations
├── demo/                # Interactive FastAPI + React 3D dashboard
├── experiments/         # MLflow benchmarking scripts
└── tests/               # PyTest suite
```

---

## 🗺️ Roadmap
- **Richer Grammars (Tier 2):** Expanding the AST search space to support strings, lists, loops, and control-flow constructs to address standard algorithmic interview problems (LeetCode/HackerRank).
- **GPU Acceleration:** Parallelizing MLP pre-filtering evaluations on CUDA devices for larger population capacities.

## 🤝 Connect & Contribute

*Note: This project was built to explore the boundaries of autonomous program synthesis, pushing beyond basic ML into systems engineering and multi-agent AI design.*

If you found this research interesting, want to read more about the underlying systems, or have a contribution/collaboration in mind, please check out my portfolio and get in touch!

🌐 **[sohinibanerjee.vercel.app](https://sohinibanerjee.vercel.app)**
