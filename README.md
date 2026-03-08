# 🧬 GENESIS: Clinical Logic Discovery Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success)

> **"Bridging the gap between Black Box AI and Medical Safety through Evolutionary Symbolic Regression."**

---

## 📖 Overview

**Genesis** is an Explainable AI (XAI) system designed for high-stakes environments like Precision Medicine. 

Unlike traditional Neural Networks which output opaque predictions, Genesis uses **Genetic Programming (GP)** to evolve human-readable mathematical formulas from raw data. It acts as an "Automated Data Scientist," ingesting clinical datasets (e.g., Patient Weight, Age, Dosage History) and autonomously discovering the underlying biological laws governing drug metabolism.

### 🚩 The Problem
In clinical settings, doctors cannot blindly trust AI predictions ("Black Boxes"). They need to know *why* a specific dosage is recommended.

### ✅ The Solution
Genesis outputs **Symbolic Logic**—an algebraic formula that can be audited, verified, and understood by medical professionals before application.

---

## ✨ Key Features

*   **🧬 Evolutionary Search Engine:** Uses Darwinian Natural Selection (Crossover, Mutation, Elitism) to synthesize logic trees.
*   **🏥 Clinical Simulator:** Integrated environment to generate synthetic patient cohorts for validation.
*   **🔌 CSV Data Pipeline:** Support for ingesting external real-world datasets for analysis.
*   **🧊 3D Manifold Visualization:** Interactive 3D plotting to visualize the "Shape" of the discovered formula against ground truth data.
*   **🧮 Math Kernel:** Built-in SymPy integration to algebraically simplify and verify derived equations.
*   **📜 Flight Recorder:** Detailed logs of the evolutionary process, showing survivors, rejected trees, and breeding events per generation.

---

## 🛠️ Tech Stack

*   **Language:** Python 3.x
*   **Frontend:** Streamlit (Cyberpunk UI Theme)
*   **Data Processing:** Pandas, NumPy
*   **Visualization:** Plotly (3D), Graphviz (Tree Structures)
*   **Symbolic Math:** SymPy

---

## Installation Guide

### Prerequisites
*   Python 3.8 or higher installed.
*   (Optional but recommended) Graphviz system binary installed for tree visualization.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/genesis-clinical-ai.git
cd genesis-clinical-ai
```

### 2. Set up a Virtual Environment 
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install streamlit pandas plotly graphviz sympy
```
---
## Usage
Run the application using the Streamlit CLI:
```bash
streamlit run app.py
```
Once running:
*   **Select Mode:** Choose between "Run on Known Formula" (Simulation) or "Invent" (CSV Upload).
*   **Configure Scenario:** Select a medical scenario (e.g., Linear Dose vs. Geriatric Decay).
*   **Initiate Discovery:** Click the "Generate Formula" button to start the evolutionary engine.
*   **Verify:** Expand the "Deep Dive" sections to view 3D graphs, Logic Trees, and Algebraic proofs.

---
## Project Structure
```
Genesis_Project/
├── app.py              # Main Application (UI, State Management, Viz)
├── gp_engine.py        # Core Logic (Node Classes, Tree Generation, Recursion)
├── requirements.txt    # List of dependencies
└── README.md           # Documentation
```

