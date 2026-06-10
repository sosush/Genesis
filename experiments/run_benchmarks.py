"""
Run benchmarks and compare Random vs Pure Evolutionary vs Neuro-Symbolic.
Saves results to a markdown table.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlflow
from src.synthesis.engine import SynthesisEngine, RandomSearchEngine
from benchmarks.arithmetic.problems import PROBLEMS as ARITH_PROBS
from benchmarks.sygus.string_problems import PROBLEMS as STRING_PROBS

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Genesis_Benchmarks")
    
    all_problems = {**ARITH_PROBS, **STRING_PROBS}
    
    results = []
    
    for name, (test_cases, variables) in all_problems.items():
        print(f"\nEvaluating: {name}")
        
        # 1. Random Search
        print("  Running Random Search...")
        with mlflow.start_run(run_name=f"{name}_random"):
            eng_rand = RandomSearchEngine(pop_size=100, max_gen=200, variables=variables)
            res_rand = eng_rand.run(test_cases)
            mlflow.log_params({"problem": name, "method": "random", "max_gen": 200})
            mlflow.log_metrics({"generations": res_rand.generations_taken, "fitness": res_rand.best_fitness})
        
        # 2. Pure Evolutionary
        print("  Running Pure Evolutionary...")
        with mlflow.start_run(run_name=f"{name}_evolution"):
            eng_evo = SynthesisEngine(pop_size=100, max_gen=200, use_neural_scorer=False, variables=variables)
            res_evo = eng_evo.run(test_cases)
            mlflow.log_params({"problem": name, "method": "evolutionary", "max_gen": 200})
            mlflow.log_metrics({"generations": res_evo.generations_taken, "fitness": res_evo.best_fitness})
            
        # 3. Neuro-Symbolic
        print("  Running Neuro-Symbolic...")
        with mlflow.start_run(run_name=f"{name}_neuro_symbolic"):
            eng_ns = SynthesisEngine(pop_size=100, max_gen=200, use_neural_scorer=True, variables=variables)
            res_ns = eng_ns.run(test_cases)
            mlflow.log_params({"problem": name, "method": "neuro_symbolic", "max_gen": 200})
            mlflow.log_metrics({"generations": res_ns.generations_taken, "fitness": res_ns.best_fitness})
            
        results.append({
            "problem": name,
            "random": res_rand.generations_taken if res_rand.best_fitness > 0.99 else "timeout",
            "evo": res_evo.generations_taken if res_evo.best_fitness > 0.99 else "timeout",
            "ns": res_ns.generations_taken if res_ns.best_fitness > 0.99 else "timeout"
        })
        
    # Write Markdown table
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_table.md", "w") as f:
        f.write("### Benchmark Results\n\n")
        f.write("| Problem | Random search | Pure evolutionary | Genesis (neuro-symbolic) |\n")
        f.write("|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['problem']} | {r['random']} gen | {r['evo']} gen | {r['ns']} gen |\n")
            
    print("\nDone. Results saved to results/benchmark_table.md")

if __name__ == "__main__":
    main()
