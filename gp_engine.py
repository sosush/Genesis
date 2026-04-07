import os
import random
import json
import subprocess
import sys
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GeneticArchitect:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def parse_problem(self, raw_text):
        """Concept: Natural Language Understanding (NLU)."""
        prompt = f"""
        Extract structured data from this algorithmic problem.
        Output ONLY a JSON object:
        {{
            "description": "Short logic summary",
            "examples": [
                {{"input": "code to define variables", "call": "method(vars)", "output": "expected_val"}}
            ],
            "constraints": ["list"]
        }}
        PROBLEM: {raw_text}
        """
        res = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)

    def evaluate_fitness(self, code, examples):
        """
        THE JUDGE: Real Fitness Evaluation via Sandbox Execution.
        This is the core AI 'Goal-Test'.
        """
        if not code or "class Solution" not in code: return 0.0
        passed = 0
        for ex in examples:
            test_script = f"""
import sys
{code}
try:
    sol = Solution()
    {ex['input']}
    result = sol.{ex['call']}
    if str(result) == str({ex['output']}):
        print("SUCCESS")
except Exception as e:
    print(f"ERROR: {{e}}")
"""
            with open("temp_exec.py", "w") as f: f.write(test_script)
            try:
                # Concept: Environment Interaction (Sandbox)
                res = subprocess.run([sys.executable, "temp_exec.py"], capture_output=True, text=True, timeout=2)
                if "SUCCESS" in res.stdout: passed += 1
            except: continue
        return passed / len(examples) if examples else 0.0

    def generate_primordial_soup(self, parsed_data, header):
        """Concept: Heuristic Population Seeding."""
        prompt = f"""
        Act as a Genetic Programming Seed Generator.
        Problem: {parsed_data['description']}
        Header: {header}
        Generate 4 diverse Python logic candidates (DP, Greedy, Stack, Recursive).
        Output ONLY the 4 code blocks separated by '---'. No text or backticks.
        """
        res = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model
        )
        candidates = res.choices[0].message.content.split('---')
        return [{"code": c.strip().replace("```python", "").replace("```", ""), "fitness": 0.0} for c in candidates[:4]]

    def smart_mutate(self, parent_code, feedback_type, registry):
        """Concept: Directed Mutation (Local Search)."""
        registry_str = "\n".join(registry)
        prompt = f"""
        PERFORM GENETIC MUTATION. 
        Mode: {feedback_type} | Failed Constraints: {registry_str}
        Current Code: {parent_code}
        Evolve logic to satisfy all constraints. Output ONLY code.
        """
        res = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model
        )
        return res.choices[0].message.content.strip().replace("```python", "").replace("```", "")