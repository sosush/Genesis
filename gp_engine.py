import os
import random
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GeneticArchitect:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def parse_problem(self, raw_text):
        """Natural Language Understanding: Extracting structural constraints."""
        prompt = f"""
        Extract structured data from this algorithmic problem text.
        Output ONLY a JSON object in this exact format:
        {{
            "description": "Logic summary",
            "examples": [
                {{"input": "raw_in", "output": "expected_out"}}
            ],
            "constraints": ["list"]
        }}
        
        PROBLEM TEXT:
        {raw_text}
        """
        res = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)

    def generate_primordial_soup(self, parsed_data, header):
        """Heuristic Population Seeding: Initializing the search space."""
        prompt = f"""
        Act as a Genetic Programming Seed Generator.
        Problem: {parsed_data['description']}
        
        CODE SIGNATURE:
        {header}
        
        Generate 4 diverse logical candidates (DP, Stack, Greedy, or Hash-Map).
        Output ONLY the 4 code blocks separated by '---'. 
        No markdown code fences, no text, no chatter.
        """
        res = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model
        )
        raw_candidates = res.choices[0].message.content.split('---')
        return [{"code": c.strip().replace("```python", "").replace("```", ""), "fitness": 0.0} for c in raw_candidates[:4]]

    def smart_mutate(self, parent_code, feedback_type, registry):
        """Global Constraint Satisfaction: Directed Local Search."""
        registry_str = "\n\n".join([f"CONSTRAINT {i+1}:\n{case}" for i, case in enumerate(registry)])
        
        prompt = f"""
        PERFORM GLOBAL GENETIC MUTATION.
        The current candidate failed multiple fitness tests.
        
        REQUIRED CONSTRAINTS:
        {registry_str}
        
        CURRENT CODE:
        {parent_code}
        
        TASK: Evolve a solution that satisfies ALL required constraints simultaneously. 
        Analyze the logic gap between 'Output' and 'Expected'.
        Output ONLY the updated code. No explanation.
        """
        res = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model
        )
        return res.choices[0].message.content.strip().replace("```python", "").replace("```", "")