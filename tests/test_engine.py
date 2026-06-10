import pytest
from src.synthesis.engine import SynthesisEngine

def test_engine_identity():
    engine = SynthesisEngine(pop_size=50, max_gen=20, use_neural_scorer=False)
    test_cases = [({"x": i}, i) for i in range(5)]
    res = engine.run(test_cases)
    assert res.best_fitness > 0.9 # Should solve identity easily
