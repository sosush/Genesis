"""
Neural fitness scorer using PyTorch.

Extracts features from an expression tree (depth, size, operator counts)
and uses a small MLP to predict the fitness score (0.0 to 1.0).
This acts as a fast proxy to pre-filter candidate programs.

Feature vector (11 dims):
  depth, size, num_vars, num_consts,
  count(+), count(-), count(*), count(/), count(%),
  count(cmp), count(ternary)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import os

from ..evolution.population import Individual, Node, OPERATORS

FEATURE_DIM = 11  # Extended to cover cmp and ternary node types


def extract_features(ind: Individual) -> List[float]:
    """Extract a fixed-length feature vector from an expression tree."""
    nodes = ind.tree.all_nodes()

    depth = ind.tree.depth()
    size = len(nodes)
    num_vars = sum(1 for n in nodes if n.ntype == 'var')
    num_consts = sum(1 for n in nodes if n.ntype == 'const')
    num_cmp = sum(1 for n in nodes if n.ntype == 'cmp')
    num_ternary = sum(1 for n in nodes if n.ntype == 'ternary')

    op_counts = {op: 0 for op in OPERATORS}  # +, -, *, /, %
    for n in nodes:
        if n.ntype == 'op' and n.value in op_counts:
            op_counts[n.value] += 1

    features = [
        float(depth),
        float(size),
        float(num_vars),
        float(num_consts),
        float(op_counts.get('+', 0)),
        float(op_counts.get('-', 0)),
        float(op_counts.get('*', 0)),
        float(op_counts.get('/', 0)),
        float(op_counts.get('%', 0)),
        float(num_cmp),
        float(num_ternary),
    ]
    return features


class NeuralScorer(nn.Module):
    """
    A lightweight Multi-Layer Perceptron to predict fitness from tree features.
    """
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(FEATURE_DIM, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, FEATURE_DIM)
        Returns:
            Tensor of shape (batch_size, 1), values in [0, 1]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def predict(self, individuals: List[Individual]) -> List[float]:
        """Predict fitness for a list of individuals."""
        if not individuals:
            return []

        self.eval()
        with torch.no_grad():
            features = [extract_features(ind) for ind in individuals]
            x = torch.tensor(features, dtype=torch.float32)
            preds = self(x).squeeze(-1).tolist()
            if isinstance(preds, float):
                preds = [preds]
            return preds

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str):
        if os.path.exists(filepath):
            self.load_state_dict(torch.load(filepath))
