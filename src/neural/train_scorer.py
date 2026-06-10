"""
Training loop for the neural scorer.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List

from .scorer import NeuralScorer, extract_features
from ..evolution.population import Individual

def train_scorer(
    model: NeuralScorer,
    individuals: List[Individual],
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.01
) -> float:
    """
    Fine-tune the neural scorer on a batch of evaluated individuals.
    
    Args:
        model: The NeuralScorer instance
        individuals: List of individuals that have their .fitness evaluated
        
    Returns:
        Final training loss
    """
    if not individuals:
        return 0.0
        
    features = [extract_features(ind) for ind in individuals]
    targets = [ind.fitness for ind in individuals]
    
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / len(dataloader)
        
    return final_loss
