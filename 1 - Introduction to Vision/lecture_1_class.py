"""
Lecture 1: Neural Network Fundamentals + MNIST Digits
Simple, clean implementation for teaching neural network basics
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchmetrics

def load_mnist_data():
    """Load and preprocess MNIST dataset"""
    transform = transforms.ToTensor()
    
    train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

class SimpleNeuralNet(nn.Module):
    """Simple MLP: 784 → 128 → 64 → 10"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

def train_model(model, train_loader, epochs=3):
    """Training loop"""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            
            # Forward pass
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}: Average Loss = {total_loss/len(train_loader):.4f}')

def evaluate_model(model, test_loader):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            predictions = model(x_batch)
            accuracy = torchmetrics.functional.accuracy(predictions, y_batch, task='multiclass', num_classes=10)
            correct += accuracy * len(y_batch)
            total += len(y_batch)
    
    print(f'Test Accuracy: {(correct/total)*100:.2f}%')

if __name__ == "__main__":
    # Load data
    train_loader, test_loader = load_mnist_data()
    
    # Create model
    model = SimpleNeuralNet()
    print(f'Model has {sum(p.numel() for p in model.parameters())} parameters')
    
    # Train and evaluate
    train_model(model, train_loader)
    evaluate_model(model, test_loader)