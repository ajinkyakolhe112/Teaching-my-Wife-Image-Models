"""
Lecture 1: Lab - MNIST Digit Classification Practice
Students implement basic neural network components step by step
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# TODO 1: Complete the data loading function
def load_data():
    """Load MNIST dataset - STUDENT TASK"""
    transform = transforms.ToTensor()
    
    # TODO: Download MNIST dataset (hint: datasets.MNIST)
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # TODO: Create DataLoaders with batch_size=32
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

# TODO 2: Complete the neural network
class MNISTNet(nn.Module):
    """Simple neural network for MNIST - STUDENT TASK"""
    def __init__(self):
        super().__init__()
        # TODO: Define layers
        # Hint: nn.Flatten(), nn.Linear(784, ?), nn.ReLU(), nn.Linear(?, 10)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)  # TODO: Students fill in dimensions
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)   # TODO: Students fill in dimensions
    
    def forward(self, x):
        # TODO: Complete forward pass
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def visualize_data(train_loader):
    """Show sample images from dataset"""
    examples = next(iter(train_loader))
    images, labels = examples
    
    plt.figure(figsize=(12, 6))
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# TODO 3: Complete the training function
def train_one_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch - STUDENT TASK"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # TODO: Complete training step
        # 1. Zero gradients
        optimizer.zero_grad()
        
        # 2. Forward pass
        output = model(data)
        
        # 3. Calculate loss
        loss = criterion(output, target)
        
        # 4. Backward pass
        loss.backward()
        
        # 5. Update weights
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}: Loss={loss.item():.4f}')
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy

# TODO 4: Complete the test function
def test_model(model, test_loader):
    """Evaluate model - STUDENT TASK"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            # TODO: Complete evaluation
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100.0 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def main():
    """Main training loop"""
    print("=== MNIST Classification Lab ===")
    
    # Load data
    train_loader, test_loader = load_data()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Visualize some data
    visualize_data(train_loader)
    
    # Create model
    model = MNISTNet()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # TODO 5: Set up training components
    # Hint: Use SGD optimizer with lr=0.01, CrossEntropyLoss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(5):
        print(f"\nEpoch {epoch+1}/5")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Test
        test_acc = test_model(model, test_loader)
    
    # TODO 6: CHALLENGE - Improve the model
    print("\n=== CHALLENGE ===")
    print("Try to achieve >95% accuracy by:")
    print("1. Adding more hidden layers")
    print("2. Changing activation functions")
    print("3. Adjusting learning rate")
    print("4. Adding dropout for regularization")

if __name__ == "__main__":
    main()

# ANSWERS FOR INSTRUCTOR:
"""
TODO 1: datasets.MNIST, DataLoader with batch_size
TODO 2: nn.Linear(784, 128), nn.Linear(128, 10)
TODO 3: optimizer.zero_grad(), forward pass, loss.backward(), optimizer.step()
TODO 4: model.eval(), torch.no_grad(), calculate accuracy
TODO 5: torch.optim.SGD, nn.CrossEntropyLoss
TODO 6: Students experiment with architecture changes
"""