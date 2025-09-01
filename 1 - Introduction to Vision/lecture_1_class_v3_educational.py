"""
Lecture 1 - Version 3: Educational Focus
Clear explanations and step-by-step learning
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data():
    """Load MNIST dataset - handwritten digits 0-9"""
    print("Loading MNIST dataset...")
    
    # Convert images to tensors
    transform = transforms.ToTensor()
    
    train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    # Create batches for efficient training
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    print(f"Loaded {len(train_data)} training samples, {len(test_data)} test samples")
    return train_loader, test_loader

class DigitClassifier(nn.Module):
    """Neural network to classify handwritten digits"""
    
    def __init__(self):
        super().__init__()
        
        # Network architecture: 784 -> 128 -> 64 -> 10
        self.flatten = nn.Flatten()           # Convert 28x28 image to 784 pixels
        self.hidden1 = nn.Linear(784, 128)    # First hidden layer
        self.relu1 = nn.ReLU()                # Activation function
        self.hidden2 = nn.Linear(128, 64)     # Second hidden layer  
        self.relu2 = nn.ReLU()                # Activation function
        self.output = nn.Linear(64, 10)       # Output layer (10 digits)
        
        print("Created neural network: 784 -> 128 -> 64 -> 10")
        self.print_parameters()
    
    def forward(self, x):
        """Forward pass through the network"""
        x = self.flatten(x)      # Flatten image
        x = self.relu1(self.hidden1(x))  # First layer + activation
        x = self.relu2(self.hidden2(x))  # Second layer + activation
        x = self.output(x)       # Final output (no activation)
        return x
    
    def print_parameters(self):
        """Print number of learnable parameters"""
        total = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total:,}")

def train_model(model, train_loader, epochs=3):
    """Train the neural network"""
    print(f"\nStarting training for {epochs} epochs...")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Good for classification
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent
    
    model.train()  # Set to training mode
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass
            predictions = model(data)
            loss = criterion(predictions, target)
            
            # Backward pass
            optimizer.zero_grad()  # Clear gradients
            loss.backward()        # Compute gradients
            optimizer.step()       # Update weights
            
            running_loss += loss.item()
            
            # Print progress
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}')

def test_model(model, test_loader):
    """Test the trained model"""
    print("\nTesting model...")
    
    model.eval()  # Set to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():  # Don't compute gradients during testing
        for data, target in test_loader:
            predictions = model(data)
            predicted_labels = predictions.argmax(1)  # Get class with highest score
            
            total += target.size(0)
            correct += (predicted_labels == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {correct}/{total} = {accuracy:.2f}%')
    return accuracy

def demonstrate_prediction(model, test_loader):
    """Show how the model makes predictions"""
    print("\nDemonstrating prediction on one sample...")
    
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Take first image
    sample_image = images[0:1]  # Keep batch dimension
    actual_label = labels[0].item()
    
    # Make prediction
    with torch.no_grad():
        prediction = model(sample_image)
        predicted_label = prediction.argmax(1).item()
        confidence = torch.softmax(prediction, dim=1)[0][predicted_label].item()
    
    print(f"Actual digit: {actual_label}")
    print(f"Predicted digit: {predicted_label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Correct: {'Yes' if predicted_label == actual_label else 'No'}")

def main():
    """Main function to run everything"""
    print("=== MNIST Digit Classification ===")
    
    # Step 1: Load data
    train_loader, test_loader = load_data()
    
    # Step 2: Create model
    model = DigitClassifier()
    
    # Step 3: Train model
    train_model(model, train_loader, epochs=3)
    
    # Step 4: Test model
    accuracy = test_model(model, test_loader)
    
    # Step 5: Demonstrate prediction
    demonstrate_prediction(model, test_loader)
    
    print(f"\n=== Training Complete ===")
    print(f"Final accuracy: {accuracy:.2f}%")
    
    return model

if __name__ == "__main__":
    model = main()