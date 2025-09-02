"""
Lecture 1 Lab: MNIST Neural Network Fundamentals - Comprehensive Homework Assignment
===================================================================================

This comprehensive lab covers essential concepts for neural network development:
1. Model Parameter Counting
2. Model Parameter Growth Analysis & Training Time Scaling
3. Loss Curve Analysis & Debugging
4. Mastering the Standard 4-Stage PyTorch Pipeline
5. Model Architecture Reporting with torchinfo

Note: GPU Memory Calculations are covered in a separate interactive HTML demo.

Complete all exercises and answer the questions in the designated sections.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np
from torchinfo import summary
import psutil
import os

# Set device for consistent results
device = torch.device('cuda' if torch.cuda.is_available() else 
                     'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# PART 1: MODEL PARAMETER COUNTING
# ============================================================================

class MNISTNet(nn.Module):
    """Configurable neural network for MNIST classification"""
    def __init__(self, hidden_sizes=[128]):
        """
        Args:
            hidden_sizes: List of hidden layer dimensions
        """
        super().__init__()
        self.hidden_sizes = hidden_sizes
        
        # Build network layers
        layers = []
        layers.append(nn.Flatten())
        
        # Input layer
        prev_size = 784  # 28x28 MNIST images
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append( nn.Linear(prev_size, hidden_size) )
            layers.append( nn.ReLU() )
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 10))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def count_parameters(self):
        """Count total parameters in the model"""
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        """Count only trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# EXERCISE 1.1: Parameter Counting
print("=" * 60)
print("EXERCISE 1.1: Manual Parameter Counting")
print("=" * 60)

# Test different architectures
architectures = [
    [128],
    [256],
    [128, 64],
    [512, 256, 128],
    [1024, 512, 256, 128]
]

def calculate_linear_layer_params(input_size, output_size):
    """
    TODO: Calculate parameters in a Linear layer
    Formula: weights + biases = (input_size * output_size) + output_size
    """
    # YOUR CODE HERE
    weights = input_size * output_size
    biases = output_size
    return weights + biases

def calculate_total_parameter_count(hidden_sizes):
    """
    TODO: Manually calculate total parameters for given architecture
    Architecture: 784 -> hidden_sizes[0] -> hidden_sizes[1] -> ... -> 10
    """
    # YOUR CODE HERE
    total_params = 0
    prev_size = 784
    
    # Count parameters for hidden layers
    for hidden_size in hidden_sizes:
        total_params = total_params + calculate_linear_layer_params(prev_size, hidden_size)
        prev_size    = hidden_size
    
    # Count parameters for output layer
    total_params += calculate_linear_layer_params(prev_size, 10)
    
    return total_params

print("Architecture -> Manual Count | PyTorch Count | Match?")
print("-" * 55)
for arch in architectures:
    model        = MNISTNet(arch)
    manual_count = calculate_total_parameter_count(arch)
    pytorch_count = model.count_parameters()
    match = "✓" if manual_count == pytorch_count else "✗"
    print(f"{str(arch):20} -> {manual_count:8,} | {pytorch_count:9,} | {match}")

# EXERCISE 1.2: GPU Memory Calculations
print("\n" + "=" * 60)
print("EXERCISE 1.2: GPU Memory Requirements")
print("=" * 60)

def calculate_model_memory_mb(num_parameters, precision='float32'):
    """
    TODO: Calculate model memory in MB
    
    Args:
        num_parameters: Number of model parameters
        precision: 'float32' (4 bytes) or 'float16' (2 bytes)
    
    Returns:
        Memory in MB
    """
    # YOUR CODE HERE
    bytes_per_param = 4 if precision == 'float32' else 2
    total_bytes = num_parameters * bytes_per_param
    mb = total_bytes / (1024 * 1024)
    return mb

def calculate_training_memory_mb(batch_size, sequence_length, hidden_size, num_layers):
    """
    TODO: Estimate training memory requirements
    
    Training memory includes:
    - Model parameters
    - Gradients (same size as parameters)  
    - Optimizer states (for Adam: 2x parameters for momentum and variance)
    - Activations (depends on batch size and architecture)
    """
    # Simplified calculation for dense networks
    # YOUR CODE HERE
    
    # Model parameters
    model_params = manual_parameter_count([hidden_size] * num_layers)
    model_memory = calculate_model_memory_mb(model_params)
    
    # Gradients (same size as model)
    gradient_memory = model_memory
    
    # Adam optimizer states (2x model size)
    optimizer_memory = model_memory * 2
    
    # Activations (rough estimate based on batch size and layers)
    activation_memory = (batch_size * hidden_size * num_layers * 4) / (1024 * 1024)
    
    total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory
    
    return {
        'model': model_memory,
        'gradients': gradient_memory,
        'optimizer': optimizer_memory,
        'activations': activation_memory,
        'total': total_memory
    }

# Test memory calculations
print("Architecture Memory Requirements:")
print("-" * 40)
for arch in architectures:
    params = manual_parameter_count(arch)
    memory = calculate_model_memory_mb(params)
    print(f"{str(arch):20} -> {params:8,} params -> {memory:6.2f} MB")

# QUESTION 1.1: Based on your calculations, which architecture would fit in:
print("\n" + "=" * 60)
print("QUESTION 1.1 ANSWERS:")
print("a) 2GB GPU can fit architectures up to: [YOUR ANSWER]")
print("b) 8GB GPU can fit architectures up to: [YOUR ANSWER]")
print("c) 24GB GPU can fit architectures up to: [YOUR ANSWER]")

# ============================================================================
# PART 2: MODEL PARAMETER GROWTH & TRAINING TIME ANALYSIS
# ============================================================================

print("\n" + "=" * 60)
print("EXERCISE 2: Parameter Growth & Training Time Scaling")
print("=" * 60)

def benchmark_training_time(model, train_loader, num_batches=100):
    """Benchmark training time for a model"""
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
            
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    return end_time - start_time

def load_mnist_data(batch_size=64):
    """Load MNIST dataset"""
    transform = transforms.ToTensor()
    train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Load data
train_loader, test_loader = load_mnist_data()

# Test different model sizes
growth_architectures = [
    [64],
    [128], 
    [256],
    [512],
    [1024],
    [2048]
]

print("Analyzing Parameter Growth vs Training Time:")
print("-" * 50)
print("Hidden Size | Parameters | Training Time (s) | Time/Param (μs)")
print("-" * 50)

results = []
for arch in growth_architectures:
    model = MNISTNet(arch)
    params = model.count_parameters()
    
    # Benchmark training time
    training_time = benchmark_training_time(model, train_loader, num_batches=50)
    time_per_param = (training_time / params) * 1_000_000  # microseconds
    
    results.append({
        'hidden_size': arch[0],
        'parameters': params,
        'training_time': training_time,
        'time_per_param': time_per_param
    })
    
    print(f"{arch[0]:10} | {params:10,} | {training_time:12.2f} | {time_per_param:11.2f}")

# EXERCISE 2.1: Analyze scaling relationship
print("\n" + "=" * 60)
print("EXERCISE 2.1: Scaling Analysis")
print("=" * 60)

def analyze_scaling():
    """
    TODO: Analyze the relationship between parameters and training time
    Calculate:
    1. Parameter growth ratio between consecutive models
    2. Training time growth ratio between consecutive models
    3. Determine if the relationship is linear, quadratic, or other
    """
    # YOUR CODE HERE
    print("Parameter Growth Ratios:")
    for i in range(1, len(results)):
        param_ratio = results[i]['parameters'] / results[i-1]['parameters']
        time_ratio = results[i]['training_time'] / results[i-1]['training_time']
        print(f"  {results[i-1]['hidden_size']} -> {results[i]['hidden_size']}: "
              f"Params x{param_ratio:.2f}, Time x{time_ratio:.2f}")
    
    return results

analyze_scaling()

# QUESTION 2.1: What type of relationship exists between model size and training time?
print("\nQUESTION 2.1: Relationship between model size and training time:")
print("Answer: [YOUR ANALYSIS HERE]")

# ============================================================================
# PART 3: LOSS CURVE ANALYSIS & DEBUGGING
# ============================================================================

print("\n" + "=" * 60)
print("EXERCISE 3: Loss Curve Analysis & Debugging")
print("=" * 60)

def train_with_logging(model, train_loader, test_loader, epochs=10, lr=0.01):
    """Train model with detailed logging for loss analysis"""
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Test phase
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_acc = 100.0 * test_correct / test_total
        
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        if epoch % 2 == 0:
            print(f'Epoch {epoch:2d}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%')
    
    return train_losses, train_accuracies, test_accuracies

def plot_training_curves(train_losses, train_accs, test_accs, title="Training Curves"):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(test_accs, label='Test Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# EXERCISE 3.1: Analyze different learning rate scenarios
print("\nTesting different learning rates:")

learning_rates = [0.001, 0.01, 0.1, 1.0]
lr_results = {}

for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    model = MNISTNet([128])
    losses, train_accs, test_accs = train_with_logging(model, train_loader, test_loader, 
                                                      epochs=8, lr=lr)
    lr_results[lr] = {
        'losses': losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'final_loss': losses[-1],
        'final_test_acc': test_accs[-1]
    }

# Plot comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for lr in learning_rates:
    plt.plot(lr_results[lr]['losses'], label=f'LR={lr}')
plt.title('Loss Curves by Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
for lr in learning_rates:
    plt.plot(lr_results[lr]['train_accs'], label=f'LR={lr}')
plt.title('Training Accuracy by Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
for lr in learning_rates:
    plt.plot(lr_results[lr]['test_accs'], label=f'LR={lr}')
plt.title('Test Accuracy by Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# QUESTION 3.1: Loss curve analysis
print("\n" + "=" * 60)
print("QUESTION 3.1: Loss Curve Debugging")
print("=" * 60)
print("Based on the loss curves above, answer the following:")
print("a) Which learning rate shows signs of being too high? [YOUR ANSWER]")
print("b) Which learning rate shows signs of being too low? [YOUR ANSWER]")
print("c) Which learning rate appears optimal? [YOUR ANSWER]")
print("d) What are signs of overfitting in the curves? [YOUR ANSWER]")

# ============================================================================
# PART 4: MASTERING THE STANDARD 4-STAGE PIPELINE
# ============================================================================

print("\n" + "=" * 60)
print("EXERCISE 4: Standard 4-Stage PyTorch Pipeline")
print("=" * 60)

class PyTorchPipeline:
    """Standard 4-stage PyTorch training pipeline"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Pipeline components
        self.train_loader = None
        self.test_loader = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        
        # Tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
    
    def stage_1_data_preparation(self, batch_size=64, data_augmentation=False):
        """
        STAGE 1: DATA PREPARATION
        TODO: Complete the data loading pipeline
        """
        print("Stage 1: Data Preparation")
        
        # YOUR CODE HERE - Define transforms
        if data_augmentation:
            train_transform = transforms.Compose([
                # TODO: Add data augmentation transforms
                transforms.ToTensor(),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(0.1),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # YOUR CODE HERE - Load datasets
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST('data', train=False, transform=test_transform)
        
        # YOUR CODE HERE - Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"  ✓ Training samples: {len(train_dataset)}")
        print(f"  ✓ Test samples: {len(test_dataset)}")
        print(f"  ✓ Batch size: {batch_size}")
        
        return self
    
    def stage_2_model_setup(self, optimizer_type='sgd', lr=0.01, weight_decay=0, 
                           loss_function='crossentropy', use_scheduler=True):
        """
        STAGE 2: MODEL & OPTIMIZER SETUP  
        TODO: Complete the model setup
        """
        print("Stage 2: Model & Optimizer Setup")
        
        # YOUR CODE HERE - Setup optimizer
        if optimizer_type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # YOUR CODE HERE - Setup loss function
        if loss_function.lower() == 'crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")
        
        # YOUR CODE HERE - Setup scheduler (optional)
        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        
        print(f"  ✓ Optimizer: {optimizer_type.upper()} (lr={lr})")
        print(f"  ✓ Loss function: {loss_function}")
        print(f"  ✓ Scheduler: {'Yes' if use_scheduler else 'No'}")
        print(f"  ✓ Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self
    
    def stage_3_training(self, epochs=10, verbose=True):
        """
        STAGE 3: TRAINING LOOP
        TODO: Complete the training loop
        """
        print("Stage 3: Training")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # YOUR CODE HERE - Complete training step
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Calculate metrics
            avg_train_loss = train_loss / len(self.train_loader)
            train_acc = 100.0 * correct / total
            
            # Validation phase
            test_loss, test_acc = self.stage_4_evaluation(verbose=False)
            
            # Store history
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            
            if verbose and epoch % 2 == 0:
                print(f'  Epoch {epoch+1:2d}/{epochs}: '
                      f'Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, '
                      f'Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%')
        
        return self
    
    def stage_4_evaluation(self, verbose=True):
        """
        STAGE 4: EVALUATION
        TODO: Complete the evaluation
        """
        if verbose:
            print("Stage 4: Evaluation")
        
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                # YOUR CODE HERE - Complete evaluation step
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_test_loss = test_loss / len(self.test_loader)
        test_acc = 100.0 * correct / total
        
        if verbose:
            print(f"  ✓ Final Test Loss: {avg_test_loss:.4f}")
            print(f"  ✓ Final Test Accuracy: {test_acc:.2f}%")
        
        return avg_test_loss, test_acc
    
    def plot_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['test_loss'], label='Test Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['test_acc'], label='Test Accuracy')
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# EXERCISE 4.1: Use the complete pipeline
print("\nDemonstrating complete 4-stage pipeline:")

# Create and run pipeline
model = MNISTNet([256, 128])
pipeline = PyTorchPipeline(model)

# Execute all 4 stages
pipeline.stage_1_data_preparation(batch_size=64, data_augmentation=True)
pipeline.stage_2_model_setup(optimizer_type='adam', lr=0.001, use_scheduler=True)
pipeline.stage_3_training(epochs=8, verbose=True)
pipeline.stage_4_evaluation(verbose=True)

# Plot results
pipeline.plot_history()


# ============================================================================
# HOMEWORK QUESTIONS & CHALLENGES
# ============================================================================

print("\n" + "=" * 60)
print("HOMEWORK QUESTIONS")
print("=" * 60)

print("""
SECTION A: PARAMETER CALCULATIONS
1. Calculate the exact number of parameters for a network: 784 → 512 → 256 → 128 → 10
   Answer: _____________

2. How much GPU memory would this model require for:
   a) Model weights only (float32): _________ MB
   b) Training (including gradients and Adam optimizer): _________ MB

3. If you double the size of each hidden layer, by what factor do the parameters increase?
   Answer: _____________

SECTION B: TRAINING ANALYSIS  
4. Based on your experiments, what happens when learning rate is:
   a) Too high (>0.1): _________________________________
   b) Too low (<0.001): _______________________________
   c) Just right (~0.01): ______________________________

5. Which phase takes longer: forward pass or backward pass? Why?
   Answer: _______________________________________________

SECTION C: ARCHITECTURE DESIGN
6. Design an architecture that has approximately 100K parameters for MNIST.
   Architecture: ______________________________________
   Actual parameters: _________________________________

7. What is the trade-off between model depth vs width?
   Answer: _______________________________________________

SECTION D: DEBUGGING
8. If your model achieves 10% accuracy on MNIST, what are 3 possible issues?
   a) ________________________________________________
   b) ________________________________________________  
   c) ________________________________________________

9. If training loss decreases but test accuracy doesn't improve, what's happening?
   Answer: _______________________________________________

SECTION E: BONUS CHALLENGES
10. Implement a model that achieves >98% accuracy on MNIST test set
11. Create a model that trains in under 30 seconds but still gets >95% accuracy
12. Design the most parameter-efficient model that achieves >97% accuracy

""")

print("=" * 60)
print("SUBMISSION INSTRUCTIONS")
print("=" * 60)
print("""
1. Complete all TODO sections in the code
2. Run the entire script and save outputs
3. Answer all homework questions above
4. For bonus challenges, create separate model classes
5. Submit your completed .py file with answers filled in

Good luck with your neural network fundamentals!
""")

if __name__ == "__main__":
    print("Lab completed! Review your outputs and complete the homework questions.")