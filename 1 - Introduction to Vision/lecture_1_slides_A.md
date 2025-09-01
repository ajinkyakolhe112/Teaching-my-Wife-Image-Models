# Lecture 1: Neural Network Fundamentals + MNIST
## Technical Deep Dive Approach

---

## Learning Objectives
- Understand mathematical foundations of neural networks
- Implement forward and backward propagation from scratch
- Master PyTorch tensor operations and autograd
- Build and train a multi-layer perceptron for MNIST

---

## Neural Network Mathematics

### The Neuron Model
```
y = f(∑(wi * xi) + b)
```

Where:
- `wi`: weights
- `xi`: inputs  
- `b`: bias
- `f`: activation function

---

## Matrix Operations in Neural Networks

### Forward Pass
```python
# Layer computation
z = W @ x + b
a = activation(z)

# Full network
a1 = σ(W1 @ x + b1)
a2 = σ(W2 @ a1 + b2)
y = W3 @ a2 + b3
```

---

## Gradient Descent Mathematics

### Loss Function
```
L = 1/2 * ||y_pred - y_true||²
```

### Parameter Updates
```
W := W - α * ∂L/∂W
b := b - α * ∂L/∂b
```

Where `α` is the learning rate

---

## PyTorch Autograd System

### Computational Graph
```python
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
z = 2 * y
z.backward()
print(x.grad)  # 4.0
```

### Gradient Flow
- Forward: compute outputs
- Backward: compute gradients using chain rule

---

## MNIST Dataset Analysis

### Data Properties
- 60,000 training images (28×28 pixels)
- 10,000 test images
- 10 classes (digits 0-9)
- Grayscale values [0, 255]

### Preprocessing
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

---

## Network Architecture Design

### MLP Architecture: 784 → 128 → 64 → 10

```python
class MLP(nn.Module):
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
```

---

## Training Loop Implementation

### Core Training Steps
1. **Forward Pass**: Compute predictions
2. **Loss Calculation**: Compare with ground truth
3. **Backward Pass**: Compute gradients
4. **Parameter Update**: Apply gradient descent

```python
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

---

## Optimization Techniques

### Optimizers Comparison
- **SGD**: Basic gradient descent
- **Adam**: Adaptive learning rates
- **AdamW**: Adam with weight decay

### Learning Rate Scheduling
```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.1
)
```

---

## Performance Metrics

### Classification Metrics
- **Accuracy**: Correct predictions / Total predictions
- **Loss**: Cross-entropy for multi-class classification
- **Per-class Performance**: Confusion matrix analysis

### Monitoring Training
```python
def evaluate(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total
```

---

## Implementation Best Practices

### Model Architecture
- Start simple, add complexity gradually
- Use appropriate activation functions
- Consider batch normalization for deeper networks

### Training Strategy
- Monitor both training and validation loss
- Use early stopping to prevent overfitting
- Save model checkpoints regularly

---

## Lab Exercise

### Build MNIST Classifier
1. Load and preprocess MNIST dataset
2. Design MLP architecture
3. Implement training loop
4. Achieve >95% accuracy
5. Analyze model performance

### Expected Results
- Training accuracy: >99%
- Test accuracy: >95%
- Training time: ~5 minutes on CPU

---

## Mathematical Foundations Review

### Chain Rule Application
```
∂L/∂W1 = ∂L/∂a2 * ∂a2/∂z2 * ∂z2/∂a1 * ∂a1/∂z1 * ∂z1/∂W1
```

### Activation Functions
- **ReLU**: `f(x) = max(0, x)`
- **Sigmoid**: `f(x) = 1/(1 + e^(-x))`
- **Softmax**: `f(xi) = e^xi / ∑e^xj`

---

## Summary

### Key Concepts Mastered
✓ Neural network mathematical foundations  
✓ PyTorch tensor operations and autograd  
✓ MLP architecture design and implementation  
✓ Training loop optimization  
✓ Performance evaluation metrics  

### Next Lecture Preview
Convolutional Neural Networks for image processing