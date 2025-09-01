# Lecture 1 Quiz A: Technical Deep Dive
## Neural Network Fundamentals + MNIST

**Instructions**: Select the best answer for each question. Show your work for calculation problems.

---

### Question 1: Mathematical Foundations (10 points)
Given a neuron with weights w = [0.5, -0.3, 0.8] and bias b = 0.2, what is the output for input x = [1.0, 2.0, 0.5] using ReLU activation?

A) 0.3  
B) 0.5  
C) 0.0  
D) 1.1  

**Work Space:**
```
Calculation: z = w·x + b = ___________
ReLU(z) = max(0, z) = ___________
```

---

### Question 2: PyTorch Autograd (10 points)
What will be the value of `x.grad` after running this code?

```python
x = torch.tensor(3.0, requires_grad=True)
y = 2 * x ** 2 + x
y.backward()
print(x.grad)
```

A) 6  
B) 12  
C) 13  
D) 19  

**Explanation**: Derive the gradient step by step:
```
dy/dx = ___________
At x = 3: dy/dx = ___________
```

---

### Question 3: Architecture Analysis (15 points)
For the MNIST MLP with architecture 784 → 128 → 64 → 10:

**3a)** How many parameters are in the first hidden layer? (5 points)
- Weights: _______ × _______ = _______
- Biases: _______
- Total: _______

**3b)** What is the total number of trainable parameters in the entire network? (5 points)

**3c)** Why is the input layer size 784? (5 points)

---

### Question 4: Training Dynamics (15 points)

**4a)** What is the purpose of `optimizer.zero_grad()`? (5 points)

A) Reset the model weights to zero  
B) Clear accumulated gradients from previous iteration  
C) Set the learning rate to zero  
D) Initialize the optimizer state  

**4b)** In which order should these operations occur in the training loop? (5 points)
Number them 1-5:
- [ ] `loss.backward()`
- [ ] `optimizer.step()`
- [ ] `output = model(data)`
- [ ] `loss = criterion(output, target)`
- [ ] `optimizer.zero_grad()`

**4c)** Why do we need `model.eval()` during testing? (5 points)

---

### Question 5: Loss Functions (10 points)
For MNIST classification, we use CrossEntropyLoss. What would happen if we used MSELoss instead?

A) Training would be faster but less accurate  
B) Training would be slower and potentially unstable  
C) No difference in final performance  
D) The model couldn't learn at all  

**Justify your answer:**

---

### Question 6: Hyperparameter Analysis (15 points)

**6a)** A student's model has training accuracy of 99% but test accuracy of 60%. What is the most likely problem? (5 points)

**6b)** If the learning rate is too high, what symptoms would you observe? (5 points)

**6c)** How would you systematically determine the optimal batch size? (5 points)

---

### Question 7: Activation Functions (10 points)
Compare ReLU and Sigmoid activation functions:

**7a)** What is the derivative of ReLU for x > 0? (3 points)

**7b)** What is the derivative of ReLU for x < 0? (3 points)

**7c)** Why might ReLU be preferred over Sigmoid in deep networks? (4 points)

---

### Question 8: Implementation Problem (15 points)
Debug this PyTorch code that should create a simple neural network:

```python
import torch
import torch.nn as nn

class BuggyNet(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return torch.softmax(x)

model = BuggyNet()
```

**Identify and fix 3 bugs:**

**Bug 1:** _______________________________________________
**Fix:** ________________________________________________

**Bug 2:** _______________________________________________
**Fix:** ________________________________________________

**Bug 3:** _______________________________________________
**Fix:** ________________________________________________

---

### Bonus Question (5 points)
Derive the gradient of the loss function L = (y_pred - y_true)² with respect to the weight w, given that y_pred = w*x + b.

**Solution:**
```
∂L/∂w = ___________
```

---

## Answer Key

### Question 1: A) 0.3
**Solution:** z = 0.5(1.0) + (-0.3)(2.0) + 0.8(0.5) + 0.2 = 0.5 - 0.6 + 0.4 + 0.2 = 0.5
ReLU(0.5) = 0.5... Wait, let me recalculate: z = 0.5 - 0.6 + 0.4 + 0.2 = 0.5, so ReLU(0.5) = 0.5. Actually, B) 0.5 is correct.

### Question 2: C) 13
**Solution:** dy/dx = 4x + 1, at x=3: dy/dx = 4(3) + 1 = 13

### Question 3: 
**3a)** Weights: 784 × 128 = 100,352, Biases: 128, Total: 100,480
**3b)** Layer 1: 100,480, Layer 2: 128×64+64 = 8,256, Layer 3: 64×10+10 = 650, Total: 109,386
**3c)** 28×28 pixel MNIST images flattened to 1D vector

### Question 4:
**4a)** B) Clear accumulated gradients from previous iteration
**4b)** 5-3-4-1-2 (zero_grad → forward → loss → backward → step)
**4c)** Disables dropout and batch norm updates for consistent evaluation

### Question 5: B) Training would be slower and potentially unstable
MSE treats all classes equally, while CrossEntropy provides better gradient flow for classification.

### Question 6:
**6a)** Overfitting - model memorized training data
**6b)** Loss oscillates or diverges, gradients explode
**6c)** Grid search with cross-validation, monitoring training speed vs convergence

### Question 7:
**7a)** 1 **7b)** 0 **7c)** Avoids vanishing gradient problem, computationally efficient

### Question 8:
**Bug 1:** Missing super().__init__() call
**Bug 2:** Missing dim parameter in softmax
**Bug 3:** Should not use softmax with CrossEntropyLoss (applied internally)

### Bonus: ∂L/∂w = 2(y_pred - y_true) * x