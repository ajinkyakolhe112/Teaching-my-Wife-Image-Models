# Lecture 1 Quiz B: Conceptual Understanding
## Neural Network Fundamentals + MNIST

**Instructions**: Choose the best answer and explain your reasoning where indicated.

---

### Question 1: Big Picture Understanding (10 points)
What is the main advantage of neural networks over traditional programming for image recognition?

A) They run faster than traditional algorithms  
B) They require less training data  
C) They can learn complex patterns automatically from examples  
D) They are easier to debug and understand  

**Explain your reasoning:**

---

### Question 2: Learning Process (15 points)
A neural network learns to recognize handwritten digits. Describe what happens during training using this analogy:

*"Training a neural network is like teaching a child to recognize numbers."*

**Fill in the analogy:**
- **Showing examples** corresponds to: ________________
- **Child makes guesses** corresponds to: ________________  
- **Correcting mistakes** corresponds to: ________________
- **Child gets better with practice** corresponds to: ________________

**Why is this analogy helpful for understanding neural networks?**

---

### Question 3: Architecture Intuition (15 points)
Our MNIST network has this structure: 784 → 128 → 64 → 10

**3a)** What does each number represent? (8 points)
- 784: ________________________________
- 128: ________________________________  
- 64: ________________________________
- 10: ________________________________

**3b)** Why do the numbers get smaller as we go deeper? (7 points)

---

### Question 4: Real-World Connection (10 points)
Besides handwritten digit recognition, what are three other problems that could be solved using similar neural network approaches?

1. _________________________________
2. _________________________________  
3. _________________________________

**For each example, explain why a neural network would be suitable:**

---

### Question 5: Problem Solving (15 points)
Your friend built a neural network for MNIST but it's only getting 60% accuracy. What are three questions you would ask to help debug the problem?

**Question 1:** ________________________________

**Question 2:** ________________________________

**Question 3:** ________________________________

**Why are these good diagnostic questions?**

---

### Question 6: Data Understanding (10 points)
Why do we split our MNIST data into training and test sets instead of using all the data for training?

A) Training on all data would take too long  
B) We need to test if the model can recognize new, unseen examples  
C) The computer can't handle that much data at once  
D) It's a tradition in machine learning  

**Explain using an everyday analogy:**

---

### Question 7: Learning Dynamics (15 points)
Imagine you're watching your neural network train. You see this progress:

- Epoch 1: 10% accuracy (random guessing)
- Epoch 3: 30% accuracy  
- Epoch 5: 70% accuracy
- Epoch 10: 95% accuracy
- Epoch 15: 95% accuracy
- Epoch 20: 95% accuracy

**7a)** What story does this progression tell? (8 points)

**7b)** What should you do at epoch 20 and why? (7 points)

---

### Question 8: Misconception Check (10 points)
Your classmate says: *"Neural networks are just fancy calculators that memorize the right answers."*

**Explain why this is incorrect and what neural networks actually do:**

**Use the MNIST example to support your explanation:**

---

### Bonus Question: Future Thinking (5 points)
If you wanted to recognize handwritten words (not just digits), what would need to change about our approach? Think about:
- Data requirements
- Architecture modifications  
- New challenges

**Your thoughts:**

---

## Rubric

### Excellent (A): 90-100 points
- Demonstrates deep conceptual understanding
- Makes clear connections between concepts and real-world applications
- Explanations are clear and use appropriate analogies
- Shows ability to troubleshoot and think critically about problems

### Good (B): 80-89 points  
- Shows solid understanding of main concepts
- Most explanations are clear with minor gaps
- Makes some connections to real-world applications
- Demonstrates basic problem-solving skills

### Satisfactory (C): 70-79 points
- Shows basic understanding of key concepts
- Explanations are partially correct but may lack depth
- Limited connections to broader applications
- Shows beginning problem-solving skills

### Needs Improvement (D/F): Below 70 points
- Shows limited understanding of concepts
- Explanations are unclear or incorrect
- Fails to make meaningful connections
- Little evidence of problem-solving ability

---

## Sample Answers

### Question 1: C) They can learn complex patterns automatically from examples
Neural networks can discover patterns that would be very difficult for humans to program explicitly.

### Question 2: 
- Showing examples → Forward pass with training data
- Child makes guesses → Model predictions  
- Correcting mistakes → Backpropagation and loss calculation
- Gets better with practice → Parameter updates through gradient descent

### Question 3:
**3a)** 784: Input pixels (28×28), 128: First hidden layer neurons, 64: Second hidden layer neurons, 10: Output classes (digits 0-9)
**3b)** Each layer extracts and combines features from the previous layer, creating increasingly abstract representations

### Question 6: B) We need to test if the model can recognize new, unseen examples
Like testing a student with new problems they haven't seen before to verify they truly understand the concept.

### Question 7:
**7a)** The network is learning successfully - starting from random guessing and improving rapidly, then plateauing when it reaches good performance
**7b)** Stop training (early stopping) since performance isn't improving and continued training might lead to overfitting