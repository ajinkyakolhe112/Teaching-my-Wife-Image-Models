# Lecture 1: Instructor Guide B - Interactive Discovery Approach
## Neural Network Fundamentals + MNIST

---

## Teaching Philosophy: Learning Through Discovery

### Core Principle
Instead of telling students how neural networks work, guide them to discover the concepts themselves through carefully designed activities and questions.

### Student-Centered Learning Objectives
By the end of class, students will have **discovered** (not just learned):
- Why we need multiple layers
- How learning happens through examples
- When to stop training
- What makes a good neural network architecture

---

## Interactive Opening: The Human Learning Experiment (15 minutes)

### Activity 1: Pattern Recognition Challenge
**Setup**: Display these quickly (2 seconds each):
```
[Show handwritten digits: 3, 8, 3, 5, 8, 3]
```

**Questions for Class**:
1. "How did you recognize these so quickly?"
2. "What if I showed you digits from a different handwriting style?"
3. "How did you learn to recognize numbers as a child?"

**Key Insight to Draw Out**: Humans learn patterns through repeated examples - just like neural networks!

### Activity 2: The Teaching Game
**Instructions**: 
- Pair students up
- One student draws digits while blindfolded by verbal instructions only
- Other student gives feedback: "make the curve rounder", "close the gap"
- Switch roles

**Debrief Questions**:
- "How did the feedback help improve the drawing?"
- "What if you could give thousands of pieces of feedback?"
- "What if a computer could learn this way?"

**Bridge to Neural Networks**: This is exactly what neural networks do - they get feedback (loss) and adjust their "drawing" (predictions)!

---

## Discovery Session 1: Building Intuition (20 minutes)

### The Mystery Box Experiment

**Setup**: Present this black box function:
```python
def mystery_function(x1, x2):
    # Hidden implementation
    result = 0.5 * x1 + 0.3 * x2 + 0.1
    return result if result > 0.4 else 0
```

**Student Challenge**: 
"Give me inputs, I'll give you outputs. Figure out what this function does!"

**Sample Interactions**:
- Student: "x1=1, x2=1" → You: "0.9"
- Student: "x1=0, x2=0" → You: "0"
- Student: "x1=0.5, x2=0.5" → You: "0"

**Discovery Questions**:
1. "What pattern do you notice?"
2. "Can you predict the output for x1=0.8, x2=0.2?"
3. "What would happen if I changed the numbers inside?"

**Reveal**: "Congratulations! You just reverse-engineered a neuron!"

### Connect to Real Neurons
```python
# Show them this is exactly what they figured out
def neuron(x1, x2):
    w1, w2, bias = 0.5, 0.3, 0.1  # weights and bias
    activation = w1*x1 + w2*x2 + bias
    return max(0, activation - 0.4)  # ReLU with threshold
```

---

## Interactive Demo: PyTorch Discovery (25 minutes)

### Discovery Challenge: Automatic Differentiation

**Stage 1: Manual Calculation**
```python
# Have students calculate by hand first
x = 2
y = x ** 2  # y = 4
# Question: "What's dy/dx when x=2?"
# Answer: 4
```

**Stage 2: The Magic Reveal**
```python
# Now show PyTorch magic
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(f"PyTorch says dy/dx = {x.grad}")
```

**Student Reaction**: "Wait, how did it know that?!"

**Group Discussion**: 
- "Why is this useful?"
- "What if we had millions of variables?"
- "How does this help with learning?"

### Guided Exploration: Building Complexity
```python
# Progress through increasingly complex examples
# Let students predict each result before running

# Example 1: Simple
x = torch.tensor(3.0, requires_grad=True)
y = 2 * x + 1
y.backward()
# Ask: "What should x.grad be?" (Answer: 2)

# Example 2: Chain rule
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y + 1
z.backward()
# Ask: "What should x.grad be?" (Answer: 4)

# Example 3: Multiple variables
x = torch.tensor(1.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)
y = w * x + 3
y.backward()
# Ask: "What are x.grad and w.grad?" (Answer: 2, 1)
```

---

## MNIST Exploration Adventure (30 minutes)

### Data Detective Activity

**Mystery Dataset Challenge**:
```python
# Load data but don't tell them what it is yet
dataset = datasets.MNIST('data', train=True, download=True)
sample_image, label = dataset[0]

# Show just the tensor
print(f"Shape: {sample_image.size()}")
print(f"Value range: {sample_image.min()} to {sample_image.max()}")
print(f"Label: {label}")
```

**Student Questions to Investigate**:
1. "What do you think this data represents?"
2. "What could the numbers mean?"
3. "How many different labels might there be?"

**The Big Reveal**:
```python
import matplotlib.pyplot as plt
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.title(f"This is a '{label}'")
plt.show()
```

### Data Exploration Stations

**Station 1: Class Distribution**
```python
# Students discover class balance
labels = [dataset[i][1] for i in range(1000)]
# Question: "Are all digits equally common?"
```

**Station 2: Variation Analysis**  
```python
# Show multiple examples of same digit
digit_3_examples = []
count = 0
for i, (img, label) in enumerate(dataset):
    if label == 3 and count < 9:
        digit_3_examples.append(img)
        count += 1
        
# Question: "How different can the same digit look?"
```

**Station 3: Difficulty Assessment**
```python
# Show hard-to-distinguish pairs
# Questions: "Which digits look most similar?"
# "Which would be hardest for a computer to distinguish?"
```

---

## Collaborative Architecture Design (25 minutes)

### The Architecture Workshop

**Challenge**: "Design a network to recognize these digits"

**Group Formation**: Teams of 3-4 students
**Materials**: Whiteboards, markers

**Guided Questions Sequence**:
1. "How many inputs do we need?" (Let them figure out 28×28=784)
2. "How many outputs do we need?" (10 for digits 0-9)
3. "What should we put in the middle?"

**Team Presentations** (5 minutes each):
- Each team presents their architecture
- Class votes on most creative design
- Instructor highlights key insights from each

**Synthesis**: 
"Let's combine the best ideas into our final architecture!"

### Interactive Architecture Building

```python
# Build the model piece by piece with student input
print("Let's build our neural network together!")

# Step 1
model = nn.Sequential()
# Ask: "What's our first step with the 28x28 image?"
model.add_module("flatten", nn.Flatten())

# Step 2  
# Ask: "How many neurons should our first layer have?"
# Take suggestions: 128, 256, 512, etc.
model.add_module("hidden1", nn.Linear(784, 128))

# Step 3
# Ask: "What comes after the linear layer?"
model.add_module("activation1", nn.ReLU())

# Continue building based on student input...
```

---

## Training Discovery Lab (35 minutes)

### Experiment-Driven Learning

**Phase 1: The Baseline Experiment**
```python
# Start with terrible parameters to show learning
model = create_simple_model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # Too small!

# Run one epoch
# Question: "What do you predict will happen?"
# Run and observe: likely very slow learning
```

**Phase 2: The Learning Rate Discovery**
```python
# Try different learning rates in real-time
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0]

for lr in learning_rates:
    print(f"\n--- Trying learning rate: {lr} ---")
    # Quick training with each LR
    # Have students predict what will happen before each
```

**Live Polling**: "Which learning rate worked best? Why?"

### Interactive Training Loop

**Step-by-Step Discovery**:
```python
# Break down the training loop - students guess each piece

# Question 1: "What's the first thing we need to do each iteration?"
optimizer.zero_grad()  # Clear old gradients

# Question 2: "What do we do with our inputs?"
outputs = model(inputs)  # Forward pass

# Question 3: "How do we measure if we're doing well?"
loss = criterion(outputs, targets)  # Compute loss

# Question 4: "How does the computer learn from mistakes?"
loss.backward()  # Backward pass

# Question 5: "How do we actually improve?"
optimizer.step()  # Update parameters
```

**Real-Time Monitoring**:
```python
# Watch the learning happen
for epoch in range(5):  # Just 5 epochs for demo
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # Training code...
        
        if i % 100 == 0:  # Every 100 batches
            print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
            
            # Interactive question
            input("Press Enter to continue training... (or type 'stop' to pause)")
```

---

## Discovery Assessment & Reflection (15 minutes)

### Peer Teaching Activity

**The Explain-It Challenge**:
- Divide class into 4 groups
- Each group gets 5 minutes to prepare a 2-minute explanation of:
  1. Group 1: "What is a neural network?" 
  2. Group 2: "How does learning work?"
  3. Group 3: "Why do we need multiple layers?"
  4. Group 4: "How do we know if our model is good?"

**Audience Participation**:
- Other students can ask clarifying questions
- Award points for best questions and clearest explanations

### Reflection Circle

**Questions for Discussion**:
1. "What surprised you most about neural networks today?"
2. "What was harder than you expected? Easier?"
3. "How would you explain this to a friend who wasn't here?"
4. "What questions do you still have?"

**Metacognitive Prompts**:
- "When did you feel most confused? What helped clear it up?"
- "What strategies did you use when your code didn't work?"
- "How did working in pairs/groups help your learning?"

---

## Interactive Assessment Techniques

### Real-Time Understanding Checks

**Traffic Light System**:
- Green card: "I understand and could explain this"
- Yellow card: "I mostly understand but have questions"  
- Red card: "I'm confused and need help"

**Use Throughout Class**:
- After each major concept
- Before moving to hands-on activities
- During coding sections

### Peer Instruction Questions

**Example Mid-Lecture Question**:
> "What happens if we remove all activation functions from our neural network?"
> 
> A) It trains faster
> B) It becomes equivalent to linear regression  
> C) It overfits more easily
> D) It can't learn at all

**Process**:
1. Students answer individually (30 seconds)
2. Discuss with neighbor (2 minutes)
3. Re-vote if needed
4. Brief explanation

### Think-Pair-Share Activities

**Example Discussion Prompts**:
- "Why might our model perform differently on training vs test data?"
- "How would you modify our approach for recognizing letters instead of digits?"
- "What real-world applications could use this technique?"

---

## Adaptive Instruction Strategies

### For Different Learning Styles

**Visual Learners**:
- Lots of plots and visualizations
- Network architecture diagrams
- Training curve animations

**Kinesthetic Learners**:
- Hands-on coding activities
- Physical analogies (committee voting)
- Interactive debugging sessions

**Auditory Learners**:
- Discussion-based discovery
- Peer explanation activities
- Think-aloud problem solving

### Differentiation Strategies

**Advanced Students**:
- Additional experiments to try
- Open-ended exploration questions
- Peer mentoring opportunities

**Struggling Students**:
- Extra scaffolding and templates
- Pair with stronger partners
- Focus on intuition before math

**English Language Learners**:
- Visual representations
- Glossary of key terms
- Peer translation support

---

## Technology Integration

### Interactive Notebooks
```python
# Use widgets for real-time parameter adjustment
import ipywidgets as widgets
from IPython.display import display

learning_rate_slider = widgets.FloatSlider(
    value=0.01,
    min=0.001,
    max=0.1,
    step=0.001,
    description='Learning Rate:'
)

# Let students adjust and see immediate results
def update_training(lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Train for a few epochs and show results
    
widgets.interact(update_training, lr=learning_rate_slider)
```

### Collaborative Tools
- Shared Google Colab notebooks
- Real-time polling with Kahoot/Poll Everywhere
- Padlet for collecting questions and insights
- Miro board for architecture brainstorming

---

## Assessment Throughout Class

### Formative Assessment Strategies

**Exit Ticket Questions**:
1. One thing you learned today
2. One thing you want to learn more about  
3. One thing that confused you

**Quick Polls**:
- "How confident do you feel about implementing a neural network?" (1-5 scale)
- "Which part of today's lesson was most valuable?"

**Misconception Checks**:
- Look for common errors in code
- Listen for incorrect explanations during peer teaching
- Address immediately through mini-lessons

### Summative Assessment Preview

**Portfolio Piece Assignment**:
"Create a 5-minute video explaining neural networks to someone who has never heard of them. Include a simple demo of your MNIST classifier."

**Rubric Focus Areas**:
- Conceptual understanding
- Technical implementation  
- Communication clarity
- Creative presentation

---

## Reflection & Iteration

### Post-Class Instructor Reflection

**What Worked Well**:
- Which discovery activities were most effective?
- Where did students show genuine excitement?
- What analogies resonated most?

**Areas for Improvement**:
- Which concepts needed more scaffolding?
- Where did pacing feel rushed or slow?
- How can we better support struggling students?

**Student Feedback Integration**:
- Review exit tickets for common themes
- Adjust next class based on lingering questions
- Celebrate student insights and contributions

### Iterative Improvements

**For Next Time**:
- Refine activities based on student engagement
- Add more/fewer interactive elements based on class dynamics
- Adjust timing based on actual vs planned duration
- Include more diverse examples if needed