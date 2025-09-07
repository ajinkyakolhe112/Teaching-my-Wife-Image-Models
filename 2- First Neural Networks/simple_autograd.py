#%%
import torch

# EXAMPLE 1
# Create a tensor with gradient tracking enabled
x = torch.tensor(2.0, requires_grad=True)

# Define a simple function: y = x^2 + 3x + 1
y = x**2 + 3*x + 1

# Backpropagate (compute dy/dx)
y.backward()

# Gradient is stored in x.grad
print(f"x: {x.item()}")
print(f"y: {y.item()}")
print(f"dy/dx: {x.grad.item()}")

#%%
# EXAMPLE 2
# A vector of inputs
x = torch.randn(3, requires_grad=True)

# A simple function: y = sum(x^2)
y = (x**2).sum()

# Compute gradients
y.backward()

print("x:", x)
print("Gradient dy/dx:", x.grad)


# Why Autograd Matters
# Neural networks: Training requires gradients of the loss w.r.t. millions of parameters → Autograd handles this automatically.
# Efficiency: Optimized C++ backend with GPU acceleration.
# Integration: Works seamlessly with torch.optim for gradient-based optimization.


#%%
from torch.autograd import grad

x1 = torch.tensor(2, requires_grad=True, dtype=torch.float16)
x2 = torch.tensor(3, requires_grad=True, dtype=torch.float16)
x3 = torch.tensor(1, requires_grad=True, dtype=torch.float16)
x4 = torch.tensor(4, requires_grad=True, dtype=torch.float16)

x1, x2, x3, x4

f = x1 * x2 + x3 * x4

# f = x1 * x2 + x3 * x4
# f = 2 * 3 + 1 * 4
# df_dx1 = 3
# df_dx4 = 1

df_dx = grad(outputs = f, inputs = [x1, x2, x3, x4])
print(f'gradient of x1 = {df_dx[0]}')
print(f'gradient of x2 = {df_dx[1]}')
print(f'gradient of x3 = {df_dx[2]}')
print(f'gradient of x4 = {df_dx[3]}')

#%%
from torch.autograd import grad

x1 = torch.tensor(2, requires_grad=True, dtype=torch.float16)
x2 = torch.tensor(3, requires_grad=True, dtype=torch.float16)
x3 = torch.tensor(1, requires_grad=True, dtype=torch.float16)
x4 = torch.tensor(4, requires_grad=True, dtype=torch.float16)

x1, x2, x3, x4

f = x1 * x2 + x3 * x4

# f = x1 * x2 + x3 * x4
# f = 2 * 3 + 1 * 4
# df_dx1 = 3
# df_dx4 = 1

df_dx = grad(outputs = f, inputs = [x1, x2, x3, x4])
print(f'gradient of x1 = {df_dx[0]}')
print(f'gradient of x2 = {df_dx[1]}')
print(f'gradient of x3 = {df_dx[2]}')
print(f'gradient of x4 = {df_dx[3]}')