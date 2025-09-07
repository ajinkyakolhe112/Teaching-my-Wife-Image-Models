# with autograd cost of compile
# without autograd cost of compile
# intensity of autograd operation. how compute intensive

import torch
import torch.nn as nn

# Let's say we assume y_actual is the value we want for x_actual
x_actual = torch.randn(10, 4)
y_actual = torch.randn(10, 2)

# 1. Model Parameters 
model_with_random_parameters = nn.Sequential(
    nn.Linear( in_features = 4, out_features = 3),
    nn.Linear( in_features = 3, out_features = 2),
)
parameters_to_optimize = list(model_with_random_parameters.named_parameters())
for name, param in parameters_to_optimize:
    print(name, param.shape)
# torch.Size([3, 4]) # WEIGHTS  # torch.size([NEURONS_IN_LAYER = 3, Incoming_Features = 4])
# torch.Size([3])    # BIASES   # torch.size([NEURONS_IN_LAYER = 3])
# torch.Size([2, 3]) # WEIGHTS  # torch.size([NEURONS_IN_LAYER = 2, Incoming_Features = 3])
# torch.Size([2])    # BIASES   # torch.size([NEURONS_IN_LAYER = 2])

# Error => y_predicted - y_actual
y_predicted = model_with_random_parameters(x_actual)

# Compute the loss
loss = torch.nn.functional.mse_loss(y_predicted, y_actual)

# Compute the gradients
# Compute the gradients.
# There are two main ways to compute gradients:
# 1. `loss.backward()`: This is the most common method. It computes gradients and stores them 
#    in the .grad attribute of each parameter that requires gradients.
# 2. `torch.autograd.grad()`: This function returns a tuple of gradients and does NOT 
#    populate the .grad attributes of the parameters.
single_gradient = torch.autograd.grad(outputs= loss, inputs = parameters_to_optimize[0], retain_graph = True, allow_unused=True)
dError_dAllParameters   = torch.autograd.grad(outputs= loss, inputs = parameters_to_optimize, retain_graph = True ) # OR loss.backward(). Both do the same thing. 
loss.backward(retain_graph = True) # All gradients are computed in loss.backward()

# Example of Partial Derivative
# dError_dAllParameters  = torch.autograd.grad(
                                        # outputs = ERROR_FUNC( y_predicted , y_actual ),
                                        # inputs = parameters_to_optimize
                                        # )


# Update the model parameters
# Registering Model Parameters with Optimizer, as variables to be minimized
optimizer = torch.optim.Adam(params = parameters_to_optimize, lr = 0.01) 
optimizer.step() # WE DO NOT PASS THE GRADIENTS TO THE OPTIMIZER. STEP DOES IT INTERNALLY.

# OR
for parameter in parameters_to_optimize:
    # dERROR_dPARAMETER = torch.autograd.grad(outputs= loss, inputs = parameter)
    dERROR_dPARAMETER = parameter.grad
    parameter = parameter - dERROR_dPARAMETER * LEARNING_RATE