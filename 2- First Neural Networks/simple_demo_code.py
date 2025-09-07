import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # Initialize the dataset
        B, C, H, W = 32, 1, 28, 28
        self.x_actual = torch.randn(B, C, H, W)
        self.y_actual = torch.randint(0, 10, (B,))

    def __len__(self):
        # Return the length of the dataset
        return len(x_actual)
    
    def __getitem__(self, index):
        # Return the item at the given index
        return x_actual, y_actual
#%%
custom_dataset = CustomDataset()
print(len(custom_dataset))
print(custom_dataset[0])
#%%
custom_dataset_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=10, shuffle=True)
for batch_number, single_batch in enumerate(custom_dataset_loader):
    print(batch_number, single_batch)
#%%

class SimpleNeuralNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer      = torch.nn.Linear (in_features = 1*28*28 , out_features = 20)
        self.prediction_layer  = torch.nn.Linear (in_features = 20      , out_features = 10)

    def forward(self, single_batch):
        x = single_batch
        x = self.linear_layer(x)
        x = torch.nn.functional.relu(x)
        x = self.prediction_layer(x)

        return x

class SimpleConvolutionalNeuralNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution_layer = torch.nn.Conv2d (in_channels = 1       , out_channels = 10, kernel_size=(3,3))
        self.prediction_layer  = torch.nn.Linear (in_features = 10*26*26, out_features = 10)

    def forward(self, single_batch):
        x = single_batch
        x = self.convolution_layer(x)
        x = torch.nn.functional.relu(x)
        x = self.prediction_layer(x)

        return x
#%%
model = SimpleNeuralNet()
print(model)

#%%
model = SimpleConvolutionalNeuralNet()
print(model)
#%%

loss_function = torch.nn.CrossEntropyLoss()
optimizer     = torch.optim.SGD(model.parameters(), lr = 0.01)

for batch_number, single_batch in enumerate(custom_dataset_loader):
    x_actual, y_actual = single_batch

    optimizer.zero_grad()
    y_predicted = model(x_actual)
    loss        = loss_function(y_predicted, y_actual)
    loss.backward()
    optimizer.step()
    for parameter in model.parameters():
       parameter = parameter - parameter.grad * 0.01

    # OPTIMIZER.step() will update the parameters of the model
    # OPTIMIZER.zero_grad() will clear the gradients of the model
    # loss.backward() will compute the gradients of the model
    # parameter.grad will have gradient of each parameter wrt loss function
    # parameter = parameter - parameter.grad * 0.01 will update the parameters of the model

    print(f'Batch {batch_number}, Loss: {loss.item()}')
#%%