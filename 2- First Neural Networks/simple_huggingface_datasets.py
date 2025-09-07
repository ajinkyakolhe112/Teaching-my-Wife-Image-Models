"""
Lecture 1: Neural Network Fundamentals + MNIST Digits
A simple, refactored implementation using Hugging Face datasets and Accelerate.
"""

import torch
from torch.utils.data import DataLoader
import datasets as huggingface_datasets
from torchvision import transforms

def load_mnist_data_from_huggingface():
    """Load MNIST dataset from Hugging Face"""
    # Load the dataset from the Hugging Face Hub
    dataset = huggingface_datasets.load_dataset("mnist")

    # Define a transformation to convert images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])

    def apply_transforms(examples):
        # Apply the tensor conversion
        for img in examples['image']:
            img = img.convert("RGB")
            img = transform(img)
            examples['pixel_values'].append(img)
        
        return examples

    # Apply the transformation to the entire dataset
    processed_dataset = dataset.with_transform(apply_transforms)
    
    # Create DataLoaders
    train_loader = DataLoader(processed_dataset['train'], batch_size=64, shuffle=True)
    test_loader = DataLoader(processed_dataset['test'], batch_size=64, shuffle=False)
    
    return train_loader, test_loader