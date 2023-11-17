import logging
import os
import argparse
import tempfile

import torch
import torchvision


def load(output_folder: str):

    # Transformations to create tensors from image files and normalize results
    transformations = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307, ), (0.03081, ))
    ]
    transform = torchvision.transforms.Compose(transformations)

    # Download MNIST data to temporary directory
    with tempfile.TemporaryDirectory() as temporary_directory:
        train_data = torchvision.datasets.MNIST(root=temporary_directory,
                                                train=True, 
                                                transform=transform,
                                                download=True)
        test_data = torchvision.datasets.MNIST(root=temporary_directory,
                                            train=False, 
                                            transform=transform)
    
    # Ensure output directory exists
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    # Save train and test data into output directory
    torch.save(train_data, f'{output_folder}/train.pt')
    torch.save(test_data, f'{output_folder}/test.pt')

    return train_data, test_data
