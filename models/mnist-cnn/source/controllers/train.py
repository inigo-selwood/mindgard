import os
import argparse
import logging
import math

import torch

import models


def train(data_folder: str,
          output_path: str,
          device: str = 'cpu',
          epoch_count: int = 2,
          batch_size: int  = 100,
          learning_rate: float = 0.01):

    # Create data loader
    data = torch.load(f'{data_folder}/train.pt')
    loader = torch.utils.data.DataLoader(data, 
                                         batch_size=batch_size, 
                                         shuffle=True)
    
    # Instance model
    model = models.CNN()
    model.to(device)

    sample_count = epoch_count * len(data)

    # Train model
    model.train()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

    accumulator = 0
    index = 0

    for epoch in range(epoch_count):
        for step, (images, labels) in enumerate(loader):

            # Move datasets to device
            images = images.to(device)
            labels = labels.to(device)

            # Reset and forward pass through model
            optimizer.zero_grad()
            output = model(images)

            # Evaluate loss and back-propagate
            loss = torch.nn.functional.nll_loss(output, labels)
            loss.backward()
            optimizer.step()

            accumulator += batch_size
            if accumulator > sample_count / 100:
                accumulator -= sample_count / 100
                index += 1

                payload = {
                    'completion': index / 100,
                    'loss': f'{loss.item():.4f}'
                }
                logging.debug(payload)

    # Ensure model directory exists
    model_folder = os.path.dirname(output_path)
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)

    # Save model
    torch.save(model.state_dict(), output_path)

    return model
