import os
import argparse
import logging
import math

import torch

import models


def evaluate(data_folder: str,
          model_path: str,
          device: str = 'cpu',
          batch_size: int  = 100,):

    # Create data loader
    data = torch.load(f'{data_folder}/test.pt')
    loader = torch.utils.data.DataLoader(data, 
                                         batch_size=batch_size, 
                                         shuffle=True)
    
    # Instance model
    model = models.CNN().to(device)
    model.load_state_dict(torch.load(model_path))

    # Test model
    model.eval()
    correct = 0
    accumulator = 0
    index = 0
    with torch.no_grad():

        for step, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            loss = torch.nn.functional.nll_loss(output, labels)
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(labels.view_as(prediction)).sum().item()

            accumulator += batch_size
            if accumulator > len(data) / 100:
                accumulator -= len(data) / 100
                index += 1

                payload = {
                    'completion': index / 100,
                    'loss': f'{loss.item():.4f}'
                }
                logging.debug(payload)
    
    accuracy = correct / len(data)
    payload = {
        'accuracy': accuracy
    }
    logging.info(payload)

    return accuracy
