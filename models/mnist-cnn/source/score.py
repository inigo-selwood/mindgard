import os
import logging
import json

import torch

import models
from infer import infer

model = None


def init():
    global model
    global device

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pth")
    model = models.CNN()
    model.load_state_dict(torch.load(model_path))


def run(raw_data):
    global model
    global device

    image = json.loads(raw_data)['data']
    image = torch.tensor(image)
    result = infer(image, model, device)

    payload = {
        'request': raw_data,
        'result': result,
    }
    logging.debug(payload)

    return result