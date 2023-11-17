import argparse
import logging

import torchvision
import torch

import models


def infer(image: torch.tensor, model: any, device: str = 'cpu'):

    model = model.to(device)

    # Load data
    image = torchvision.transforms.Normalize((0.1307, ), (0.03081, ))(image)
    batch = torch.stack([image])

    # Perform inference
    model.eval()
    with torch.no_grad():
        result = model(batch).argmax(dim=1, keepdim=True)[0][0].tolist()
        payload = {
            'result': result
        }
        logging.info(payload)
    
    return result
    

if __name__ == '__main__':
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(level=logging.NOTSET)

    parser = argparse.ArgumentParser(prog='MNIST', 
                                     description='Example MNIST CNN')
    parser.add_argument('--image-path', 
                        type=str, 
                        default='image.png', 
                        required=True)
    parser.add_argument('--model-directory', 
                        type=str, 
                        default='models/', 
                        required=True)
    arguments = parser.parse_args()

    # Load image
    image = torchvision.io.read_image(arguments.image_path)
    print(image.tolist()[0])
    import json
    thing = {
        'data': image.tolist()[0]
    }
    print(json.dumps(thing))
    # image = image.float()

    # Instance model
    model = models.CNN()
    model.load_state_dict(torch.load(f'{arguments.model_directory}/model.pth'))

    print(image.tolist()[0])

    # infer(image=image, 
    #       model=model, 
    #       device=device)