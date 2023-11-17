import copy
import sys
import argparse
import logging

import torch

import controllers


if __name__ == '__main__':
    # Seed torch and set up device
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Configure logging
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(level=logging.NOTSET)

    # Create parser and controller subparsers
    parser = argparse.ArgumentParser(prog="MNIST CNN", 
                                     description="ML model example")
    subparsers = parser.add_subparsers()

    # Define load parser
    load_parser = subparsers.add_parser('load')
    load_parser.add_argument('--output-folder', 
                             type=str, 
                             default='data/', 
                             required=True)
    load_parser.set_defaults(controller=controllers.load)

    # Define train parser
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--data-folder', 
                              type=str, 
                              default='data/', 
                              required=True)
    train_parser.add_argument('--output-path', 
                              type=str, 
                              default='models/model.pth', 
                              required=True)
    train_parser.add_argument('--epoch-count', 
                              type=int,
                              default=1,
                              required=False)
    train_parser.add_argument('--batch-size', 
                              type=int,
                              default=100,
                              required=False)
    train_parser.add_argument('--learning-rate',
                              type=float, 
                              default=0.01,
                              required=False)
    train_parser.set_defaults(controller=controllers.train)
    train_parser.set_defaults(device=device)
    
    # Define evaluate parser
    evaluate_parser = subparsers.add_parser('evaluate')
    evaluate_parser.add_argument('--data-folder', 
                                 type=str, 
                                 default='data/', 
                                 required=True)
    evaluate_parser.add_argument('--model-path', 
                                 type=str, 
                                 default='models/model.pth', 
                                 required=True)
    evaluate_parser.add_argument('--batch-size', 
                                 type=int,
                                 default=100,
                                 required=False)
    evaluate_parser.set_defaults(controller=controllers.evaluate)
    evaluate_parser.set_defaults(device=device)

    # Parse arguments
    arguments = parser.parse_args(sys.argv[1:])

    parameters = copy.copy(vars(arguments))
    del parameters['controller']

    logging.debug(parameters)
    arguments.controller(**parameters)