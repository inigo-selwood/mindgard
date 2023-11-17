import torch


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        layers = [
            torch.nn.Conv2d(in_channels=1, 
                      out_channels=32, 
                      kernel_size=3, 
                      stride=1),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=32, 
                      out_channels=64, 
                      kernel_size=3, 
                      stride=1),
            torch.nn.ReLU(),
            
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.25),
            torch.nn.Flatten(),

            torch.nn.Linear(in_features=9216, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(),

            torch.nn.Linear(in_features=128, out_features=10),
            torch.nn.LogSoftmax(dim=1)
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)