import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, input_shape: tuple, output_shape: int, network_config: dict) -> None:
        super(FCN, self).__init__()
        self.input_shape = input_shape
        input_features = torch.prod(torch.tensor(input_shape)).item()
        self.output_shape = output_shape

        # Dynamically create the layers based on the config
        layers = []
        filters = [input_features] + network_config['filters'] + [self.output_shape]
        
        for idx in range(1, len(filters)):
            layers.append(nn.Linear(filters[idx-1], filters[idx]))
            # Add a ReLU activation function between linear layers, except for the output layer
            if idx < len(filters) - 1:  # No ReLU after the last layer
                layers.append(nn.ReLU())
        
        # Using ModuleList to hold all layers
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, self.input_shape)
        for layer in self.layers:
            x = layer(x)
        return F.log_softmax(x, dim=1)
