import torch 
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_shape: tuple, output_shape: int, network_config: dict) -> None:
        super(CNN, self).__init__()
        self.input_shape = input_shape
        input_channels = torch.prod(torch.tensor(input_shape)).item()
        layers = []
        current_channels = input_channels

        for block_config in network_config['cnn']:
            # Convolutional layer
            conv_layer = nn.Conv2d(
                in_channels=current_channels,
                out_channels=block_config['filters'],
                kernel_size=block_config['kernel_size'],
                stride=block_config['stride'],
                padding=block_config['padding']
            )
            layers.append(conv_layer)
            current_channels = block_config['filters']

            # Normalization layer (if specified)
            if block_config.get('norm') == 'batch':
                layers.append(nn.BatchNorm2d(block_config['filters']))

            # Activation
            layers.append(nn.ReLU())

            # Pooling layer (if specified)
            if 'pool' in block_config:
                layers.append(nn.MaxPool2d(kernel_size=block_config['pool']))

        # Adaptive pooling layer to ensure the output is flattened
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.conv_layers = nn.Sequential(*layers)

        # Fully connected output layer
        self.fc = nn.Linear(current_channels, output_shape)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
