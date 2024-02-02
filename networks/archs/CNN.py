import torch 
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_shape: tuple, output_shape: int, network_config: dict) -> None:
        super(CNN, self).__init__()
        self.input_shape = input_shape
        layers = []

        current_channels = input_shape[2]  # Assuming input_shape is ( H, W,C)

        for block_config in network_config['cnn']:
            # Convolutional layer
            layers.append(nn.Conv2d(
                in_channels=current_channels,
                out_channels=block_config['filters'],
                kernel_size=block_config['kernel_size'],
                stride=block_config['stride'],
                padding=block_config['padding']
            ))
            current_channels = block_config['filters']

            # Normalization layer (if specified)
            if block_config.get('norm') == 'batch':
                layers.append(nn.BatchNorm2d(current_channels))

            # Activation - Assuming 'relu' for simplicity; extend as needed
            if block_config.get('activation') == 'relu':
                layers.append(nn.ReLU(inplace=True))

            # Pooling layer (if specified)
            if 'pool' in block_config:
                layers.append(nn.MaxPool2d(kernel_size=block_config['pool']))

        # Adaptive pooling layer to ensure the output is flattened to a vector
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.conv_layers = nn.Sequential(*layers)

        # Fully connected layer
        self.dropout = nn.Dropout(network_config['output_layer']['dropout'])
        self.fc = nn.Linear(current_channels, output_shape)  # current_channels is the output of the last conv layer

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.dropout(x)  # Apply dropout before the fully connected layer
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
