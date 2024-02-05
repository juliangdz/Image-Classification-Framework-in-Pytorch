import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_shape, output_shape, network_config):
        super(CNN, self).__init__()
        H, W, C = input_shape  # Unpacking the input shape

        self.layers = nn.ModuleList()
        current_channels = C  # Starting with the input channels
        fc_size = None  # To store the dynamically calculated size for the fully connected layer

        for layer_config in network_config['layers']:
            if layer_config['type'] == 'conv2d':
                self.layers.append(nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=layer_config['filters'],
                    kernel_size=layer_config['kernel_size'],
                    stride=layer_config['stride']
                ))
                # Adjust dimensions
                H = (H - layer_config['kernel_size']) // layer_config['stride'] + 1
                W = (W - layer_config['kernel_size']) // layer_config['stride'] + 1
                current_channels = layer_config['filters']
            elif layer_config['type'] == 'maxpool2d':
                self.layers.append(nn.MaxPool2d(
                    kernel_size=layer_config['pool_size'],
                    stride=layer_config['stride']
                ))
                # Adjust dimensions
                H = (H - layer_config['pool_size']) // layer_config['stride'] + 1
                W = (W - layer_config['pool_size']) // layer_config['stride'] + 1
            elif layer_config['type'] == 'flatten':
                self.layers.append(nn.Flatten())
                fc_size = H * W * current_channels  # Calculate the size for the fully connected layer
            elif layer_config['type'] == 'linear':
                if fc_size is None:
                    raise ValueError("FC layer size not calculated. Check the layer order.")
                # Use the calculated fc_size for in_features
                self.layers.append(nn.Linear(
                    in_features=fc_size,
                    out_features=layer_config['out_features']
                ))
                fc_size = None  # Reset fc_size if your network has multiple linear layers
                current_channels = layer_config['out_features']  # Update for potentially more linear layers
            elif layer_config['type'] == 'dropout':
                self.layers.append(nn.Dropout(layer_config['rate']))

        if fc_size is not None:  # If fc_size is calculated and not used, add the final layer
            self.layers.append(nn.Linear(fc_size, output_shape))
        else:
            self.layers.append(nn.Linear(current_channels, output_shape))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.log_softmax(x, dim=1)
