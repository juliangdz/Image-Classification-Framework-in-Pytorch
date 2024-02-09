import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self,input_shape,output_shape):
        super(LeNet5, self).__init__()
        # Adjusted first convolutional layer to accept 3 input channels
        self.conv1 = nn.Conv2d(input_shape[2], 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Adjust pooling and convolution operations to handle the larger image size
        # Assuming two rounds of pooling reduce each spatial dimension by a factor of 4,
        # the size before the fully connected layer is computed accordingly.
        # Adjust the size of the first fully connected layer based on the output from the last pooling layer
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # Adjusted for the output size after pooling
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_shape)  # Assuming 10 classes as an example

    def forward(self, x):
        # Apply convolutions and max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, self.num_flat_features(x))
        # Fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
