import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.model.eval()

        # If target_layer is not provided, attempt to automatically find it
        if target_layer is None:
            self.target_layer = self.find_last_conv_layer(self.model)
        else:
            self.target_layer = target_layer

        self.hook_layers()

    def find_last_conv_layer(self, model):
        # Reverse iterate through children (assuming the last conv layer is what we need)
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.modules.conv.Conv2d):
                return module
        raise ValueError("No convolutional layer found.")

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_backward_hook(hook_function)

    def generate_cam(self, input_image, target_class=None):
        model_output = self.model(input_image)
        if target_class is None:
            target_class = model_output.argmax(dim=1).item()
        self.model.zero_grad()
        class_loss = model_output[0, target_class]
        class_loss.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.target_layer(input_image).detach()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        return heatmap
