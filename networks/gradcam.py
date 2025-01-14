import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        # Will store activations and gradients
        self.activations = None
        self.gradients = None

        # Automatically find a convolution layer if none is specified
        if target_layer is None:
            self.target_layer = self.find_last_conv_layer(model)
        else:
            self.target_layer = target_layer

        # Register hooks
        self.hook_forward_backward()

    def find_last_conv_layer(self, model):
        # Reverse through all submodules
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
        raise ValueError("No convolutional layer found in the model.")

    def hook_forward_backward(self):
        # Forward hook: to store the *actual* activations of the target layer
        def forward_hook(module, input, output):
            self.activations = output

        # Backward hook: to store the gradients of the target layer
        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple, [0] is the actual gradients we need
            self.gradients = grad_out[0]

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(forward_hook)
        # Use register_full_backward_hook if available/preferred (PyTorch 1.10+)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class=None):
        """
        input_image: a 4D tensor [B, C, H, W]
        target_class: integer class index (or None, to use argmax)
        """

        # 1) Forward pass
        output = self.model(input_image)

        # If no target class provided, take the highest-scoring class for the first sample
        if target_class is None:
            # This picks from the *first* item in the batch
            target_class = output.argmax(dim=1).item()

        # 2) Zero grads and backward pass on the target class
        self.model.zero_grad()
        # If your batch size is > 1, make sure you handle that carefully
        loss = output[0, target_class]  # using first sample in batch
        loss.backward()

        # Now self.gradients and self.activations are populated

        # 3) Compute the weights: global-average-pool the gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # 4) Weight the channels of the stored activations by the pooled gradients
        #    self.activations shape: [B, #channels, H, W]
        #    pooled_gradients shape: [#channels]
        activations = self.activations[0].detach().clone()  # focusing on first sample if batch>1
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        # 5) Sum (or mean) across channels to get a single 2D map
        heatmap = torch.mean(activations, dim=0)  # shape: [H, W]

        # 6) ReLU and normalize
        heatmap = F.relu(heatmap)
        if torch.max(heatmap) != 0:
            heatmap /= torch.max(heatmap)

        return heatmap
