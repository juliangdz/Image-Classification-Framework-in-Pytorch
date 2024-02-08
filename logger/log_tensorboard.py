import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from networks.gradcam import GradCAM
import torch

class TensorBoardCallback:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_training(self, loss, metrics:dict, step):
        self.writer.add_scalar('Loss/Train', loss, step)
        for key,value in metrics.items():
            self.writer.add_scalar(f'{key}/Train', value, step)

    def log_validation(self, loss, metrics:dict, step):
        self.writer.add_scalar('Loss/Val', loss, step)
        for key,value in metrics.items():
            self.writer.add_scalar(f'{key}/Val', value, step)
            
    def log_images(self, images, labels, tag='samples'):
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image(tag, img_grid)
    
    def log_evaluation_images(self, images, predicted_labels, true_labels, step, tag='Eval Samples'):
        # Convert the predicted and true labels into a grid of text labels
        label_grid = [f'Pred: {pred}, True: {true}' for pred, true in zip(predicted_labels, true_labels)]
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image(tag, img_grid, step)
        self.writer.add_text(tag + '_labels', ', '.join(label_grid), step)
    
    def log_model_architecture(self, model, input_size):
        example_input = torch.rand(input_size).to(next(model.parameters()).device)  
        self.writer.add_graph(model, example_input)

    def log_model_parameters(self, model, step):
        for name, param in model.named_parameters():
            self.writer.add_histogram(name, param, step)
            
    def log_test(self, loss, metrics:dict, step):
        self.writer.add_scalar('Loss/Test', loss, step)
        for key, value in metrics.items():
            self.writer.add_scalar(f'{key}/Test', value, step)
            
    def log_confusion_matrix(self, matrix, step):
        # Create a figure for the confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        # Display the confusion matrix
        ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=np.arange(matrix.shape[0])).plot(values_format='d', cmap='Blues', ax=ax)
        # Add titles and labels as needed
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        
        # Now, you can log the figure to TensorBoard
        self.writer.add_figure('Confusion Matrix', fig, step)

        # Close the figure to free memory
        plt.close(fig)
    
    def apply_gradcam_and_log_batch(self,model, images, device, step, tag='GradCAM'):
        # Assuming `target_layer` is your model's final convolutional layer
        grad_cam = GradCAM(model, target_layer=model.features[-1])  # Adjust `target_layer` as per your model architecture

        heatmaps = []
        for i in range(images.size(0)):  # Iterate through each image in the batch
            input_image = images[i].unsqueeze(0).to(device)  # Add batch dimension
            heatmap = grad_cam.generate_cam(input_image)
            # Normalize the heatmap for visualization
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            heatmaps.append(heatmap)

        # Convert the list of heatmaps to a tensor and create a grid
        heatmap_grid = torchvision.utils.make_grid(torch.stack(heatmaps), nrow=5)  # Adjust nrow as needed

        # Log the heatmap grid to TensorBoard
        self.writer.add_image(tag, heatmap_grid, global_step=step)
    
    def log_feature_maps(self, model, images, step, tag_prefix='FeatureMaps'):
        """
        Automatically detect convolutional layers in the model and log their feature maps.
        :param model: The model being evaluated.
        :param images: A batch of images to log feature maps for.
        :param step: The current step in training for logging.
        :param tag_prefix: Prefix for the TensorBoard tag.
        """
        model.eval()  # Ensure the model is in evaluation mode

        activations = []
        hooks = []

        # Function to register hooks on convolutional layers
        def register_hooks():
            for name, layer in model.named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    def hook(module, input, output, name=name):
                        # Normalize the output for better visualization
                        # Use global average pooling to reduce the size for visualization
                        output = torch.mean(output, dim=1, keepdim=True)
                        activations.append((name, output))
                    hooks.append(layer.register_forward_hook(hook))

        register_hooks()

        # Forward pass to trigger the hooks and capture activations
        with torch.no_grad():
            _ = model(images)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Log each captured activation
        for name, activation in activations:
            # Normalize and make grid
            grid = torchvision.utils.make_grid(activation, normalize=True, scale_each=True, nrow=4)
            self.writer.add_image(f'{tag_prefix}/{name}', grid, global_step=step)
        
        model.train()  # Set model back to train mode if necessary

    def close(self):
        self.writer.close()
