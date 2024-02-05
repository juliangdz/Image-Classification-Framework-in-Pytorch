from networks.gradcam import GradCAM
import torchvision
import torch
import wandb
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and freed from memory."""
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    return buf

class WandBCallback:
    def __init__(self,config:dict):
        project = config['logging']['wandb']['project']
        wandb.login()
        wandb.init(
            project=project,
            config={
                "dataset":config['data']['dataset'],
                "transforms":config['data']['transforms']['options'][config['data']['transforms']['name']],
                "criterion":config['hyperparams']['loss'],
                "optimizer":config['hyperparams']['optimizer'],
                "network":config['hyperparams']['network'],
                "batch_size":config['hyperparams']['batch_size'],
                "epochs":config['hyperparams']['epochs'],
                "seed":config['hyperparams']['seed'],
                "network_config":config['networks'][config['hyperparams']['network']]
            }
        )

    def log(self, data, step):
        wandb.log(data, step=step)

    def log_images(self, images, labels, tag='samples'):
        images = [wandb.Image(image, caption=str(label)) for image, label in zip(images, labels)]
        wandb.log({tag: images}, step=0)
        
    def log_confusion_matrix(self, matrix, step):
        # Create a figure for the confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        # Display the confusion matrix
        ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=np.arange(matrix.shape[0])).plot(values_format='d', cmap='Blues', ax=ax)
        # Add titles and labels as needed
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        
        # Convert the matplotlib figure to an image that wandb can log
        fig.canvas.draw()  # Draw the figure
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Log the confusion matrix image to wandb
        wandb.log({"confusion_matrix": [wandb.Image(image, caption="Confusion Matrix")]}, step=step)
        
        # Close the figure to free memory
        plt.close(fig)
        
    def apply_gradcam_and_log_batch(self, model, images, device, step, target_layer=None, tag='GradCAM'):
        if hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
            # If the model has a 'features' attribute and it's an nn.Sequential,
            # assume it's a pretrained model and use the last layer from 'features'.
            target_layer = model.features[-1]
        elif target_layer is None:
            raise ValueError("Please specify the target_layer for custom models.")

        grad_cam = GradCAM(model, target_layer=target_layer)

        heatmaps = []
        for i in range(images.size(0)):  # Iterate through each image in the batch
            input_image = images[i].unsqueeze(0).to(device)  # Add batch dimension
            heatmap = grad_cam.generate_cam(input_image)
            # Normalize the heatmap for visualization
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            heatmaps.append(heatmap)

        # Convert the list of heatmaps to a tensor and create a grid
        heatmap_grid = torchvision.utils.make_grid(torch.stack(heatmaps), nrow=5)  # Adjust nrow as needed

        # Convert the heatmap grid to a NumPy array
        heatmap_grid_np = heatmap_grid.permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()

        # Log the heatmap grid to WandB as an image
        wandb.log({tag: [wandb.Image(heatmap_grid_np, caption=f'GradCAM Heatmaps for Step {step}')]}, step=step)
        
    def log_evaluation_images(self, images, predicted_labels, true_labels, tag='Eval Samples', step=0):
        images = [transforms.functional.to_pil_image(image) for image in images]
        logged_images = [wandb.Image(image, caption=f'Pred: {pred}, True: {true}') 
                         for image, pred, true in zip(images, predicted_labels, true_labels)]
        wandb.log({tag: logged_images}, step=step)

    def log_test(self, data, step):
        wandb.log(data, step=step)

    def finish(self):
        wandb.finish()