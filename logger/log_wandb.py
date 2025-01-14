from networks.gradcam import GradCAM
import pdb
import torchvision
import torch
import torch.nn.functional as F
import wandb
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import pandas as pd 
from logger.model_logger import generate_model_graph, dot_to_image
import os
from io import BytesIO
from PIL import Image
import cv2

def plot_to_image(figure):
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    return buf


def classification_report_to_figure(report_dict, title="Classification Report"):
    """
    Converts a sklearn classification_report (as a dict) into a styled table
    on a Matplotlib figure. Returns the figure object.
    """
    rows = []
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            row = {
                "class": label,
                "precision": round(metrics.get("precision", 0.0),1),
                "recall": round(metrics.get("recall", 0.0),1),
                "f1-score": round(metrics.get("f1-score", 0.0),1),
                "support": round(metrics.get("support", 0),1)
            }
            rows.append(row)
        else:
            row = {
                "class": label,
                "precision": round(metrics,1),  # or store it under a specific column
                "recall": "",
                "f1-score": "",
                "support": ""
            }
            rows.append(row)

    df_report = pd.DataFrame(rows)

    # Create a figure
    fig, ax = plt.subplots(figsize=(8, len(df_report)*0.8))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Hide axes and just display a table
    ax.axis("tight")
    ax.axis("off")

    # Generate table from the DataFrame
    table = ax.table(
        cellText=df_report.values,
        colLabels=df_report.columns,
        loc="center"
    )
    # Adjust layout to make it more readable
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    fig.tight_layout()
    return fig

class WandBCallback:
    def __init__(self, config: dict):
        project = config['logging']['wandb']['project']
        experiment_name = config['experiment']['experiment_name']
        wandb.login()
        wandb.init(
            project=project,
            name=experiment_name,  # set the run name
            config={
                "dataset": config['data']['dataset'],
                "transforms": config['data']['transforms']['options'][config['data']['transforms']['name']],
                "criterion": config['hyperparams']['loss'],
                "optimizer": config['hyperparams']['optimizer'],
                "network": config['hyperparams']['network'],
                "batch_size": config['hyperparams']['batch_size'],
                "epochs": config['hyperparams']['epochs'],
                "seed": config['hyperparams']['seed'],
                "network_config": config['networks'][config['hyperparams']['network']]
            }
        )

    def log(self, data, step):
        wandb.log(data, step=step)

    def log_images(self, images, labels, tag='samples'):
        images = [wandb.Image(image, caption=str(label)) for image, label in zip(images, labels)]
        wandb.log({tag: images}, step=0)

    def log_confusion_matrix(self, matrix, step):
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=np.arange(matrix.shape[0]))\
            .plot(values_format='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        wandb.log({"confusion_matrix": [wandb.Image(image, caption="Confusion Matrix")]}, step=step)
        plt.close(fig)

    def log_model_graph(self, model, input_size, step=0):
        dummy_tensor = torch.rand(1, input_size[0], input_size[1], input_size[2]).cuda()
        dot = generate_model_graph(model, dummy_tensor)
        image_path = dot.render(format='png')
        wandb.log({"Model Graph": wandb.Image(image_path)}, step=step)

    def create_gradcam_overlay(self,original_img_np, heatmap_resized_np, alpha=0.6):
        """
        Creates a Grad-CAM overlay using OpenCV.

        Parameters:
        - original_img_np: Original image in [H, W, C] format, values in [0, 1].
        - heatmap_resized_np: Resized heatmap in [H, W], values in [0, 1].
        - alpha: Transparency factor for the heatmap.

        Returns:
        - overlay_bgr: BGR image with the heatmap overlay.
        """
        # Prepare the heatmap
        heatmap_np = np.uint8(255 * heatmap_resized_np)  # Convert to [0, 255] for OpenCV

        # Apply the colormap (e.g., JET)
        heatmap_colored = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)

        # Convert original image to uint8
        original_img_uint8 = np.uint8(255 * original_img_np)

        # Ensure the original image is in BGR format (if it's in RGB)
        original_img_bgr = cv2.cvtColor(original_img_uint8, cv2.COLOR_RGB2BGR)

        # Resize heatmap_colored to match original image size if necessary
        if heatmap_colored.shape[:2] != original_img_bgr.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, (original_img_bgr.shape[1], original_img_bgr.shape[0]))

        # Overlay the heatmap on the original image
        overlay = cv2.addWeighted(heatmap_colored, alpha, original_img_bgr, 1 - alpha, 0)

        return overlay
    
    def apply_gradcam_and_log_batch(self, model, images, device, step, tag='GradCAM',
                                    preds=None, targets=None, class_names=None):
        """
        Creates a side-by-side subplot of original image and GradCAM heatmap.
        Logs them to W&B, including captions with ground truth and predicted labels.
        
        Args:
            model: the PyTorch model
            images: a batch of images (B, C, H, W)
            device: torch device
            step: int, training step or epoch
            tag: str, a tag for logging to W&B
            preds: list or tensor of predicted labels (same length as images)
            targets: list or tensor of ground truth labels (same length as images)
            class_names: optional list of class name strings, indexed by label
        """
        # target_layer = model.layer4[-1].conv3
        # grad_cam = GradCAM(model,target_layer=target_layer)
        grad_cam = GradCAM(model)
        
        logged_images = []
        # Ensure we are not logging an overly large batch; 
        # you could trim if needed, e.g. max 16 images
        # batch_size = min(images.size(0), 16)
        batch_size = images.size(0)

        for i in range(batch_size):
            input_image = images[i].unsqueeze(0).to(device)
            
            # Generate the heatmap
            heatmap = grad_cam.generate_cam(input_image)
            heatmap_4d = heatmap.unsqueeze(0).unsqueeze(0) 
            # Upsample to the original (H_img, W_img) size using bilinear interpolation
            heatmap_resized = F.interpolate(
                heatmap_4d, 
                size=(images.shape[2], images.shape[3]),  # (H_img, W_img)
                mode='bilinear', 
                align_corners=False
            )
            # Squeeze back to [H_img, W_img]
            heatmap_resized = heatmap_resized.squeeze(0).squeeze(0)  # shape [H_img, W_img]
            heatmap_resized = heatmap_resized - heatmap_resized.min()
            if heatmap_resized.max() != 0:
                heatmap_resized = heatmap_resized / heatmap_resized.max()
            
            # Convert torch tensors to NumPy for matplotlib
            # Original image: (C, H, W) -> (H, W, C)
            original_img_np = images[i].permute(1, 2, 0).cpu().numpy()
            # If your image is in [0,1] or [-1,1], adjust as necessary
            # For standard [0,1] range, we can just do a clamp
            original_img_np = np.clip(original_img_np, 0, 1)
            
            # Create the Grad-CAM overlay using OpenCV
            overlay_bgr = self.create_gradcam_overlay(original_img_np, heatmap_resized.cpu().numpy(), alpha=0.6) 
            # Convert overlay from BGR to RGB for Matplotlib
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            
            # Create a Matplotlib figure with two subplots
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            
            # Subplot 1: Original image
            axs[0].imshow(original_img_np)
            axs[0].axis('off')
            axs[0].set_title("Original Image")
            
            # Option 2: Overlay heatmap on original image
            axs[1].imshow(overlay_rgb)
            axs[1].axis('off')
            axs[1].set_title("GradCAM")

            # Build caption: ground truth, predicted
            if targets is not None:
                gt_label = targets[i].item() if torch.is_tensor(targets) else targets[i]
            else:
                gt_label = "N/A"
            if preds is not None:
                pred_label = preds[i].item() if torch.is_tensor(preds) else preds[i]
            else:
                pred_label = "N/A"
            
            if class_names is not None:
                gt_label_str = class_names[gt_label] if gt_label != "N/A" else "N/A"
                pred_label_str = class_names[pred_label] if pred_label != "N/A" else "N/A"
            else:
                gt_label_str = str(gt_label)
                pred_label_str = str(pred_label)
            
            caption = f"GT: {gt_label_str} | Pred: {pred_label_str}"
            
            # Add caption as subplot suptitle, or just keep in mind for W&B
            fig.suptitle(caption)
            
            # Convert the Matplotlib figure to a PIL Image (for W&B logging)
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            
            pil_image = Image.open(buf).convert('RGB')
            logged_images.append(wandb.Image(pil_image, caption=caption))
        
        # Finally, log to W&B
        wandb.log({tag: logged_images}, step=step)

    def log_evaluation_images(self, images, predicted_labels, true_labels, tag='Eval Samples', step=0):
        images = [transforms.functional.to_pil_image(image) for image in images]
        logged_images = [wandb.Image(image, caption=f'Pred: {pred}, True: {true}') 
                         for image, pred, true in zip(images, predicted_labels, true_labels)]
        wandb.log({tag: logged_images}, step=step)
        
    def log_classification_report(self, clf_report, step,tag):
        """
        Create a Matplotlib figure from the classification report dict,
        then log it to W&B as an image.
        """
        if clf_report is None:
            return  # Nothing to log

        fig = classification_report_to_figure(clf_report, title="Classification Report")

        wandb.log(
            {
                f"{tag}/classification_report": [wandb.Image(fig, caption="Classification Report")]
            },
            step=step
        )

        plt.close(fig)  # Close the figure after logging

    def log_test(self, data, step):
        wandb.log(data, step=step)

    def finish(self):
        wandb.finish()
