from networks.gradcam import GradCAM
import pdb
import torchvision
import torch
import wandb
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import pandas as pd 
from logger.model_logger import generate_model_graph, dot_to_image
import os

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

    def apply_gradcam_and_log_batch(self, model, images, device, step, tag='GradCAM'):
        grad_cam = GradCAM(model)
        heatmaps = []
        for i in range(images.size(0)):
            input_image = images[i].unsqueeze(0).to(device)
            heatmap = grad_cam.generate_cam(input_image)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            heatmaps.append(heatmap)
        heatmap_grid = torchvision.utils.make_grid(torch.stack(heatmaps), nrow=5)
        heatmap_grid_np = heatmap_grid.mul(255).clamp(0, 255).byte().cpu().numpy()
        wandb.log({tag: [wandb.Image(hmap, caption=f'GradCAM Heatmaps') for hmap in heatmap_grid_np]}, step=step)

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
