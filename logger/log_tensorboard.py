import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

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

    def close(self):
        self.writer.close()
