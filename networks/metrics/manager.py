from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import torch

class MetricsManager:
    def __init__(self, config):
        self.metrics = self.get_metrics(config)

    def get_metrics(self, config):
        """
        Reads the config dict and returns a list of metric functions to be used.
        """
        metrics_config = config.get('metrics', ['accuracy'])
        metric_functions = []
        for metric in metrics_config:
            if metric == 'accuracy':
                metric_functions.append(self.accuracy)
            elif metric == 'precision':
                metric_functions.append(self.precision)
            elif metric == 'recall':
                metric_functions.append(self.recall)
            elif metric == 'f1':
                metric_functions.append(self.f1)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        return metric_functions

    def accuracy(self, predictions, targets):
        preds = torch.argmax(predictions, dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        return accuracy_score(targets, preds)

    def precision(self, predictions, targets):
        preds = torch.argmax(predictions, dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        return precision_score(targets, preds, average='macro')

    def recall(self, predictions, targets):
        preds = torch.argmax(predictions, dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        return recall_score(targets, preds, average='macro')

    def f1(self, predictions, targets):
        preds = torch.argmax(predictions, dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        return f1_score(targets, preds, average='macro')
    
    def compute_confusion_matrix(self, predictions, targets):
        # Since predictions and targets are already numpy arrays, we can directly use them
        return confusion_matrix(targets, predictions)

    def compute_metrics(self, predictions, targets,phase='Train'):
        """
        Computes all the metrics specified in the config.
        """
        return {f'{phase}/{metric.__name__}': metric(predictions, targets) for metric in self.metrics}

# Example usage:
# config = {'metrics': ['accuracy', 'precision', 'recall']}
# metrics_manager = MetricsManager(config)
# metrics = metrics_manager.compute_metrics(predictions, targets)
