from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_auc_score, classification_report)
import torch
import numpy as np

class MetricsManager:
    def __init__(self, config):
        self.metrics = self.get_metrics(config)
        self.last_clf_report = None

    def get_metrics(self, config):
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
            elif metric == 'auc':
                metric_functions.append(self.auc)
            elif metric == 'classification_report':
                self.clf_report
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

    def auc(self, predictions, targets):
        # predictions => Probability distributions from model
        preds = predictions.detach().cpu().numpy()
        targets = targets.cpu().numpy()

        num_classes = preds.shape[1]
        # Convert targets to one-hot
        one_hot_targets = np.eye(num_classes)[targets]
        
        try:
            # Check if targets have fewer than 2 unique classes
            if len(np.unique(targets)) < 2:
                # Return a default or skip
                return float('nan')  # or return 0.0, or skip entirely
            else:
                return roc_auc_score(
                    one_hot_targets, 
                    preds, 
                    average='macro', 
                    multi_class='ovr'
                )
        except Exception:
            return float("nan")

    def clf_report(self, predictions, targets):
        """
        Returns a 'dummy' float so it can be logged like other metrics,
        but we also store the classification report in self.last_clf_report
        for reference (or you can store it in a global or so).
        """
        preds = torch.argmax(predictions, dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        classification_report(targets, preds, output_dict=True)
        return  classification_report(targets, preds, output_dict=True)

    def compute_confusion_matrix(self, predictions, targets):
        return confusion_matrix(targets, predictions)

    def compute_metrics(self, predictions, targets, phase='Train'):
        results = {}
        for metric in self.metrics:
            val = metric(predictions, targets)
            # The function name is tricky to retrieve but we can do:
            metric_name = metric.__name__
            results[f'{phase}/{metric_name}'] = val
        return results