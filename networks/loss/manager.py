import torch.nn as nn

class LossFunctionManager:
    def __init__(self, config:dict):
        self.loss_func = self.get_loss_function(config)

    def get_loss_function(self, config):
        """
        Reads the config dict and returns the corresponding PyTorch loss function.
        """
        loss_name = config.get('loss_name', 'cross_entropy').lower()

        if loss_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'nll':
            return nn.NLLLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def compute_loss(self, predictions, targets):
        """
        Computes the loss using the specified loss function.
        """
        return self.loss_func(predictions, targets)

# Example usage
# loss_manager = LossFunctionManager(config)
# Assume some predictions and targets tensors
# loss = loss_manager.compute_loss(predictions, targets)
