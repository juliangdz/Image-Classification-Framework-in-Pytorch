import torch.optim as optim

class OptimizerManager:
    def __init__(self, model, config):
        self.optimizer = self.setup_optimizer(model, config)
        self.scheduler = self.setup_scheduler(config) if 'scheduler' in config else None

    def setup_optimizer(self, model, config):
        """
        Creates and returns the optimizer based on the configuration.
        """
        optimizer_name = config.get('optimizer_name', 'sgd').lower()
        lr = config.get('learning_rate', 0.001)
        weight_decay = config.get('weight_decay', 0.0005)  

        if optimizer_name == 'adamw':
            optimizer_params = {'params': model.parameters(), 'lr': lr,'weight_decay': weight_decay}
        else:
            optimizer_params = {'params': model.parameters(), 'lr': lr}

        if optimizer_name == 'sgd':
            optimizer_params['momentum'] = config.get('momentum', 0.9)
            return optim.SGD(**optimizer_params)
        elif optimizer_name == 'adam':
            optimizer_params['betas'] = config.get('betas', (0.9, 0.999))
            return optim.Adam(**optimizer_params)
        elif optimizer_name == 'adamw':
            optimizer_params['betas'] = config.get('betas', (0.9, 0.999))
            return optim.AdamW(**optimizer_params)
        elif optimizer_name == 'rmsprop':
            optimizer_params['alpha'] = config.get('alpha', 0.99)
            return optim.RMSprop(**optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def setup_scheduler(self, config):
        """
        Configures a learning rate scheduler if specified in the config.
        """
        scheduler_name = config['scheduler'].get('name', '').lower()
        scheduler_params = config['scheduler'].get('params', {})

        if scheduler_name == 'steplr':
            return optim.lr_scheduler.StepLR(self.optimizer, **scheduler_params)
        elif scheduler_name == 'exponentiallr':
            return optim.lr_scheduler.ExponentialLR(self.optimizer, **scheduler_params)
        elif scheduler_name == 'cosineannealinglr':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_params)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

# Example usage
config_with_scheduler = {
    'optimizer_name': 'adamw',
    'scheduler': {
        'name': 'steplr',
        'params': {'step_size': 10, 'gamma': 0.1}
    }
}

config_without_scheduler = {
    'optimizer_name': 'adamw',
    'learning_rate': 0.001
}

config_adamw_cosineannealing = {
    'optimizer_name': 'adamw',
    'learning_rate': 0.001,
    'scheduler': {
        'name': 'cosineannealinglr',
        'params': {
            'T_max': 50,      # Number of iterations (epochs) to restart or reduce LR
            'eta_min': 1e-5,  # Minimum learning rate
            'last_epoch': -1  # Sets the initial epoch. Use -1 if you want to start training fresh
        }
    }
}

# Assume 'model' is your neural network model
# optimizer_manager_with_scheduler = OptimizerManager(model, config_with_scheduler)
# optimizer_with_scheduler = optimizer_manager_with_scheduler.optimizer
# scheduler_with_scheduler = optimizer_manager_with_scheduler.scheduler

# optimizer_manager_without_scheduler = OptimizerManager(model, config_without_scheduler)
# optimizer_without_scheduler = optimizer_manager_without_scheduler.optimizer
