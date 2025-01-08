from networks.helper import *
from networks.loss.manager import LossFunctionManager
from networks.optim.manager import OptimizerManager
from networks.metrics.manager import MetricsManager
from networks.archs.CNN import CNN
from networks.archs.FCN import FCN
from networks.archs.LeNet import LeNet5
from networks.archs.TLManager import TLManager
from torchsummary import summary
from utils.helper import modify_config

def build_model(network_name, config,input_shape,num_classes):
    network_config = config['networks'][network_name]
    
    if network_name == 'fcn':
        model = FCN(
            input_shape=input_shape,#(1, 28, 28),  # or some placeholder
            output_shape=num_classes,#10,         # placeholder
            network_config=network_config
        )
    elif network_name == 'deep_cnn':
        model = CNN(
            input_shape=input_shape,#(1, 28, 28),  # or some placeholder
            output_shape=num_classes,#10,         # placeholder  
            network_config=network_config
        )
    elif network_name == 'lenet':
        model = LeNet5(
            input_shape=input_shape,#(1, 28, 28),  # or some placeholder
            output_shape=num_classes#10,         # placeholder
        )
    elif network_name == 'pretrained':
        pretrained_model_name = network_config['name']
        pretrained_model_manager = TLManager(
            model_name=pretrained_model_name,
            num_classes=num_classes
        )
        model = pretrained_model_manager.get_model()
    else:
        raise ValueError(f'{network_name} - Invalid Network Name')
    return model

def runner(config: dict, 
           tensorboard_cb, 
           wandb_cb, 
           train_loader, 
           val_loader, 
           test_loader, 
           input_shape, 
           num_classes, 
           device):
    
    network_name = config['hyperparams']['network']
    model = build_model(network_name, config, input_shape=input_shape, num_classes=num_classes)

    optimizer_manager = OptimizerManager(model, config['hyperparams']['optimizer'])
    optimizer_func = optimizer_manager.optimizer

    criterion_manager = LossFunctionManager(config['hyperparams']['loss'])
    criterion_func = criterion_manager.loss_func

    metrics_manager = MetricsManager(config['evaluation'])

    log_interval = config['logging']['interval']
    checkpoint_dir = config['checkpoints']['dir']
    early_stopping_patience = config['hyperparams'].get('early_stopping_patience', 5)

    model = model.to(device)
    
    # Print summary
    if network_name != 'fcn':
        summary(model,(input_shape[2],input_shape[0],input_shape[1]))
    else:
        summary(model, input_shape)

    # Log model architecture
    if network_name != 'fcn':
        tensorboard_cb.log_model_architecture(model, input_size=(1, input_shape[2], input_shape[0], input_shape[1]))
        wandb_cb.log_model_graph(model, input_size=(input_shape[2], input_shape[0], input_shape[1]))
    else:
        tensorboard_cb.log_model_architecture(model, input_size=input_shape)
        wandb_cb.log_model_graph(model, input_size=input_shape)

    # Train
    train(
        network_name=network_name,
        model=model,
        tensorboard_cb=tensorboard_cb,
        wandb_cb=wandb_cb,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer_func,
        criterion=criterion_func,
        metrics_manager=metrics_manager,
        epochs=config['hyperparams']['epochs'],
        device=device,
        log_interval=log_interval,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=early_stopping_patience
    )