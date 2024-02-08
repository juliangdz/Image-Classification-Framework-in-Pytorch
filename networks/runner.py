from networks.helper import *
from networks.loss.manager import LossFunctionManager
from networks.optim.manager import OptimizerManager
from networks.metrics.manager import MetricsManager
from networks.archs.CNN import CNN
from networks.archs.FCN import FCN
from networks.archs.TLManager import TLManager
from torchsummary import summary

def runner(config:dict,train_loader,val_loader,test_loader,input_shape,num_classes,tensorboard_cb,wandb_cb,device):
    network_name = config['hyperparams']['network']
    network_config = config['networks'][network_name]
    
    if network_name == 'fcn':
        model = FCN(
            input_shape=input_shape,
            output_shape=num_classes,
            network_config=network_config
        )
    elif network_name == 'deep_cnn':
        model = CNN(
            input_shape=input_shape,
            output_shape=num_classes,
            network_config=network_config
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

    summary(model,input_shape)
    
    optimizer_manager = OptimizerManager(model,config['hyperparams']['optimizer'])
    optimizer_func = optimizer_manager.optimizer
    
    criterion_manager = LossFunctionManager(config['hyperparams']['loss'])
    criterion_func = criterion_manager.loss_func
    
    metrics_manager = MetricsManager(config['evaluation'])
    
    log_interval = config['logging']['interval']
    
    model = model.to(device)
    
    tensorboard_cb.log_model_architecture(model, input_size=input_shape)
    wandb_cb.log_model_graph(model, input_size=input_shape)
    
    train(
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
        log_interval=log_interval
    )