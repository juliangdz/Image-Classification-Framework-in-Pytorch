from networks.helper import *
from networks.loss.manager import LossFunctionManager
from networks.optim.manager import OptimizerManager
from networks.metrics.manager import MetricsManager
from networks.archs.CNN import CNN
from networks.archs.FCN import FCN
from networks.archs.TLManager import TLManager

def runner(config:dict,train_loader,val_loader,test_loader,input_shape,num_classes,tensorboard_cb,wandb_cb,device):
    network_name = config['hyperparams']['network']
    network_config = config['networks'][network_name]
    
    if network_name == 'fcn':
        model = FCN(
            input_shape=input_shape,
            output_shape=num_classes,
            network_config=network_config
        )
    elif network_config == 'cnn':
        model = CNN(
            input_shape=input_shape,
            output_shape=num_classes,
            network_config=network_config
        )
    elif network_config == 'pretrained':
        pretrained_model_name = network_config['pretrained']['name']
        pretrained_model_manager = TLManager(
            model_name=pretrained_model_name,
            num_classes=num_classes
        )
        model = pretrained_model_manager.get_model()
    else:
        raise ValueError(f'{network_name} - Invalid Network Name')
    
    optimizer_manager = OptimizerManager(model,config['hyperparams']['optimizer'])
    optimizer_func = optimizer_manager.optimizer
    
    criterion_manager = LossFunctionManager(config['hyperparams']['loss'])
    criterion_func = criterion_manager.loss_func
    
    metrics_manager = MetricsManager(config['evaluation'])
    
    log_interval = config['logging']['interval']
    
    model = model.to(device)
    
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