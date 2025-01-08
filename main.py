from utils.helper import *
from data.manager import load_dataset
import argparse
from networks.runner import runner
from logger.log_tensorboard import TensorBoardCallback
from logger.log_wandb import WandBCallback

def main(args):
    config = read_config(args.config_path)
    # Set Seed and Device
    set_seed(config['hyperparams']['seed'])
    device = get_device()

    # Load Dataset
    train_loader, val_loader, test_loader, input_shape, num_classes = load_dataset(config)

    # Setup Loggers
    log_dir = check_and_create_directory(config['logging']['tensorboard']['logdir'])
    tensorboard_cb = TensorBoardCallback(log_dir=log_dir)
    wandb_cb = WandBCallback(config)
    
    config['data']['input_shape'] = input_shape
    config['data']['num_classes'] = num_classes
    modify_config(config,args.config_path)
    
    # Run
    runner(
        config=config,
        tensorboard_cb=tensorboard_cb,
        wandb_cb=wandb_cb,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_shape=input_shape,
        num_classes=num_classes,
        device=device
    )

    tensorboard_cb.close()
    wandb_cb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', default='config.json', type=str, help='Path to config.json')
    args = parser.parse_args()
    main(args)
