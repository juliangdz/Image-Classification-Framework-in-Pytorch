from data.dataset.mnist import load_mnist_dataset,infer_input_shape
from utils.helper import check_and_create_directory
from data.transforms import apply_transform

def load_dataset(config:dict):
    data_config = config['data']
    if data_config['dataset'] == 'mnist':
        data_dir = check_and_create_directory(data_config['data_directory'])
        
        transform_config = data_config["transforms"]["options"][data_config["transforms"]["name"]]
        transforms = apply_transform(transform_config)
        
        train_loader,val_loader,test_loader,num_classes = load_mnist_dataset(
            data_dir=data_dir,
            ratio=data_config['ratio'],
            batch_size=config['hyperparams']['batch_size'],
            transforms=transforms,
        )
        
        input_shape = infer_input_shape(train_loader)
        
        return train_loader,val_loader,test_loader,input_shape,num_classes
    else:
        raise ValueError(f"Dataset {data_config['dataset'] } not supported.")
    