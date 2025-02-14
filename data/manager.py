from data.dataset.mnist import load_mnist_dataset, infer_input_shape
from data.dataset.custom import CustomImageDataset  
from utils.helper import check_and_create_directory
from data.transforms import apply_transform
from torch.utils.data import DataLoader,WeightedRandomSampler
import numpy as np

def load_dataset(config: dict):
    data_config = config['data']
    data_dir = check_and_create_directory(data_config['data_directory'])
    
    transform_config = data_config["transforms"]["options"][data_config["transforms"]["name"]]
    transforms = apply_transform(transform_config)

    if data_config['dataset'].lower() == 'mnist':
        train_loader, val_loader, test_loader, num_classes = load_mnist_dataset(
            data_dir=data_dir,
            ratio=data_config['ratio'],
            batch_size=config['hyperparams']['batch_size'],
            transforms=transforms,
        )
        input_shape = infer_input_shape(train_loader)
        return train_loader, val_loader, test_loader, input_shape, num_classes,[0,1,2,3,4,5,6,7,8,9]
    
    elif data_config['dataset'].lower() == 'custom':
        # Example usage with CustomImageDataset
        # We create three separate datasets for train/val/test
        train_dataset = CustomImageDataset(phase='train', transforms=transforms, data_dir=data_dir, ratio=data_config['ratio'])
        val_dataset   = CustomImageDataset(phase='val',   transforms=transforms, data_dir=data_dir, ratio=data_config['ratio'])
        
        # Calculate Weights for Batch Sampling
        labels = [sample[1] for sample in train_dataset.samples]
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),  
            replacement=True
        )

        train_loader = DataLoader(train_dataset, batch_size=config['hyperparams']['batch_size'], shuffle=False,sampler=sampler)
        val_loader   = DataLoader(val_dataset,   batch_size=config['hyperparams']['batch_size'], shuffle=False)

        # For input shape, just get one batch from train_loader
        input_shape = infer_input_shape_custom(train_loader)  # You can create a function similar to `infer_input_shape` for custom data
        num_classes = train_dataset.num_classes
        class_names = train_dataset.class_names

        return train_loader, val_loader, val_loader, input_shape, num_classes,class_names
    
    else:
        raise ValueError(f"Dataset {data_config['dataset']} not supported.")


def infer_input_shape_custom(data_loader):
    # Fetch one batch of data
    images, _ = next(iter(data_loader))
    # Return the shape of a single image
    return images[0].size()[1], images[0].size()[2], images[0].size()[0]