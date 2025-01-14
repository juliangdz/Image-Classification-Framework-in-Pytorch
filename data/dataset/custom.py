import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from data.stratify_logic import train_test_val_split

class CustomImageDataset(Dataset):
    """
    Example Custom dataset that expects a structure like:
    data_dir/class0/xxx.png
    data_dir/class0/xxy.png
    data_dir/class1/123.png
    ...
    """
    def __init__(self, phase='train', transforms=None, data_dir='./data', ratio=[0.8,0.1,0.1]):
        super().__init__()
        self.data_dir = os.path.join(data_dir,phase)
        self.transforms = transforms
        self.ratio = ratio

        # Collect all image paths and labels
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        classes = sorted(os.listdir(self.data_dir))
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            self.idx_to_class[idx] = cls
            class_folder = os.path.join(self.data_dir, cls)
            if os.path.isdir(class_folder):
                for image_name in os.listdir(class_folder):
                    if image_name.lower().endswith(('png', 'jpg', 'jpeg')):
                        image_path = os.path.join(class_folder, image_name)
                        self.samples.append((image_path, idx))
        print('Class Map (label): ',self.class_to_idx)
        print('Class Map (idx): ',self.idx_to_class)
        # Convert to dataset-like structure
        self.num_classes = len(self.class_to_idx)
        self.class_names = list(self.class_to_idx.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']

        if not isinstance(image, torch.Tensor):
            image = T.ToTensor()(image)

        return image/255., label
