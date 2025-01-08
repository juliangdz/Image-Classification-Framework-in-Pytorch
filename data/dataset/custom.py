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
        self.data_dir = data_dir
        self.transforms = transforms
        self.ratio = ratio

        # Collect all image paths and labels
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        classes = sorted(os.listdir(data_dir))
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            self.idx_to_class[idx] = cls
            class_folder = os.path.join(data_dir, cls)
            if os.path.isdir(class_folder):
                for image_name in os.listdir(class_folder):
                    if image_name.lower().endswith(('png', 'jpg', 'jpeg')):
                        image_path = os.path.join(class_folder, image_name)
                        self.samples.append((image_path, idx))

        # Convert to dataset-like structure
        full_dataset = self.samples
        self.num_classes = len(self.class_to_idx)

        # Use train_test_val_split logic
        self.train_dataset, self.val_dataset, self.test_dataset = train_test_val_split(full_dataset, ratio)

        if phase == 'train':
            self.dataset = self.train_dataset
        elif phase == 'val':
            self.dataset = self.val_dataset
        else:
            self.dataset = self.test_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path, label = self.dataset[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']

        if not isinstance(image, torch.Tensor):
            image = T.ToTensor()(image)

        return image, label
