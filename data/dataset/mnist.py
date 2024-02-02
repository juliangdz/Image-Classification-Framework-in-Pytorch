from torch.utils.data import Dataset,DataLoader
import numpy as np
import torchvision
from data.stratify_logic import train_test_val_split
import torch
import matplotlib.pyplot as plt
from torchvision import transforms as T
import pdb

__all__ = ["load_mnist_dataset","infer_input_shape"]

class MNIST(Dataset):
    def __init__(self,phase='train',transforms=None,data_dir:str='./data',ratio:list=[0.8,0.1,0.1]):
        super().__init__()
        self.mnist = torchvision.datasets.MNIST(root=data_dir,train=True,download=True)
        self.num_classes = self.__get_num_classes__()
        self.train_dataset,self.val_dataset,self.test_dataset = train_test_val_split(self.mnist,ratio)
        self.transforms = transforms
        if phase == 'val':
            self.dataset = self.val_dataset
        elif phase == 'test':
            self.dataset = self.test_dataset
        else:
            self.dataset = self.train_dataset
    
    def __get_num_classes__(self):
        unique_labels = set()
        for _, label in self.mnist:
            unique_labels.add(label)
        return len(unique_labels)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image.convert('RGB'))
        augmented  = self.transforms(image=image)
        image = augmented['image']
        if not isinstance(image, torch.Tensor):
            image = T.ToTensor()(image)
            
        return image/255., label
    
    
def load_mnist_dataset(data_dir:str,ratio:list[0.8,0.1,0.1],transforms=None,batch_size:int=64):
    train_dataset = MNIST(phase='train',transforms=transforms,data_dir=data_dir,ratio=ratio)
    val_dataset = MNIST(phase='val',transforms=transforms,data_dir=data_dir,ratio=ratio)
    test_dataset = MNIST(phase='test',transforms=transforms,data_dir=data_dir,ratio=ratio)
    
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    
    return train_loader,val_loader,test_loader,train_dataset.num_classes

def infer_input_shape(data_loader):
    # Fetch one batch of data
    images, _ = next(iter(data_loader))
    # Return the shape of a single image
    return images[0].size()[1],images[0].size()[2],images[0].size()[0]

def plot_distribution(dataset):
    counts = torch.zeros(10, dtype=torch.int32)
    for _, label in dataset:
        counts[label] += 1
    digits = range(10)
    plt.bar(digits, counts.numpy())
    plt.xlabel('Digits')
    plt.ylabel('Frequency')
    plt.title('Distribution of Digits in Dataset')
    plt.xticks(digits)
    plt.show()