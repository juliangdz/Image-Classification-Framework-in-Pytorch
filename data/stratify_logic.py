from torch.utils.data import Dataset,random_split

def train_test_val_split(dataset:Dataset,ratio:list):
    if sum(ratio) != 1:
        raise ValueError("The Sum of Stratify Ratio has to be 1")
    total_size = len(dataset)
    train_size = int(ratio[0]*total_size)
    val_size = int(ratio[1]*total_size)
    test_size = int(ratio[2]*total_size)
    
    train_dataset,val_dataset,test_dataset = random_split(dataset,[train_size,val_size,test_size])
    return train_dataset,val_dataset,test_dataset