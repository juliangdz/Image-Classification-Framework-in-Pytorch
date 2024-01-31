import torch
import numpy as np
import random
import json 

def read_config(config_path='config.json'):
    with open(config_path,"r") as cf:
        data = json.load(cf)
    return data

def modify_config(mod_config,config_path='config.json'):
    with open(config_path,"w") as cf:
        json.dump(mod_config,cf,indent=4)
    config = read_config(config_path=config_path)
    return config

def set_seed(seed_value):
    """
    Set seed for reproducibility.
    """
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # for multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)

def get_device():
    """
    Returns the device that PyTorch should use (CUDA if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
