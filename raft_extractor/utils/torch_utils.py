import random
import numpy as np
import torch
def device():
    """
    Check if cuda is avaliable else choose the cpu
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu') # For testing purposes
    print(f"pyTorch is using {device}")
    return device


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_GPU():
    return torch.cuda.empty_cache()
