import os
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def get_dataloaders(batch_size=64, data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR

    train_transform = T.Compose([
        T.RandomRotation(15),
        T.RandomAffine(0, translate=(0.1, 0.1)),
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    test_ds  = MNIST(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
