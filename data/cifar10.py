from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

import pytorch_lightning as pl

def shift(x):
    return x - 0.5

class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.data_dir = config.data_dir
    
    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Lambda(shift)])

        dataset = CIFAR10(root=self.data_dir, 
                        train=True, 
                        download=True, 
                        transform=transform)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=self.num_workers,
                                pin_memory=True,
                                drop_last=True)
        
        return dataloader
    
    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Lambda(shift)])
        
        dataset = CIFAR10(root=self.data_dir,
                        train=False,
                        transform=transform,
                        download=True)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True,
                                drop_last=True)
        
        return dataloader
    
    def test_dataloader(self):
        return self.val_dataloader()
        
        