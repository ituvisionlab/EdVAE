from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA

import pytorch_lightning as pl

def shift(x):
    return x - 0.5

class CelebAData(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.data_dir = config.data_dir
        self.img_size = config.img_size

    def train_dataloader(self):
        transform = [transforms.CenterCrop(140),
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Lambda(shift)]
        
        transform = transforms.Compose(transform)


        dataset = CelebA(root=self.data_dir, 
                        split="train", 
                        target_type="attr",
                        transform=transform, 
                        target_transform=None, 
                        download=False)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=self.num_workers,
                                pin_memory=True,
                                drop_last=True)
        
        return dataloader
    
    def val_dataloader(self):

        transform = [transforms.CenterCrop(140),
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Lambda(shift)]

        transform = transforms.Compose(transform)

        dataset = CelebA(root=self.data_dir, 
                        split="valid", 
                        target_type="attr",
                        transform=transform, 
                        target_transform=None, 
                        download=False)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True,
                                drop_last=True)
        
        return dataloader
    
    def test_dataloader(self):

        transform = [transforms.CenterCrop(140),
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Lambda(shift)]
        
        transform = transforms.Compose(transform)

        dataset = CelebA(root=self.data_dir, 
                        split="test", 
                        target_type="attr",
                        transform=transform, 
                        target_transform=None, 
                        download=False)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True,
                                drop_last=True)
        
        return dataloader