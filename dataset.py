from torchvision.datasets import CIFAR10
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, ToTensor, Compose, Normalize
# from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import numpy as np

def get_normalization_parameters(dataset):
    stacked = [np.array(k[0]) for k in dataset]
    stacked = np.stack(stacked, axis=-1)
    means = stacked.mean(axis=(1, 2, 3))
    stds = stacked.std(axis=(1, 2, 3))
    return means, stds

class CIFAR10_wrapper(pl.LightningDataModule):
    def __init__(self, data_location, batch_size, workers):
        super().__init__()
        self.workers = workers
        self.path = data_location
        self.batch_size = batch_size

    def setup(self, stage=None):
        to_tensor = ToTensor()
        train_data = CIFAR10(self.path, train=True, download=True, transform=to_tensor)
        means, stds = get_normalization_parameters(train_data)
        normalization_transform = Normalize(mean=means, std=stds)
        augment_policy = AutoAugment(AutoAugmentPolicy.CIFAR10)

        transforms_train = Compose([
            augment_policy,
            to_tensor,
            normalization_transform
        ])

        transforms_test = Compose([
            to_tensor,
            normalization_transform
        ])

        train_data = CIFAR10(self.path, train=True, download=True, transform=transforms_train)
        self.cifar_test = CIFAR10(self.path, train=False, download=True, transform=transforms_test)
        self.cifar_train, self.cifar_val = random_split(train_data, (.90, .1))


    def train_dataloader(self):
        return DataLoader(self.cifar_train, self.batch_size, shuffle=True, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, self.batch_size, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, self.batch_size, num_workers=self.workers)