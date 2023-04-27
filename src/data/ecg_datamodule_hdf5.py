from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T

import os
import pandas as pd
import numpy as np
from PIL import Image
import h5py


from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity

"""This implementation assumes that the png files are located in a subfolder called "images" inside the `data_dir` 
directory. The `ECGDataset` class loads the images and their corresponding labels from the csv file, and transforms 
the images using `PIL` and `torchvision` transforms. In the `setup` method, the dataset is split into train, 
validation, and test sets using the `random_split` method from PyTorch. Finally, the `train_dataloader`, 
`val_dataloader`, and `test_dataloader` methods return PyTorch `DataLoader` objects that can be used to iterate over 
the datasets during training and testing. The `teardown` method is currently empty, as no cleanup is needed after 
training or testing. Note that you may need to modify the transforms used for the train and test datasets, 
or add additional arguments to the `ECGDataset` class depending on the characteristics of your data.

"""


class ECGDataset(Dataset):
    def __init__(self, data_dir, transform=None, classification_category='AF'):
        self.data_dir = data_dir
        self.transform = transform
        self.classification_category = classification_category
        self.data = self.load_data()

    def load_data(self):
        # load the hdf5 file
        with h5py.File(os.path.join(self.data_dir, 'hdf5/data.hdf5'), 'r') as f:
            # load the labels
            labels = f['labels'][:]
            # load the image names
            img_names = [f['img_names'][i].decode('utf-8') for i in range(len(f['img_names']))]
            # load the images
            images = f['images'][:]
            # load the label names
            # print(f'label names: {f["label_names"][:]}')
            label_names = f["label_names"][:]

        return list(zip(images, labels, label_names, img_names))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        ecg, labels, label_names, img_name = self.data[index]
        labels = labels[:, label_names.index(self.classification_category)]
        labels = torch.tensor(labels, dtype=torch.long)

        if self.transform is not None:
            ecg = self.transform(ecg)

        return ecg, labels


class ECGDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/",
            train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            data_path=None,
            classification_path=None,
            classification_category=None,
            dataset=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(224),
            T.Normalize((0.5,), (0.5,))
        ])
        self.classification_category = classification_category

    def prepare_data(self):
        # download data, pre-process, split, save to disk, etc...
        pass

    def setup(self, stage=None):
        # load data, set variables, etc...
        ecg_dataset = ECGDataset(self.data_dir, transform=self.transform,
                                 classification_category=self.classification_category)

        train_size, val_size, test_size = self.train_val_test_split
        train_dataset, val_dataset, test_dataset = random_split(ecg_dataset, [train_size, val_size, test_size])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        # return train dataloader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        # return validation dataloader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        # return test dataloader
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def teardown(self, stage=None):
        # clean up after fit or test
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "ecg_npz.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
