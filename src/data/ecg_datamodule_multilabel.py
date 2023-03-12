from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms

import os
import pandas as pd
import numpy as np
from PIL import Image

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
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.csv_file = pd.read_csv(os.path.join(self.root_dir, 'labels.csv'))

        self.img_names = self.csv_file['filename'].tolist()

        self.labels = self.csv_file.drop('filename', axis=1).values

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_names[idx] + '.png')
        img = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img, label


class ECGDataModule(LightningDataModule):
    """LightningDataModule for an ECG dataset.
    A DataModule implements 5 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test
    """

    def __init__(
            self,
            data_dir: str = "data/",
            train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            dataset=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset = dataset

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    @property
    def num_classes(self):
        return 2

    def setup(self, stage=None):
        # split into train/val/test
        if self.dataset is None:
            # use ImageFolder dataset
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

            dataset = ImageFolder(os.path.join(self.data_dir, 'BR', 'images'), transform=train_transform)

            self.dataset = dataset

        train_size, val_size, test_size = self.train_val_test_split

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "default.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
