from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T

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


class NYDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None, label=None):
        self.data_dir = data_dir
        self.files_in_database = os.listdir(self.data_dir)
        self.files_in_database = [i for i in self.files_in_database if i.endswith('.png')]
        self.files_in_database = sorted(self.files_in_database, key=lambda x: int(x.split('.')[0]))
        self.label = label
        # self.csv_file = pd.read_csv(csv_file, index_col=0, on_bad_lines='warn')
        # coloumns: id	
        self.csv_file = pd.read_csv(csv_file, index_col=0, dtype={'Dx1': str,
                                                                  'Dx2': str,
                                                                  'Dx3': str,
                                                                  'Dx4': str,
                                                                  'Dx5': str,
                                                                  'Dx6': str,
                                                                  'Dx7': str,
                                                                  'Dx8': str,
                                                                  'Dx9': str,
                                                                  'Dx10': str,
                                                                  'Dx11': str,
                                                                  'Dx12': str,})
        self.labels = self.upload_labels()
        self.transform = transform

    def upload_labels(self):
        """Labels are in the form of a dictionary with filename as key and labels as value
        """
        labels = {}
        for filename in self.files_in_database:
            filename_without_extension = int(filename.split('.')[0])
            current_labels = self.csv_file.loc[filename_without_extension][['Dx1',
                                                      'Dx2',
                                                      'Dx3',
                                                      'Dx4',
                                                      'Dx5',
                                                      'Dx6',
                                                      'Dx7',
                                                      'Dx8',
                                                      'Dx9',
                                                      'Dx10',
                                                      'Dx11',
                                                      'Dx12']].values
            if self.label is not None:
                if self.label in current_labels:
                    labels[filename_without_extension] = True
                else:
                    labels[filename_without_extension] = False
            else:
                labels[filename_without_extension] = current_labels
        return labels

    def __len__(self):
        return len(self.files_in_database)

    def __getitem__(self, idx):
        file_id = int(self.files_in_database[idx].split('.')[0])
        image_path = os.path.join(self.data_dir, f"{file_id}.png")
        image = Image.open(image_path)
        image = image.crop((0, 270, 1650, 1150))
        if image.mode == 'RGBA':
            # Convert RGBA to RGB format
            image = image.convert('RGB')

        ventricular_rate = self.csv_file.loc[file_id]['Ventricular Rate']
        # label = torch.tensor(ventricular_rate)
        label = self.labels[file_id]
        # label = torch.tensor(labels)
        if self.transform:
            image = self.transform(image)
        return image, label

class NYDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/",
            train_val_test_split: Tuple[int, int, int] = (62_000, 5_600, 11_636),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            label: str = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = T.Compose([
            # # T.Resize(224),
            T.ToTensor(),
            # T.Normalize((0.5,), (0.5,))
            
        ])
        self.label = label

    def prepare_data(self):
        # download data, pre-process, split, save to disk, etc...
        pass

    def setup(self, stage=None):
        # load data, set variables, etc...
        ecg_dataset = NYDataset(self.data_dir, os.path.join(self.data_dir, "labels.csv"), transform=self.transform, label=self.label)
        
        train_size, val_size, test_size = self.train_val_test_split
        train_dataset, val_dataset, test_dataset = random_split(ecg_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "ny.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
