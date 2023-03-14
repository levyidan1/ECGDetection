from pathlib import Path

import pytest
import torch
import os

from PIL import Image

from src.data.mnist_datamodule import MNISTDataModule
from src.data.ny_datamodule import NYDataModule
from src.data.ny_datamodule import NYDataset



# @pytest.mark.parametrize("batch_size", [32, 128])
# def test_mnist_datamodule(batch_size):
#     data_dir = "data/"

#     dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
#     dm.prepare_data()

#     assert not dm.data_train and not dm.data_val and not dm.data_test
#     assert Path(data_dir, "MNIST").exists()
#     assert Path(data_dir, "MNIST", "raw").exists()

#     dm.setup()
#     assert dm.data_train and dm.data_val and dm.data_test
#     assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

#     num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
#     assert num_datapoints == 70_000

#     batch = next(iter(dm.train_dataloader()))
#     x, y = batch
#     assert len(x) == batch_size
#     assert len(y) == batch_size
#     assert x.dtype == torch.float32
#     assert y.dtype == torch.int64

@pytest.mark.parametrize("batch_size", [32, 128])
def test_ny_datamodule(batch_size):
    data_dir = "data/NY/"
    assert Path(data_dir).exists()

    dm = NYDataModule(data_dir=data_dir, batch_size=batch_size)

    dm.setup()
    assert dm.train_dataset and dm.val_dataset and dm.test_dataset
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
    assert num_datapoints == 79_226

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.short
    assert x.shape == torch.Size([batch_size, 4, 1275, 1650])
    assert y.shape == (batch_size,)

    dataset = NYDataset(data_dir, os.path.join(data_dir, "labels.csv"))
    assert dataset[0][1] == 62
    assert dataset[1][1] == 59
    assert dataset[2][1] == 97
    assert dataset[-1][1] == 76
    
    assert Image.open(os.path.join(data_dir, f"{78558}.png")) == dataset[0][0]
    assert Image.open(os.path.join(data_dir, f"{44646}.png")) == dataset[-1][0]

    assert x[0] != dataset[0][0]
    



